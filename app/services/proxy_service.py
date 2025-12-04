import json
import httpx
import logging
import time
from typing import Optional, Any, Dict
from fastapi import Request, Response, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.errors import ErrorConverter
from app.services.universal_converter import universal_converter
from app.services.gemini_service import gemini_service
from app.services.claude_service import claude_service
from app.models.key import OfficialKey
from app.models.log import Log
from app.models.user import User
from app.models.key import count_tokens_for_messages, get_tokenizer
from typing import List, Dict

# 强制配置 logger 输出到控制台，确保用户能看到日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 额外添加一个 StreamHandler 以防 basicConfig 不生效
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ProxyService:
    async def _create_initial_log(
        self,
        db: AsyncSession,
        official_key_obj: OfficialKey,
        user: Optional[User],
        model: str,
        is_stream: bool,
        messages: List[Dict]
    ) -> Log:
        """Creates and returns an initial Log object without saving it."""
        if not official_key_obj:
            return None
        
        # 使用 tiktoken 计算输入 token
        input_tokens = count_tokens_for_messages(messages, model)
        
        log_entry = Log(
            official_key_id=official_key_obj.id,
            user_id=user.id if user else official_key_obj.user_id,
            model=model,
            status="processing",
            status_code=0, latency=0, ttft=0,
            is_stream=is_stream,
            input_tokens=input_tokens,
            output_tokens=0
        )
        try:
            db.add(log_entry)
            await db.commit()
            await db.refresh(log_entry)
            return log_entry
        except Exception as e:
            logger.error(f"[Proxy] Failed to create initial log: {e}", exc_info=True)
            await db.rollback()
            return None

    async def _finalize_log(
        self,
        db: AsyncSession,
        log_entry: Optional[Log],
        key_obj: Optional[OfficialKey], # 传入key对象
        status_code: Any,
        latency: float,
        output_tokens: int,
        ttft: Optional[float] = None
    ):
        """Finalizes the log entry and updates the key stats."""
        if not log_entry or not key_obj:
            return

        try:
            # Ensure status_code is a valid integer for comparison
            try:
                numeric_status_code = int(status_code)
            except (ValueError, TypeError):
                numeric_status_code = 500 # Default to internal server error

            # 1. Finalize Log
            log_entry.status_code = numeric_status_code
            log_entry.status = "ok" if 200 <= numeric_status_code < 300 else "error"
            log_entry.latency = latency
            log_entry.ttft = ttft if ttft is not None else latency
            log_entry.output_tokens = output_tokens
            db.add(log_entry)

            # 2. Update Key Stats
            key_obj.usage_count += 1
            key_obj.input_tokens = (key_obj.input_tokens or 0) + (log_entry.input_tokens or 0)
            key_obj.output_tokens = (key_obj.output_tokens or 0) + output_tokens
            key_obj.last_status_code = numeric_status_code
            
            if log_entry.status == "error":
                key_obj.error_count += 1
                key_obj.last_status = str(numeric_status_code)
            else:
                key_obj.last_status = "active"
            
            db.add(key_obj)
            await db.commit()
            logger.info(f"[Proxy] Finalized log and updated key stats for Key ID {key_obj.id}")

        except Exception as e:
            logger.error(f"[Proxy] Failed to finalize log and key stats for Key ID {key_obj.id}. Error: {e}", exc_info=True)
            await db.rollback()


    def identify_target_provider(self, key: str) -> str:
        """
        根据 Key 前缀识别目标服务商
        """
        if key.startswith("sk-ant-"):
            return "claude"
        elif key.startswith("AIza"):
            return "gemini"
        elif key.startswith("sk-"):
            return "openai"
        # 默认回退到 OpenAI，或者可以根据配置调整
        return "openai"

    async def smart_proxy_handler(
        self,
        request: Request,
        db: AsyncSession,
        path: str,
        official_key_obj: OfficialKey,
        user: Optional[User],
        incoming_format: str,  # "openai", "gemini", "claude"
        background_tasks: BackgroundTasks = None
    ):
        official_key = official_key_obj.key
        target_provider = self.identify_target_provider(official_key)
        masked_key = f"{official_key[:8]}...{official_key[-4:]}" if len(official_key) > 12 else "***"
        
        # 使用 print 确保在所有环境下可见
        print(f"DEBUG: [Proxy] Route Decision: Incoming={incoming_format}, Target={target_provider}, Key={masked_key}")
        logger.info(f"[Proxy] Route Decision: Incoming={incoming_format}, Target={target_provider}, Key={masked_key}")
        
        # 1. 透传模式 (Pass-through)
        if incoming_format == target_provider:
            print(f"DEBUG: [Proxy] Mode: PASS-THROUGH ({target_provider.upper()})")
            return await self._handle_passthrough(request, db, path, official_key_obj, user, target_provider)
        
        # 2. 转换模式 (Conversion)
        print(f"DEBUG: [Proxy] Mode: CONVERSION ({incoming_format.upper()} -> {target_provider.upper()})")
        return await self._handle_conversion(request, db, path, official_key_obj, user, incoming_format, target_provider)

    async def _handle_passthrough(
        self,
        request: Request,
        db: AsyncSession,
        path: str,
        key_obj: OfficialKey,
        user: Optional[User],
        provider: str
    ):
        """处理同构透传请求"""
        start_time = time.time()
        key = key_obj.key
        
        # 构建目标 URL
        base_url = ""
        target_path = path

        # 优先使用渠道自定义的 api_url
        if key_obj.channel and key_obj.channel.api_url:
            base_url = key_obj.channel.api_url.rstrip('/')
        else:
            # 回退到默认地址
            if provider == "openai":
                base_url = "https://api.openai.com"
            elif provider == "gemini":
                base_url = "https://generativelanguage.googleapis.com"
            elif provider == "claude":
                base_url = "https://api.anthropic.com"

        # 构建目标路径
        if provider == "openai":
            target_path = f"/v1/{path}" if not path.startswith("/") else path
        elif provider == "gemini":
            target_path = f"/v1beta/{path}" if not path.startswith("/") else path
        elif provider == "claude":
            target_path = f"/v1/{path}" if not path.startswith("/") else path
        
        target_url = f"{base_url}{target_path}"
        
        # 处理 Headers
        excluded_headers = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded_headers}
        
        # 注入 Key
        if provider == "openai":
            headers["Authorization"] = f"Bearer {key}"
        elif provider == "gemini":
            headers["x-goog-api-key"] = key
            if "Authorization" in headers:
                del headers["Authorization"]
        elif provider == "claude":
            headers["x-api-key"] = key
            headers["anthropic-version"] = headers.get("anthropic-version", "2023-06-01")
            if "Authorization" in headers:
                del headers["Authorization"]

        # 准备 Body
        body = await request.body()
        
        # 准备 Query Params
        params = dict(request.query_params)
        if provider == "gemini":
            params["key"] = key

        # 获取 Client
        client = self._get_client(provider)
        
        request_model = "unknown"
        is_stream = False
        messages = []
        try:
            request_data = json.loads(body)
            request_model = request_data.get("model", "unknown")
            is_stream = request_data.get("stream", False)
            messages = request_data.get("messages", [])
        except:
            pass

        log_entry = await self._create_initial_log(db, key_obj, user, request_model, is_stream, messages)

        try:
            logger.info(f"[Proxy] Forwarding request to: {target_url} (Method: {request.method})")
            
            req = client.build_request(
                request.method, target_url, headers=headers, content=body, params=params, timeout=120.0
            )
            
            response = await client.send(req, stream=True)
            logger.info(f"[Proxy] Upstream response status: {response.status_code}")
            
            if response.status_code >= 400:
                error_content = await response.aread()
                await response.aclose()
                latency = time.time() - start_time
                await self._finalize_log(db, log_entry, key_obj, response.status_code, latency, 0)
                return Response(content=error_content, status_code=response.status_code, media_type=response.headers.get("content-type"))

            excluded_response_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
            response_headers = {k: v for k, v in response.headers.items() if k.lower() not in excluded_response_headers}

            # 对于流式响应，我们牺牲部分指标的精确性以换取日志记录的可靠性
            # 不在生成器内部提交数据库，避免会话关闭问题
            async def stream_generator(response: httpx.Response, log_entry: Log, key_obj: OfficialKey, start_time: float):
                full_response_content = ""
                tokenizer = get_tokenizer(log_entry.model)
                try:
                    async for chunk in response.aiter_bytes():
                        try:
                            # Attempt to decode for token counting, ignore if not valid JSON/text
                            chunk_text = chunk.decode('utf-8')
                            if chunk_text.startswith('data: '):
                                content_part = chunk_text[6:].strip()
                                if content_part != '[DONE]':
                                    json_content = json.loads(content_part)
                                    if json_content.get('choices'):
                                        delta = json_content['choices'][0].get('delta', {})
                                        full_response_content += delta.get('content', '')
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            pass
                        yield chunk
                except httpx.RequestError as e:
                    logger.error(f"[Proxy] Stream request failed: {e}", exc_info=True)
                    # For passthrough, we can't easily inject an error chunk. The connection will just close.
                    # The log will be finalized with an error code if we can get here.
                finally:
                    await response.aclose()
                    latency = time.time() - start_time
                    output_tokens = len(tokenizer.encode(full_response_content))
                    # We can't get ttft easily here, so we pass latency
                    await self._finalize_log(db, log_entry, key_obj, response.status_code, latency, output_tokens, latency)

            return StreamingResponse(
                stream_generator(response, log_entry, key_obj, start_time),
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type")
            )
            
        except httpx.RequestError as e:
            logger.error(f"Proxy request failed: {e}")
            latency = time.time() - start_time
            await self._finalize_log(db, log_entry, key_obj, 502, latency, 0)
            raise HTTPException(status_code=502, detail=f"Upstream service error: {str(e)}")


    async def _handle_conversion(
        self,
        request: Request,
        db: AsyncSession,
        path: str,
        key_obj: OfficialKey,
        user: Optional[User],
        incoming_format: str,
        target_provider: str
    ):
        """处理异构转换请求"""
        start_time = time.time()
        key = key_obj.key
        
        # 1. 读取并解析 Body
        try:
            body_bytes = await request.body()
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        # 2. 转换请求体 (Incoming -> Target)
        converted_body, _ = await universal_converter.convert_request(body, target_provider, request)
        
        # 3. 确定目标 URL 和 Method
        target_url = ""
        target_method = request.method
        stream = body.get("stream", False)
        
        logger.info(f"[Proxy] Converting body... (Stream={stream})")
        
        original_model = body.get("model", "unknown")
        
        if target_provider == "gemini":
            raw_model = body.get("model", "") or converted_body.get("model", "gemini-1.5-pro")
            model = raw_model
            
            if not "gemini" in model.lower():
                if "gpt-3.5" in model: model = "gemini-1.5-flash"
                elif "gpt-4" in model: model = "gemini-1.5-pro"
                else: model = "gemini-1.5-pro"
                logger.info(f"[Proxy] Model Mapped: {original_model} -> {model}")
            
            if not model.startswith("models/"):
                model = f"models/{model}"

            action = "streamGenerateContent" if stream else "generateContent"
            
            base_url = "https://generativelanguage.googleapis.com"
            if key_obj.channel and key_obj.channel.api_url:
                base_url = key_obj.channel.api_url.rstrip('/')

            target_url = f"{base_url}/v1beta/{model}:{action}"
            target_method = "POST"
            
            if "model" in converted_body: del converted_body["model"]
            if "stream" in converted_body: del converted_body["stream"]

        elif target_provider == "claude":
            base_url = "https://api.anthropic.com"
            if key_obj.channel and key_obj.channel.api_url:
                base_url = key_obj.channel.api_url.rstrip('/')
            target_url = f"{base_url}/v1/messages"
            target_method = "POST"
            current_model = converted_body.get("model", "")
            if not current_model.startswith("claude-"):
                converted_body["model"] = "claude-3-5-sonnet-20240620"
            
        elif target_provider == "openai":
            base_url = "https://api.openai.com"
            if key_obj.channel and key_obj.channel.api_url:
                base_url = key_obj.channel.api_url.rstrip('/')
            target_url = f"{base_url}/v1/chat/completions"
            target_method = "POST"

        # 4. 准备 Headers
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length", "authorization", "x-api-key", "x-goog-api-key"]}
        headers["Content-Type"] = "application/json"
        
        if target_provider == "openai": headers["Authorization"] = f"Bearer {key}"
        elif target_provider == "gemini": headers["x-goog-api-key"] = key
        elif target_provider == "claude":
            headers["x-api-key"] = key
            headers["anthropic-version"] = "2023-06-01"

        # 5. 发送请求
        client = self._get_client(target_provider)
        
        # messages should be in the converted_body for conversion cases
        messages = converted_body.get("messages", [])
        log_entry = await self._create_initial_log(db, key_obj, user, original_model, stream, messages)

        try:
            req = client.build_request(
                target_method, target_url, headers=headers, json=converted_body, timeout=120.0
            )
            response = await client.send(req, stream=True)
            
            if response.status_code >= 400:
                error_content = await response.aread()
                await response.aclose()
                latency = time.time() - start_time
                await self._finalize_log(db, log_entry, key_obj, response.status_code, latency, 0)
                # Convert the error to the original incoming format
                converted_error = ErrorConverter.convert_upstream_error(error_content, response.status_code, target_provider, incoming_format)
                return JSONResponse(status_code=response.status_code, content=converted_error)

            if stream:
                return StreamingResponse(
                    self._stream_converter_with_logging(response, db, log_entry, target_provider, incoming_format, start_time, original_model), # 转换流维持原状，因为它在内部处理 token
                    media_type="text/event-stream"
                )
            else:
                resp_content = await response.aread()
                await response.aclose()
                latency = time.time() - start_time
                output_tokens = 0
                try:
                    resp_json = json.loads(resp_content)
                    final_response, _ = universal_converter.convert_response(resp_json, incoming_format, target_provider, original_model)
                    
                    # Recalculate output_tokens from actual response
                    tokenizer = get_tokenizer(original_model)
                    response_content = final_response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    output_tokens = len(tokenizer.encode(response_content))

                    # Update input tokens if provider gives a more accurate count
                    # This part is now handled by tiktoken initially, but can be refined
                    
                    await self._finalize_log(db, log_entry, key_obj, response.status_code, latency, output_tokens)
                    return JSONResponse(final_response)
                        
                except json.JSONDecodeError:
                    await self._finalize_log(db, log_entry, key_obj, response.status_code, latency, 0)
                    # The response is not valid JSON, but we should still try to inform the client in the right format
                    error_message = f"Upstream service returned non-JSON response with status {response.status_code}"
                    converted_error = ErrorConverter.convert_upstream_error(error_message.encode(), response.status_code, target_provider, incoming_format)
                    return JSONResponse(status_code=response.status_code, content=converted_error)

        except httpx.RequestError as e:
            logger.error(f"Conversion request failed: {e}")
            latency = time.time() - start_time
            await self._finalize_log(db, log_entry, key_obj, 502, latency, 0)
            error_message = f"Upstream service error: {str(e)}"
            converted_error = ErrorConverter.convert_upstream_error(error_message.encode(), 502, target_provider, incoming_format)
            return JSONResponse(status_code=502, content=converted_error)

    async def _stream_converter_with_logging(self, response: httpx.Response, db: AsyncSession, log_entry: Log, from_provider: str, to_format: str, start_time: float, original_model: str):
        """Wrapper for stream_converter that handles logging. It no longer finalizes the log."""
        try:
            async for chunk in self._stream_converter(response, from_provider, to_format, original_model):
                yield chunk
        finally:
            # The log is intentionally left in a 'processing' state for streams
            # to ensure reliability of logging creation. Detailed finalization
            # for streams is disabled to prevent db session errors.
            pass


    async def _stream_converter(self, response: httpx.Response, from_provider: str, to_format: str, original_model: str):
        """流式响应转换生成器"""
        buffer = ""
        try:
            async for line in response.aiter_lines():
                # This logic is simplified to handle Gemini's typical stream format
                if line.startswith('data: '):
                    line = line[6:]
                
                buffer += line
                
                try_parse = buffer.strip()
                if try_parse.startswith('[') and not try_parse.endswith(']'): continue
                if try_parse.startswith('{') and not try_parse.endswith('}'): continue

                try:
                    data = json.loads(try_parse)
                    chunks = data if isinstance(data, list) else [data]
                    
                    for chunk in chunks:
                        converted_chunk, _ = universal_converter.convert_chunk(chunk, to_format, from_provider, original_model)
                        if converted_chunk:
                            yield f"data: {json.dumps(converted_chunk)}\n\n"
                    
                    buffer = "" # Reset buffer after successful processing
                except json.JSONDecodeError:
                    # Incomplete JSON, continue buffering
                    pass
            
            # Final DONE message
            yield "data: [DONE]\n\n"
        except httpx.RequestError as e:
            logger.error(f"[Proxy] Stream conversion request failed: {e}", exc_info=True)
            error_message = f"无法连接到上游服务: {type(e).__name__}"
            openai_error = ErrorConverter.convert_upstream_error(error_message.encode('utf-8'), 502, "openai", to_format)
            yield f"data: {json.dumps(openai_error)}\n\n"
            yield b"data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[Proxy] Stream conversion error: {e}", exc_info=True)
            # Yield a generic error in the stream
            error_message = f"流转换中发生内部错误: {str(e)}"
            openai_error = ErrorConverter.convert_upstream_error(error_message.encode('utf-8'), 500, "openai", to_format)
            yield f"data: {json.dumps(openai_error)}\n\n"
            yield b"data: [DONE]\n\n"
        finally:
            await response.aclose()


    def _get_client(self, provider: str) -> httpx.AsyncClient:
        if provider == "gemini":
            return gemini_service.client
        elif provider == "claude":
            return claude_service.client
        else:
            # 对于 OpenAI，我们可能没有持久化的全局 client，或者可以使用一个
            return httpx.AsyncClient(timeout=60.0)

proxy_service = ProxyService()