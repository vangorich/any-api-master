import json
import httpx
import logging
import time
from typing import Optional, Any, Dict
from fastapi import Request, Response, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.services.universal_converter import universal_converter
from app.services.gemini_service import gemini_service
from app.services.claude_service import claude_service
from app.models.key import OfficialKey
from app.models.log import Log
from app.models.user import User

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
        request_body: bytes
    ) -> Log:
        """Creates and returns an initial Log object without saving it."""
        if not official_key_obj:
            return None
        
        # 粗略计算输入token
        input_tokens = len(request_body) // 4
        
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
        status_code: int,
        latency: float,
        output_tokens: int,
        ttft: Optional[float] = None
    ):
        """Finalizes and saves the log entry."""
        if not log_entry:
            return

        try:
            log_entry.status_code = status_code
            log_entry.status = "ok" if 200 <= status_code < 300 else "error"
            log_entry.latency = latency
            log_entry.ttft = ttft if ttft is not None else latency
            log_entry.output_tokens = output_tokens
            
            db.add(log_entry)
            await db.commit()
            print(f"DEBUG: [Proxy] Finalized log for key ID {log_entry.official_key_id}")
        except Exception as e:
            logger.error(f"[Proxy] Failed to finalize log: {e}", exc_info=True)


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
        
        if provider == "openai":
            base_url = "https://api.openai.com"
            if not path.startswith("/"):
                target_path = f"/v1/{path}"
            else:
                target_path = path
                
        elif provider == "gemini":
            base_url = "https://generativelanguage.googleapis.com"
            if not path.startswith("/"):
                target_path = f"/v1beta/{path}"
            else:
                target_path = path

        elif provider == "claude":
            base_url = "https://api.anthropic.com"
            if not path.startswith("/"):
                target_path = f"/v1/{path}"
            else:
                target_path = path

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
        try:
            request_data = json.loads(body)
            request_model = request_data.get("model", "unknown")
            is_stream = request_data.get("stream", False)
        except:
            pass

        log_entry = await self._create_initial_log(db, key_obj, user, request_model, is_stream, body)

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
                await self._finalize_log(db, log_entry, response.status_code, latency, 0)
                return Response(content=error_content, status_code=response.status_code, media_type=response.headers.get("content-type"))

            excluded_response_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
            response_headers = {k: v for k, v in response.headers.items() if k.lower() not in excluded_response_headers}

            # 对于流式响应，我们牺牲部分指标的精确性以换取日志记录的可靠性
            # 不在生成器内部提交数据库，避免会话关闭问题
            async def stream_generator(response: httpx.Response):
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                finally:
                    await response.aclose()

            return StreamingResponse(
                stream_generator(response),
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type")
            )
            
        except httpx.RequestError as e:
            logger.error(f"Proxy request failed: {e}")
            latency = time.time() - start_time
            await self._finalize_log(db, log_entry, 502, latency, 0)
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
            target_url = f"https://generativelanguage.googleapis.com/v1beta/{model}:{action}"
            target_method = "POST"
            
            if "model" in converted_body: del converted_body["model"]
            if "stream" in converted_body: del converted_body["stream"]

        elif target_provider == "claude":
            target_url = "https://api.anthropic.com/v1/messages"
            target_method = "POST"
            current_model = converted_body.get("model", "")
            if not current_model.startswith("claude-"):
                converted_body["model"] = "claude-3-5-sonnet-20240620"
            
        elif target_provider == "openai":
            target_url = "https://api.openai.com/v1/chat/completions"
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
        
        log_entry = await self._create_initial_log(db, key_obj, user, original_model, stream, body_bytes)

        try:
            req = client.build_request(
                target_method, target_url, headers=headers, json=converted_body, timeout=120.0
            )
            response = await client.send(req, stream=True)
            
            if response.status_code >= 400:
                error_content = await response.aread()
                await response.aclose()
                latency = time.time() - start_time
                await self._finalize_log(db, log_entry, response.status_code, latency, 0)
                return Response(content=error_content, status_code=response.status_code)

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
                    final_response, usage = universal_converter.convert_response(resp_json, incoming_format, target_provider, original_model)
                    output_tokens = usage.get("completion_tokens", 0)
                    # input_tokens can also be updated here if available
                    if log_entry and usage.get("prompt_tokens"):
                        log_entry.input_tokens = usage.get("prompt_tokens")
                    
                    await self._finalize_log(db, log_entry, response.status_code, latency, output_tokens)
                    return JSONResponse(final_response)
                        
                except json.JSONDecodeError:
                    await self._finalize_log(db, log_entry, response.status_code, latency, 0)
                    return Response(content=resp_content, status_code=response.status_code)

        except httpx.RequestError as e:
            logger.error(f"Conversion request failed: {e}")
            latency = time.time() - start_time
            await self._finalize_log(db, log_entry, 502, latency, 0)
            raise HTTPException(status_code=502, detail=f"Upstream service error: {str(e)}")

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
        except Exception as e:
            logger.error(f"[Proxy] Stream conversion error: {e}", exc_info=True)
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