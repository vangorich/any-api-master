import json
import time
import httpx
import logging
from typing import AsyncGenerator, Tuple, List, Dict, Any, Optional
from sqlalchemy.exc import PendingRollbackError
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.services.universal_converter import universal_converter, ApiFormat
from app.core.errors import ErrorConverter
from app.services.variable_service import variable_service
from app.services.regex_service import regex_service
from app.models.user import User
from app.models.key import ExclusiveKey, OfficialKey
from app.models.preset import Preset
from app.models.regex import RegexRule
from app.models.preset_regex import PresetRegexRule
from app.models.log import Log
from app.models.key import count_tokens_for_messages, get_tokenizer
from app.core.config import settings
from sqlalchemy.future import select
from fastapi import Request

import datetime

logger = logging.getLogger(__name__)

class ChatProcessor:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0)

    async def process_request(
        self,
        request: Request,
        db: AsyncSession,
        official_key: OfficialKey, # Changed from str to OfficialKey object
        exclusive_key: ExclusiveKey,
        user: User,
        model_override: str = None,
        original_format: ApiFormat = "openai" # Assume openai by default
    ) -> Tuple[Dict[str, Any], int, ApiFormat]:
        start_time = time.time()
        body_bytes = await request.body()
        body = json.loads(body_bytes)
        # 仅在DEBUG模式下打印完整请求体
        logger.debug(f"客户端原始请求体: {json.dumps(body, indent=2, ensure_ascii=False)}")
        
        # The target format is determined by the channel configuration
        target_format = "gemini" # Default value
        if official_key.channel and official_key.channel.type:
            target_format = official_key.channel.type.lower()

        # Convert the incoming request (original_format) to our internal standard (openai)
        # The original_format from the request body detection is less reliable than the one passed from the endpoint.
        # We will use the one passed from the endpoint.
        converted_body, _ = await universal_converter.convert_request(body, "openai", request=request)
        
        logger.info(f"渠道 '{exclusive_key.name}' (ID: {exclusive_key.channel_id}) 接收到客户端请求: {request.url}")
        logger.info(f"请求转换: 客户端格式 ({original_format}) -> 内部格式 (openai)")
        logger.info(f"上游转换: 内部格式 (openai) -> 目标格式 ({target_format})")
        
        if model_override:
            converted_body["model"] = model_override
            
        openai_request = ChatCompletionRequest(**converted_body)
        
        log_entry = await self._create_initial_log(db, exclusive_key, official_key, user, openai_request.model, openai_request.stream, [msg.dict() for msg in openai_request.messages])

        presets, regex_rules, preset_regex_rules = await self._load_context(db, exclusive_key)
        openai_request = self._apply_preprocessing(openai_request, presets, regex_rules, preset_regex_rules)
        final_payload, _ = await universal_converter.convert_request(openai_request.dict(), target_format)
        logger.debug(f"发送到上游的最终请求体: {json.dumps(final_payload, indent=2, ensure_ascii=False)}")
        
        if openai_request.stream:
            return self._logged_stream_generator(
                self.stream_chat_completion(
                    final_payload, target_format, original_format, openai_request.model,
                    official_key, regex_rules, preset_regex_rules
                ),
                db=db,
                log_entry=log_entry,
                official_key=official_key, # Pass official_key object
                start_time=start_time
            )
        else:
            result, status_code, _ = await self.non_stream_chat_completion(
                final_payload, target_format, original_format, openai_request.model,
                official_key, regex_rules, preset_regex_rules
            )
            latency = time.time() - start_time
            # Re-calculate output tokens from the actual response content
            tokenizer = get_tokenizer(openai_request.model)
            response_content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            output_tokens = len(tokenizer.encode(response_content))
            await self._finalize_log(db, log_entry, official_key, status_code, latency, output_tokens)
            return result, status_code, original_format

    async def _load_context(self, db: AsyncSession, exclusive_key: ExclusiveKey) -> Tuple[List, List, List]:
        """从数据库加载预设和正则规则"""
        presets, regex_rules, preset_regex_rules = [], [], []
        if exclusive_key.preset_id:
            result = await db.execute(select(Preset).filter(Preset.id == exclusive_key.preset_id))
            preset = result.scalars().first()
            if preset:
                await db.refresh(preset)
                presets.append({"id": preset.id, "name": preset.name, "content": preset.content})
                result = await db.execute(select(PresetRegexRule).filter(PresetRegexRule.preset_id == preset.id, PresetRegexRule.is_active == True))
                preset_regex_rules = result.scalars().all()
        
        if exclusive_key.enable_regex:
            result = await db.execute(select(RegexRule).filter(RegexRule.is_active == True))
            regex_rules = result.scalars().all()
            
        return presets, regex_rules, preset_regex_rules

    def _apply_preprocessing(
        self,
        request: ChatCompletionRequest,
        presets: List,
        global_rules: List,
        local_rules: List
    ) -> ChatCompletionRequest:
        """应用所有前置处理: 全局正则 -> 局部正则 -> 预设 -> 变量"""
        # 1. 应用正则
        global_pre = [r for r in global_rules if r.type == "pre"]
        local_pre = [r for r in local_rules if r.type == "pre"]
        for msg in request.messages:
            if isinstance(msg.content, str):
                msg.content = regex_service.process(msg.content, global_pre)
                msg.content = regex_service.process(msg.content, local_pre)

        # 2. 应用预设
        if presets and request.messages:
            for preset in presets:
                try:
                    content_str = preset.get('content')
                    if not content_str: continue
                    preset_content = json.loads(content_str) if isinstance(content_str, str) else content_str
                    items = preset_content.get('preset') or preset_content.get('items', [])
                    if not items: continue

                    sorted_items = sorted(items, key=lambda x: x.get('order', 0))
                    processed_messages, original_messages = [], list(request.messages)
                    last_user_message = next((msg for msg in reversed(original_messages) if msg.role == 'user'), None)
                    history_messages = [msg for msg in original_messages if msg != last_user_message]
                    
                    for item in sorted_items:
                        if not item.get('enabled', True): continue
                        item_type = item.get('type', 'normal')
                        if item_type == 'normal':
                            processed_messages.append({'role': item.get('role', 'system'), 'content': item.get('content', '')})
                        elif item_type == 'user_input' and last_user_message:
                            processed_messages.append({'role': last_user_message.role, 'content': last_user_message.content})
                        elif item_type == 'history':
                            processed_messages.extend([{'role': h.role, 'content': h.content if isinstance(h.content, str) else str(h.content)} for h in history_messages])
                    
                    if processed_messages:
                        request.messages = [ChatMessage(**msg) for msg in processed_messages]
                except Exception as e:
                    logger.error(f"预设处理失败: {e}")
                    continue

        # 3. 应用变量
        for msg in request.messages:
            if isinstance(msg.content, str):
                msg.content = variable_service.parse_variables(msg.content)
        
        return request

    def _apply_postprocessing(self, content: str, global_rules: List, local_rules: List) -> str:
        """应用所有后置处理: 局部正则 -> 全局正则"""
        local_post = [r for r in local_rules if r.type == "post"]
        global_post = [r for r in global_rules if r.type == "post"]
        content = regex_service.process(content, local_post)
        content = regex_service.process(content, global_post)
        return content

    async def _create_initial_log(self, db: AsyncSession, exclusive_key: ExclusiveKey, official_key: OfficialKey, user: User, model: str, is_stream: bool, messages: List[Dict[str, Any]]) -> Log:
        # Use tiktoken for input tokens
        input_tokens = count_tokens_for_messages(messages, model)
        
        log_entry = Log(
            exclusive_key_id=exclusive_key.id,
            official_key_id=official_key.id,
            user_id=user.id,
            model=model,
            status="processing",
            status_code=0,
            latency=0, ttft=0,
            is_stream=is_stream,
            input_tokens=input_tokens,
            output_tokens=0
        )
        return log_entry

    async def _finalize_log(self, db: AsyncSession, log_entry: Optional[Log], official_key: OfficialKey, status_code: Any, latency: float, output_tokens: int, ttft: Optional[float] = None):
        if not log_entry or not official_key:
            return

        try:
            # Ensure status_code is a valid integer for comparison
            try:
                numeric_status_code = int(status_code)
            except (ValueError, TypeError):
                numeric_status_code = 500  # Default to internal server error if conversion fails
            
            # 1. Finalize Log
            log_entry.status_code = numeric_status_code
            log_entry.status = "ok" if 200 <= numeric_status_code < 300 else "error"
            log_entry.latency = latency
            log_entry.ttft = ttft if ttft is not None else latency
            log_entry.output_tokens = output_tokens
            db.add(log_entry)

            # 2. Update Official Key Stats
            official_key.usage_count += 1
            official_key.input_tokens = (official_key.input_tokens or 0) + (log_entry.input_tokens or 0)
            official_key.output_tokens = (official_key.output_tokens or 0) + output_tokens
            official_key.last_status_code = numeric_status_code
            
            if log_entry.status == "error":
                official_key.error_count += 1
                official_key.last_status = str(status_code)
            else:
                official_key.last_status = "active"

            db.add(official_key)
            await db.commit()
            logger.info(f"[ChatProcessor] 日志已归档，Key 统计已更新 (Official Key ID {official_key.id})")

        except PendingRollbackError as e:
            # 当事务因客户端取消等原因回滚时，会触发此异常。
            # 我们先回滚以清理会话，然后重新尝试提交日志。
            key_id = log_entry.official_key_id if log_entry else 'N/A'
            logger.warning(f"[ChatProcessor] 检测到待回滚的事务，可能是由客户端断开连接引起。正在尝试恢复日志记录 (Official Key ID: {key_id})")
            await db.rollback()
            try:
                # 重新设置日志状态为错误，因为原始事务失败了
                log_entry.status = "error"
                log_entry.status_code = 499 # Client Closed Request
                db.add(log_entry)
                # Key的统计信息可能不完整，但我们至少记录这次请求
                await db.commit()
                logger.info(f"[ChatProcessor] 成功恢复并记录了客户端断开连接的日志 (Official Key ID: {key_id})")
            except Exception as retry_e:
                logger.error(f"[ChatProcessor] 恢复日志记录失败 (Official Key ID: {key_id})。错误: {retry_e}", exc_info=True)
                await db.rollback()
        except Exception as e:
            # 捕获所有其他异常
            key_id = log_entry.official_key_id if log_entry else 'N/A'
            logger.error(f"[ChatProcessor] 归档日志或更新 Key 统计时发生未知错误 (Official Key ID: {key_id})。错误: {e}", exc_info=True)
            await db.rollback()

    async def non_stream_chat_completion(
        self, payload: Dict, upstream_format: ApiFormat, original_format: ApiFormat, model: str,
        official_key: OfficialKey, global_rules: List, local_rules: List
    ) -> Tuple[Dict, int, ApiFormat]:
        # 动态构建请求
        base_url = ""
        if upstream_format == "gemini":
            base_url = settings.GEMINI_BASE_URL
        elif upstream_format == "claude":
            base_url = "https://api.anthropic.com"
        elif upstream_format == "openai":
            base_url = "https://api.openai.com"

        if official_key.channel and official_key.channel.api_url:
            base_url = official_key.channel.api_url.rstrip('/')

        target_url, headers = self._build_request_params(
            base_url=base_url,
            upstream_format=upstream_format,
            model=model,
            official_key=official_key.key,
            is_stream=False
        )
        
        logger.info(f"请求上游 URL: {target_url}")
        response = await self.client.post(target_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            converted_error = ErrorConverter.convert_upstream_error(response.content, response.status_code, upstream_format, original_format)
            return converted_error, response.status_code, original_format

        # --- Dynamic Response Handling ---
        upstream_response = response.json()
        
        # 1. Convert to a universal internal format (OpenAI format)
        if upstream_format == "gemini":
            internal_response, _ = universal_converter.convert_response(upstream_response, "openai", "gemini", model)
        else: # claude, openai are already compatible or close to openai format
            internal_response, _ = universal_converter.convert_response(upstream_response, "openai", upstream_format, model)

        # 2. Apply post-processing
        if internal_response.get('choices') and internal_response['choices'][0].get('message', {}).get('content'):
            content = internal_response['choices'][0]['message']['content']
            content = self._apply_postprocessing(content, global_rules, local_rules)
            internal_response['choices'][0]['message']['content'] = content

        # 3. Convert back to the original client format if necessary
        if original_format != "openai":
            final_response, _ = universal_converter.convert_response(internal_response, original_format, "openai", model)
            return final_response, 200, original_format
        
        logger.info(f"响应转换: 上游格式 ({upstream_format}) -> 内部格式 (openai) -> 客户端格式 ({original_format})")
        logger.debug(f"从上游接收的原始响应体: {json.dumps(upstream_response, indent=2, ensure_ascii=False)}")
        logger.debug(f"准备发送给客户端的最终响应体: {json.dumps(internal_response, indent=2, ensure_ascii=False)}")
        return internal_response, 200, original_format

    async def _logged_stream_generator(self, generator: AsyncGenerator, db: AsyncSession, log_entry: Log, official_key: OfficialKey, start_time: float):
        ttft = 0.0
        first_chunk = True
        full_response_content = ""
        status_code = 200
        try:
            async for chunk in generator:
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
                
                if chunk.startswith(b'data: '):
                    content_part = chunk[6:].strip()
                    if content_part != b'[DONE]':
                        try:
                            json_content = json.loads(content_part)
                            # Handle cases where json_content might be a list of chunks
                            chunks_to_process = json_content if isinstance(json_content, list) else [json_content]
                            for chunk_item in chunks_to_process:
                                if not isinstance(chunk_item, dict): continue

                                if chunk_item.get('error'):
                                    # Ensure status_code is an integer
                                    code = chunk_item.get('error', {}).get('code', 500)
                                    try:
                                        status_code = int(code)
                                    except (ValueError, TypeError):
                                        status_code = 500 # Fallback for non-integer codes
                                if chunk_item.get('choices'):
                                    delta = chunk_item['choices'][0].get('delta', {})
                                    full_response_content += delta.get('content', '')
                        except json.JSONDecodeError:
                            pass
                yield chunk
        finally:
            latency = time.time() - start_time
            tokenizer = get_tokenizer(log_entry.model)
            output_tokens = len(tokenizer.encode(full_response_content))
            await self._finalize_log(db, log_entry, official_key, status_code, latency, output_tokens, ttft)


    async def stream_chat_completion(
        self, payload: Dict, upstream_format: ApiFormat, original_format: ApiFormat, model: str,
        official_key: OfficialKey, global_rules: List, local_rules: List
    ) -> AsyncGenerator[bytes, None]:
        # 动态构建请求
        base_url = ""
        if upstream_format == "gemini":
            base_url = settings.GEMINI_BASE_URL
        elif upstream_format == "claude":
            base_url = "https://api.anthropic.com"
        elif upstream_format == "openai":
            base_url = "https://api.openai.com"

        if official_key.channel and official_key.channel.api_url:
            base_url = official_key.channel.api_url.rstrip('/')
        
        target_url, headers = self._build_request_params(
            base_url=base_url,
            upstream_format=upstream_format,
            model=model,
            official_key=official_key.key,
            is_stream=True
        )

        logger.info(f"响应转换 (流式): 上游格式 ({upstream_format}) -> 内部格式 (openai) -> 客户端格式 ({original_format})")
        try:
            logger.info(f"请求上游 URL (流式): {target_url}")
            async with self.client.stream("POST", target_url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    converted_error = ErrorConverter.convert_upstream_error(error_content, response.status_code, upstream_format, original_format)
                    yield f"data: {json.dumps(converted_error)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return

                # --- Brace-Matching Buffered Stream Parser ---
                buffer = ""
                brace_counter = 0
                in_string = False
                
                async for raw_chunk in response.aiter_bytes():
                    decoded_chunk = raw_chunk.decode('utf-8')
                    
                    # 逐字符处理以正确处理流中的多个JSON对象
                    for char in decoded_chunk:
                        buffer += char
                        
                        if char == '"' and (len(buffer) == 1 or buffer[-2] != '\\'):
                            in_string = not in_string
                        elif not in_string:
                            if char == '{':
                                brace_counter += 1
                            elif char == '}':
                                brace_counter -= 1
                        
                        # 当括号匹配且缓冲区不为空时，尝试解析
                        if brace_counter == 0 and not in_string and buffer.strip():
                            # 缓冲区可能包含一个或多个完整的JSON对象
                            potential_json = buffer.strip()
                            
                            # 兼容 SSE "data: " 前缀
                            if potential_json.startswith('data:'):
                                potential_json = potential_json[5:].strip()
                            
                            # 处理 SSE "event: " 行 (Claude 等)
                            if "event: " in potential_json:
                                lines = potential_json.split('\n')
                                valid_lines = []
                                for line in lines:
                                    line = line.strip()
                                    if line.startswith("data:"):
                                        valid_lines.append(line[5:].strip())
                                    elif line.startswith("{") or line.startswith("["):
                                        valid_lines.append(line)
                                
                                if valid_lines:
                                    potential_json = "".join(valid_lines)
                                else:
                                    if not potential_json.strip().startswith("{") and not potential_json.strip().startswith("["):
                                         buffer = ""
                                         continue

                            if not potential_json or potential_json == '[DONE]':
                                buffer = ""
                                continue

                            try:
                                upstream_chunks = []
                                # Gemini 流可能会返回一个JSON数组
                                if potential_json.startswith('[') and potential_json.endswith(']'):
                                    parsed_data = json.loads(potential_json)
                                    upstream_chunks.extend(parsed_data)
                                else:
                                    upstream_chunks.append(json.loads(potential_json))

                                for upstream_chunk in upstream_chunks:
                                    # 1. 转换为内部格式 (OpenAI)
                                    internal_chunk, _ = universal_converter.convert_chunk(upstream_chunk, "openai", upstream_format, model)
                                    if not internal_chunk:
                                        continue
                                    
                                    # 2. 应用后处理
                                    if internal_chunk.get('choices') and internal_chunk['choices'][0].get('delta', {}).get('content'):
                                        content = internal_chunk['choices'][0]['delta']['content']
                                        processed_content = self._apply_postprocessing(content, global_rules, local_rules)
                                        internal_chunk['choices'][0]['delta']['content'] = processed_content
                                    
                                    # 3. 转换为最终客户端格式
                                    final_chunk, _ = universal_converter.convert_chunk(internal_chunk, original_format, "openai", model)
                                    if not final_chunk:
                                        continue
                                    
                                    yield f"data: {json.dumps(final_chunk)}\n\n".encode()

                                # 清空缓冲区
                                buffer = ""

                            except json.JSONDecodeError:
                                # JSON仍然不完整，继续缓冲
                                pass
                
            yield b"data: [DONE]\n\n"
        except httpx.RequestError as e:
            logger.error(f"[ChatProcessor] 上游请求失败: {e}", exc_info=True)
            error_message = f"无法连接到上游服务: {type(e).__name__}"
            converted_error = ErrorConverter.convert_upstream_error(error_message.encode(), 502, "openai", original_format)
            yield f"data: {json.dumps(converted_error)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    def _build_request_params(self, base_url: str, upstream_format: str, model: str, official_key: str, is_stream: bool) -> Tuple[str, Dict]:
        """根据目标平台构建URL和Headers"""
        headers = {"Content-Type": "application/json"}
        target_url = ""

        if upstream_format == "gemini":
            action = "streamGenerateContent" if is_stream else "generateContent"
            target_url = f"{base_url}/v1beta/models/{model}:{action}"
            headers["x-goog-api-key"] = official_key
        
        elif upstream_format == "claude":
            target_url = f"{base_url}/v1/messages"
            headers["x-api-key"] = official_key
            headers["anthropic-version"] = "2023-06-01"

        elif upstream_format == "openai":
            target_url = f"{base_url}/v1/chat/completions"
            headers["Authorization"] = f"Bearer {official_key}"
            
        return target_url, headers

chat_processor = ChatProcessor()