import json
import time
import httpx
import logging
from typing import AsyncGenerator, Tuple, List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.services.universal_converter import universal_converter, ApiFormat
from app.services.gemini_safety_service import gemini_safety_service
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
from fastapi import Request, BackgroundTasks
import traceback

import datetime
import asyncio

logger = logging.getLogger(__name__)

class ChatProcessor:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0)

    async def process_request(
        self,
        request: Request,
        db: AsyncSession,
        official_key: OfficialKey,
        exclusive_key: ExclusiveKey,
        user: User,
        background_tasks: BackgroundTasks,
        model_override: str = None,
        original_format: ApiFormat = "openai"
    ):
        start_time = time.time()
        body = await request.json()
        logger.debug(f"客户端原始请求体: {json.dumps(body, indent=2, ensure_ascii=False)}")

        target_format = official_key.channel.type.lower() if official_key.channel and official_key.channel.type else "gemini"
        
        logger.info(f"渠道 '{exclusive_key.name}' (ID: {exclusive_key.channel_id}) 接收到客户端请求: {request.url}")
        logger.info(f"请求转换: 客户端格式 ({original_format}) -> 内部格式 (openai)")
        logger.info(f"上游转换: 内部格式 (openai) -> 目标格式 ({target_format})")

        converted_body, _ = await universal_converter.convert_request(body, "openai", request=request)
        if model_override:
            converted_body["model"] = model_override
            
        openai_request = ChatCompletionRequest(**converted_body)
        
        log_entry = await self._create_and_commit_initial_log(db, exclusive_key, official_key, user, openai_request.model, openai_request.stream, [msg.dict() for msg in openai_request.messages])

        presets, regex_rules, preset_regex_rules = await self._load_context(db, exclusive_key)
        openai_request = self._apply_preprocessing(openai_request, presets, regex_rules, preset_regex_rules)
        final_payload, _ = await universal_converter.convert_request(openai_request.dict(), target_format)
        logger.debug(f"发送到上游的最终请求体: {json.dumps(final_payload, indent=2, ensure_ascii=False)}")
        
        is_pseudo_stream = False
        if openai_request.model.startswith("伪流/"):
            is_pseudo_stream = True
            openai_request.model = openai_request.model[3:]
            final_payload["model"] = openai_request.model
            openai_request.stream = True # 强制流式

        if openai_request.stream:
            full_response_content = ""
            ttft = 0.0

            async def stream_generator_wrapper():
                nonlocal full_response_content, ttft
                first_chunk_time = None
                
                if is_pseudo_stream:
                    gen = self.pseudo_stream_chat_completion(
                        final_payload, target_format, original_format, openai_request.model,
                        official_key, regex_rules, preset_regex_rules
                    )
                else:
                    gen = self.stream_chat_completion(
                        final_payload, target_format, original_format, openai_request.model,
                        official_key, regex_rules, preset_regex_rules
                    )
                
                async for chunk in gen:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        ttft = first_chunk_time - start_time

                    if chunk.startswith(b'data: '):
                        content_part = chunk[6:].strip()
                        if content_part != b'[DONE]':
                            try:
                                json_content = json.loads(content_part)
                                
                                # 处理 json_content 可能是列表或字典的情况
                                content_items = []
                                if isinstance(json_content, list):
                                    content_items.extend(json_content)
                                elif isinstance(json_content, dict):
                                    content_items.append(json_content)

                                for item in content_items:
                                    if item.get('choices'):
                                        delta = item['choices'][0].get('delta', {})
                                        full_response_content += delta.get('content', '')
                            except json.JSONDecodeError:
                                pass
                    yield chunk

            background_tasks.add_task(
                self._update_final_log,
                db=db,
                log_id=log_entry.id,
                start_time=start_time,
                get_full_response=lambda: full_response_content,
                get_ttft=lambda: ttft
            )
            
            return stream_generator_wrapper()
        else:
            result, status_code, _ = await self.non_stream_chat_completion(
                final_payload, target_format, original_format, openai_request.model,
                official_key, regex_rules, preset_regex_rules
            )
            latency = time.time() - start_time
            tokenizer = get_tokenizer(openai_request.model)
            response_content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            output_tokens = len(tokenizer.encode(response_content))
            
            await self._update_final_log(db, log_entry.id, start_time, lambda: response_content, lambda: latency, status_code)
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

    async def _create_and_commit_initial_log(self, db: AsyncSession, exclusive_key: ExclusiveKey, official_key: OfficialKey, user: User, model: str, is_stream: bool, messages: List[Dict[str, Any]]) -> Log:
        input_tokens = count_tokens_for_messages(messages, model)
        
        log_entry = Log(
            exclusive_key_id=exclusive_key.id,
            official_key_id=official_key.id,
            user_id=user.id,
            model=model,
            status="processing", # Will be updated by background task
            status_code=0,
            latency=0,
            ttft=0,
            is_stream=is_stream,
            input_tokens=input_tokens,
            output_tokens=0 # Placeholder
        )
        db.add(log_entry)
        
        # Immediately update usage count
        official_key.usage_count += 1
        official_key.input_tokens = (official_key.input_tokens or 0) + (input_tokens or 0)
        db.add(official_key)
        
        await db.commit()
        await db.refresh(log_entry)
        
        logger.info(f"已创建初始日志条目 (ID: {log_entry.id}) 并更新密钥使用次数 (Official Key ID: {official_key.id})")
        return log_entry

    async def _update_final_log(self, db: AsyncSession, log_id: int, start_time: float, get_full_response, get_ttft, status_code: int = 200):
        try:
            latency = time.time() - start_time
            full_response_content = get_full_response()
            ttft = get_ttft()

            result = await db.execute(select(Log).filter(Log.id == log_id))
            log_entry = result.scalars().first()
            if not log_entry:
                logger.error(f"后台任务无法找到日志条目: ID={log_id}")
                return

            result = await db.execute(select(OfficialKey).filter(OfficialKey.id == log_entry.official_key_id))
            official_key = result.scalars().first()
            if not official_key:
                logger.error(f"后台任务无法找到官方密钥: ID={log_entry.official_key_id}")
                return

            tokenizer = get_tokenizer(log_entry.model)
            output_tokens = len(tokenizer.encode(full_response_content))

            log_entry.status_code = status_code
            log_entry.status = "ok" if 200 <= status_code < 300 else "error"
            log_entry.latency = latency
            log_entry.ttft = ttft if ttft > 0 else latency
            log_entry.output_tokens = output_tokens
            db.add(log_entry)

            official_key.output_tokens = (official_key.output_tokens or 0) + output_tokens
            official_key.last_status_code = status_code
            if log_entry.status == "error":
                official_key.error_count += 1
                official_key.last_status = str(status_code)
            else:
                official_key.last_status = "active"
            db.add(official_key)
            
            await db.commit()
            logger.info(f"后台任务成功更新最终日志 (ID: {log_id})")

        except Exception as e:
            logger.error(f"后台日志更新任务失败 (Log ID: {log_id})。错误: {e}\n{traceback.format_exc()}")
            await db.rollback()

    async def non_stream_chat_completion(
        self, payload: Dict, upstream_format: ApiFormat, original_format: ApiFormat, model: str,
        official_key: OfficialKey, global_rules: List, local_rules: List
    ) -> Tuple[Dict, int, ApiFormat]:
        if upstream_format == "gemini":
            payload = gemini_safety_service.add_safety_settings_to_payload(payload)

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

    # This function is no longer needed as the logic is now inside process_request
    # async def _logged_stream_generator(...):


    async def stream_chat_completion(
        self, payload: Dict, upstream_format: ApiFormat, original_format: ApiFormat, model: str,
        official_key: OfficialKey, global_rules: List, local_rules: List
    ) -> AsyncGenerator[bytes, None]:
        if upstream_format == "gemini":
            payload = gemini_safety_service.add_safety_settings_to_payload(payload)
            
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
                
                async for decoded_chunk in response.aiter_text():
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

    async def pseudo_stream_chat_completion(
        self, payload: Dict, upstream_format: ApiFormat, original_format: ApiFormat, model: str,
        official_key: OfficialKey, global_rules: List, local_rules: List
    ) -> AsyncGenerator[bytes, None]:
        """
        处理伪流请求：使用非流式请求上游，但在等待时向客户端发送空流以保持连接。
        """
        logger.info(f"启动伪流模式 (模型: {model})")
        
        # 创建一个 future 来等待非流式请求的结果
        non_stream_task = asyncio.create_task(
            self.non_stream_chat_completion(
                payload, upstream_format, original_format, model,
                official_key, global_rules, local_rules
            )
        )

        try:
            while not non_stream_task.done():
                # 每秒发送一个空的SSE注释或事件以保持连接
                yield b": keep-alive\n\n"
                await asyncio.sleep(1)

            # 获取非流式请求的结果
            result, status_code, _ = await non_stream_task
            
            if status_code != 200:
                # 如果上游出错，将错误信息作为单个数据块发送
                yield f"data: {json.dumps(result)}\n\n".encode()
            else:
                # 将完整的非流式响应转换为单个流式数据块
                # 我们需要将完整的消息内容包装在一个 'delta' 中
                full_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # 创建一个模拟的流式块
                stream_chunk = {
                    "id": f"chatcmpl-pseudo-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": full_content
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(stream_chunk)}\n\n".encode()

        except Exception as e:
            logger.error(f"[PseudoStream] 伪流处理失败: {e}", exc_info=True)
            error_message = f"伪流处理时发生内部错误: {type(e).__name__}"
            converted_error = ErrorConverter.convert_upstream_error(error_message.encode(), 500, "openai", original_format)
            yield f"data: {json.dumps(converted_error)}\n\n".encode()
        finally:
            # 确保发送 [DONE] 消息
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