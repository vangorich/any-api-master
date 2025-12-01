import json
import time
import httpx
import logging
from typing import AsyncGenerator, Tuple, List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.services.universal_converter import universal_converter, ApiFormat
from app.services.variable_service import variable_service
from app.services.regex_service import regex_service
from app.models.user import User
from app.models.key import ExclusiveKey, OfficialKey
from app.models.preset import Preset
from app.models.regex import RegexRule
from app.models.preset_regex import PresetRegexRule
from app.models.log import Log
from app.core.config import settings
from sqlalchemy.future import select
from fastapi import Request

logger = logging.getLogger(__name__)

class ChatProcessor:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0)

    async def process_request(
        self,
        request: Request,
        db: AsyncSession,
        official_key: str,
        exclusive_key: ExclusiveKey,
        user: User,
        log_level: str,
        model_override: str = None
    ) -> Tuple[Dict[str, Any], int, ApiFormat]:
        start_time = time.time()
        body_bytes = await request.body()
        body = json.loads(body_bytes)
        target_format = "gemini"

        converted_body, original_format = await universal_converter.convert_request(body, "openai", request=request)
        
        if model_override:
            converted_body["model"] = model_override
            
        openai_request = ChatCompletionRequest(**converted_body)
        
        log_entry = await self._create_initial_log(db, exclusive_key, user, openai_request.model, openai_request.stream, body_bytes)

        presets, regex_rules, preset_regex_rules = await self._load_context(db, exclusive_key)
        openai_request = self._apply_preprocessing(openai_request, presets, regex_rules, preset_regex_rules)
        final_payload, _ = await universal_converter.convert_request(openai_request.dict(), target_format)
        
        if openai_request.stream:
            return self._logged_stream_generator(
                self.stream_chat_completion(
                    final_payload, target_format, original_format, openai_request.model,
                    official_key, regex_rules, preset_regex_rules
                ),
                db=db,
                log_entry=log_entry,
                start_time=start_time
            )
        else:
            result, status_code, _ = await self.non_stream_chat_completion(
                final_payload, target_format, original_format, openai_request.model,
                official_key, regex_rules, preset_regex_rules
            )
            latency = time.time() - start_time
            output_tokens = result.get('usage', {}).get('completion_tokens', 0)
            await self._finalize_log(db, log_entry, status_code, latency, output_tokens)
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

    async def _create_initial_log(self, db: AsyncSession, exclusive_key: ExclusiveKey, user: User, model: str, is_stream: bool, request_body: bytes) -> Log:
        input_tokens = len(request_body) // 4
        log_entry = Log(
            exclusive_key_id=exclusive_key.id,
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

    async def _finalize_log(self, db: AsyncSession, log_entry: Optional[Log], status_code: int, latency: float, output_tokens: int, ttft: Optional[float] = None):
        if not log_entry: return
        try:
            log_entry.status_code = status_code
            log_entry.status = "ok" if 200 <= status_code < 300 else "error"
            log_entry.latency = latency
            log_entry.ttft = ttft if ttft is not None else latency
            log_entry.output_tokens = output_tokens
            db.add(log_entry)
            await db.commit()
        except Exception as e:
            logger.error(f"[ChatProcessor] Failed to finalize log: {e}", exc_info=True)

    async def non_stream_chat_completion(
        self, payload: Dict, upstream_format: ApiFormat, original_format: ApiFormat, model: str,
        official_key: str, global_rules: List, local_rules: List
    ) -> Tuple[Dict, int, ApiFormat]:
        target_url = f"{settings.GEMINI_BASE_URL}/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": official_key.key if hasattr(official_key, 'key') else official_key}
        
        response = await self.client.post(target_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            openai_error = universal_converter.generic_error_to_openai(response.content, response.status_code, upstream_format)
            return openai_error, response.status_code, "openai"

        gemini_response = response.json()
        openai_response = universal_converter.gemini_response_to_openai_response(gemini_response, model)
        
        if openai_response.get('choices') and openai_response['choices'][0]['message'].get('content'):
            content = openai_response['choices'][0]['message']['content']
            content = self._apply_postprocessing(content, global_rules, local_rules)
            openai_response['choices'][0]['message']['content'] = content

        if original_format == "gemini":
            gemini_response = universal_converter.openai_response_to_gemini_response(openai_response)
            return gemini_response, 200, original_format
        
        return openai_response, 200, original_format

    async def _logged_stream_generator(self, generator: AsyncGenerator, db: AsyncSession, log_entry: Log, start_time: float):
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
                            if json_content.get('error'):
                                status_code = json_content.get('error', {}).get('code', 500)
                            if json_content.get('choices'):
                                delta = json_content['choices'][0].get('delta', {})
                                full_response_content += delta.get('content', '')
                        except json.JSONDecodeError:
                            pass
                yield chunk
        finally:
            latency = time.time() - start_time
            output_tokens = len(full_response_content) // 4
            await self._finalize_log(db, log_entry, status_code, latency, output_tokens, ttft)


    async def stream_chat_completion(
        self, payload: Dict, upstream_format: ApiFormat, original_format: ApiFormat, model: str,
        official_key: str, global_rules: List, local_rules: List
    ) -> AsyncGenerator[bytes, None]:
        target_url = f"{settings.GEMINI_BASE_URL}/v1beta/models/{model}:streamGenerateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": official_key.key if hasattr(official_key, 'key') else official_key}

        async with self.client.stream("POST", target_url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                error_content = await response.aread()
                openai_error = universal_converter.generic_error_to_openai(error_content, response.status_code, upstream_format)
                yield f"data: {json.dumps(openai_error)}\n\n".encode()
                return

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                decoder = json.JSONDecoder()
                while buffer:
                    buffer = buffer.lstrip(' \t\n\r,([')
                    if not buffer: break
                    try:
                        gemini_chunk, idx = decoder.raw_decode(buffer)
                        openai_chunk = universal_converter.gemini_to_openai_chunk(gemini_chunk, model)
                        if openai_chunk.get('choices') and openai_chunk['choices'][0]['delta'].get('content'):
                            content = openai_chunk['choices'][0]['delta']['content']
                            content = self._apply_postprocessing(content, global_rules, local_rules)
                            openai_chunk['choices'][0]['delta']['content'] = content
                        
                        if original_format == "gemini":
                            gemini_response_chunk = universal_converter.openai_chunk_to_gemini_chunk(openai_chunk)
                            yield f"data: {json.dumps(gemini_response_chunk)}\n\n".encode()
                        else:
                            yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
                        
                        buffer = buffer[idx:]
                    except json.JSONDecodeError:
                        break
        yield b"data: [DONE]\n\n"

chat_processor = ChatProcessor()