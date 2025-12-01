from typing import Dict, Any, Tuple, Literal
import json
import time
import uuid
import httpx
import base64
import asyncio
import logging
from fastapi import Request
from app.schemas.openai import ChatCompletionRequest
from app.core.errors import ErrorConverter

logger = logging.getLogger(__name__)

# 定义支持的API格式
ApiFormat = Literal["openai", "gemini", "claude"]

class UniversalConverter:
    """
    一个通用的API格式转换器，用于在OpenAI、Gemini和Claude格式之间进行转换。
    """

    # --- 格式检测 ---
    def detect_format(self, body: Dict[str, Any]) -> ApiFormat:
        """
        通过请求体的结构检测API格式。
        """
        if "contents" in body and isinstance(body["contents"], list):
            return "gemini"
        if "messages" in body and "max_tokens" in body: # Claude-specific field
            return "claude"
        if "messages" in body:
            return "openai"
        raise ValueError("无法检测到API格式或格式不支持")

    # --- 主转换入口 ---
    async def convert_request(self, body: Dict[str, Any], to_format: ApiFormat, request: Request = None) -> Tuple[Dict[str, Any], ApiFormat]:
        """
        将请求体从一种格式转换为另一种格式。
        """
        # Handle empty body (e.g. GET requests)
        if not body:
            return body, to_format

        from_format = self.detect_format(body)
        if from_format == to_format:
            return body, from_format

        # 中转枢纽：任何格式都先转为OpenAI格式
        openai_body = body
        if from_format != "openai":
            converter_func = getattr(self, f"{from_format}_request_to_openai_request", None)
            if not callable(converter_func):
                raise NotImplementedError(f"从 {from_format} 请求到 openai 请求的转换未实现")

            # 检查函数是否接受 request 参数
            import inspect
            sig = inspect.signature(converter_func)
            
            kwargs = {'body': body}
            if 'request' in sig.parameters and request:
                kwargs['request'] = request

            if asyncio.iscoroutinefunction(converter_func):
                openai_body = await converter_func(**kwargs)
            else:
                openai_body = converter_func(**kwargs)

        # 如果目标是OpenAI，直接返回
        if to_format == "openai":
            return openai_body, from_format

        # 从OpenAI格式转换为目标格式
        final_converter_func = getattr(self, f"openai_request_to_{to_format}_request", None)
        if not callable(final_converter_func):
            raise NotImplementedError(f"从 openai 请求到 {to_format} 请求的转换未实现")
        
        if not isinstance(openai_body, ChatCompletionRequest):
             openai_request = ChatCompletionRequest(**openai_body)
        else:
             openai_request = openai_body
        
        if asyncio.iscoroutinefunction(final_converter_func):
            converted_body = await final_converter_func(openai_request)
        else:
            converted_body = final_converter_func(openai_request)

        return converted_body, from_format

    def convert_chunk(self, chunk: Dict[str, Any], to_format: ApiFormat, from_provider: ApiFormat, original_model: str) -> Tuple[Dict[str, Any], bool]:
        """
        转换单个流式块。
        返回 (转换后的块, 是否为结束块)
        """
        if from_provider == to_format:
            return chunk, False # Passthrough, we don't know if it's the end

        converter_func = getattr(self, f"{from_provider}_to_{to_format}_chunk", None)
        if not callable(converter_func):
            # Fallback for simplicity, just wrap it
            return {"converted_chunk": chunk}, False

        converted = converter_func(chunk, original_model)
        is_done = converted.get("choices", [{}])[0].get("finish_reason") is not None
        
        return converted, is_done

    # --- OpenAI <-> Gemini ---
    
    def gemini_request_to_openai_request(self, body: Dict[str, Any], request: Request = None) -> Dict[str, Any]:
        """将Gemini请求转换为OpenAI请求"""
        messages = []
        contents = body.get("contents", [])
        system_instruction = body.get("system_instruction")
        
        if system_instruction:
            parts = system_instruction.get("parts", [])
            content = "".join([p.get("text", "") for p in parts if "text" in p])
            if content:
                messages.append({"role": "system", "content": content})
                
        for content in contents:
            role = content.get("role")
            if role == "model": role = "assistant"
            
            parts = content.get("parts", [])
            text_content = ""
            for part in parts:
                if "text" in part:
                    text_content += part["text"]
                # TODO: Handle function calls/responses in parts if needed
                
            messages.append({"role": role, "content": text_content})
            
        is_stream = False
        if request and "streamGenerateContent" in str(request.url):
            is_stream = True

        return {
            "messages": messages,
            "stream": is_stream,
            # Gemini doesn't strictly have a model in the body usually (it's in URL),
            # but we can try to extract or default.
            "model": "gemini-1.5-pro"
        }

    async def openai_request_to_gemini_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """将OpenAI请求转换为Gemini请求"""
        contents = []
        system_instruction = None
        
        system_parts = []
        for msg in request.messages:
            if msg.role == "system":
                system_parts.append({"text": msg.content})
            elif msg.role == "user":
                parts = []
                if isinstance(msg.content, str):
                    parts.append({"text": msg.content})
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if item.get("type") == "text":
                            parts.append({"text": item["text"]})
                        elif item.get("type") == "image_url":
                            # 图像处理逻辑（保持不变）
                            image_url = item["image_url"]["url"]
                            mime_type = "image/jpeg"
                            data = None
                            if image_url.startswith("data:"):
                                header, encoded = image_url.split(",", 1)
                                data = encoded
                                mime_type = header.split(";")[0].split(":")[1]
                            else:
                                async with httpx.AsyncClient() as client:
                                    resp = await client.get(image_url)
                                    if resp.status_code == 200:
                                        data = base64.b64encode(resp.content).decode("utf-8")
                                        content_type = resp.headers.get("content-type")
                                        if content_type:
                                            mime_type = content_type
                            if data:
                                parts.append({"inline_data": {"mime_type": mime_type, "data": data}})
                contents.append({"role": "user", "parts": parts})
            elif msg.role == "assistant":
                parts = []
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        parts.append({"functionCall": {"name": tool_call["function"]["name"], "args": json.loads(tool_call["function"]["arguments"])}})
                if msg.content:
                    parts.append({"text": msg.content})
                contents.append({"role": "model", "parts": parts})
            elif msg.role == "tool":
                function_name = msg.name
                if not function_name:
                    for prev_msg in reversed(request.messages):
                        if prev_msg.role == "assistant" and prev_msg.tool_calls:
                            for tc in prev_msg.tool_calls:
                                if tc["id"] == msg.tool_call_id:
                                    function_name = tc["function"]["name"]
                                    break
                        if function_name:
                            break
                
                if function_name:
                    contents.append({
                        "role": "function",
                        "parts": [{"functionResponse": {"name": function_name, "response": {"content": msg.content}}}]
                    })
                else:
                    contents.append({"role": "user", "parts": [{"text": f"Tool output: {msg.content}"}]})
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "topP": request.top_p,
                "maxOutputTokens": request.max_tokens,
                "stopSequences": request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else []
            }
        }
        if system_parts:
            payload["system_instruction"] = {"parts": system_parts}
        
        # 工具和工具选择的转换
        if request.tools:
            tools_list = []
            for tool in request.tools:
                if tool["type"] == "function":
                    tools_list.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description"),
                        "parameters": tool["function"].get("parameters")
                    })
            if tools_list:
                payload["tools"] = [{"function_declarations": tools_list}]
        
        if request.tool_choice:
            mode = "AUTO"
            allowed_function_names = None
            if isinstance(request.tool_choice, str):
                if request.tool_choice == "none": mode = "NONE"
                elif request.tool_choice == "auto": mode = "AUTO"
                elif request.tool_choice == "required": mode = "ANY"
            elif isinstance(request.tool_choice, dict):
                if request.tool_choice.get("type") == "function":
                    mode = "ANY"
                    allowed_function_names = [request.tool_choice["function"]["name"]]
            
            tool_config = {"function_calling_config": {"mode": mode}}
            if allowed_function_names:
                tool_config["function_calling_config"]["allowed_function_names"] = allowed_function_names
            payload["tool_config"] = tool_config
            
        return payload

    def gemini_response_to_openai_response(self, response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """将Gemini响应转换为OpenAI响应"""
        choices = []
        if "candidates" in response:
            for i, candidate in enumerate(response["candidates"]):
                message = {"role": "assistant", "content": None}
                finish_reason = "stop"
                
                if "content" in candidate and "parts" in candidate["content"]:
                    content_str = ""
                    tool_calls = []
                    
                    for part in candidate["content"]["parts"]:
                        # 优先检查 thought 标记 (new-api 风格: thought 是 bool, text 是内容)
                        is_thought = part.get("thought", False)
                        
                        if "text" in part:
                            if is_thought:
                                if "reasoning_content" not in message:
                                    message["reasoning_content"] = ""
                                message["reasoning_content"] += part["text"]
                            else:
                                content_str += part["text"]
                        elif "thought" in part and isinstance(part["thought"], str):
                            # 兼容旧逻辑或非标准格式，如果 thought 本身是字符串内容
                            if "reasoning_content" not in message:
                                message["reasoning_content"] = ""
                            message["reasoning_content"] += part["thought"]
                        
                        if "functionCall" in part:
                            fc = part["functionCall"]
                            tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": fc["name"],
                                    "arguments": json.dumps(fc["args"])
                                }
                            })
                    
                    if content_str:
                        message["content"] = content_str
                    elif message["content"] is None and ("reasoning_content" in message or tool_calls):
                        # 如果只有思考内容或工具调用，确保 content 不为 None，避免客户端报错
                        message["content"] = ""

                    if tool_calls:
                        message["tool_calls"] = tool_calls
                        finish_reason = "tool_calls"
                
                if candidate.get("finishReason") == "MAX_TOKENS":
                    finish_reason = "length"
                
                choices.append({
                    "index": i,
                    "message": message,
                    "finish_reason": finish_reason
                })
        
        usage = response.get("usageMetadata", {})
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
        }

    def openai_response_to_gemini_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """将OpenAI响应转换为Gemini响应"""
        # OpenAI Response: {"id":..., "choices":[{"message":{"content":...}}], ...}
        # Gemini Response: {"candidates": [{"content": {"parts": [{"text": ...}], "role": "model"}}]}
        
        candidates = []
        if "choices" in response:
            for choice in response["choices"]:
                parts = []
                message = choice.get("message", {})
                content = message.get("content")
                if content:
                    parts.append({"text": content})
                
                # 处理 function calls
                if "tool_calls" in message:
                     for tool_call in message["tool_calls"]:
                        if tool_call["type"] == "function":
                            parts.append({
                                "functionCall": {
                                    "name": tool_call["function"]["name"],
                                    "args": json.loads(tool_call["function"]["arguments"])
                                }
                            })

                if parts:
                    candidate = {
                        "content": {
                            "role": "model",
                            "parts": parts
                        },
                        "finishReason": "STOP" # Default
                    }
                    
                    if choice.get("finish_reason") == "length":
                        candidate["finishReason"] = "MAX_TOKENS"
                    elif choice.get("finish_reason") == "tool_calls":
                         candidate["finishReason"] = "STOP" # Gemini uses STOP for function calls too usually, or handled differently
                    
                    candidates.append(candidate)

        gemini_response = {"candidates": candidates}
        
        # Usage metadata
        if "usage" in response:
            usage = response["usage"]
            gemini_response["usageMetadata"] = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0)
            }
            
        return gemini_response

    # --- OpenAI <-> Claude ---

    def claude_request_to_openai_request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """将Claude请求转换为OpenAI请求"""
        messages = []
        
        # 处理 System Prompt
        system_prompt = body.get("system")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # 处理 Messages
        claude_messages = body.get("messages", [])
        for msg in claude_messages:
            messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })
            
        return {
            "messages": messages,
            "max_tokens": body.get("max_tokens"),
            "temperature": body.get("temperature"),
            "stream": body.get("stream", False),
            "model": body.get("model", "claude-3-opus-20240229") # Default or pass through
        }

    def openai_request_to_claude_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """将OpenAI请求转换为Claude请求"""
        system_prompt = None
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                # Claude 的消息格式需要特殊处理
                # 特别是对于多模态内容
                messages.append({"role": msg.role, "content": msg.content})

        return {
            "model": request.model,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
            # 其他参数映射...
        }

    def claude_response_to_openai_response(self, response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """将Claude响应转换为OpenAI响应"""
        # 实现逻辑...
        # 这需要根据Claude的具体响应格式来定
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.get("content", [{}])[0].get("text", "")
                    },
                    "finish_reason": "stop" # 简化处理
                }
            ],
            "usage": {
                "prompt_tokens": 0, # Claude响应中可能没有token信息
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        }
        
    # --- 流式和错误处理辅助函数 ---
    def gemini_to_openai_chunk(self, response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """将Gemini流式块转换为OpenAI格式"""
        choices = []
        if "candidates" in response:
            for i, candidate in enumerate(response["candidates"]):
                delta = {}
                finish_reason = None
                
                if "content" in candidate and "parts" in candidate["content"]:
                    content_str = ""
                    tool_calls = []
                    for part in candidate["content"]["parts"]:
                        is_thought = part.get("thought", False)

                        if "text" in part:
                            if is_thought:
                                if "reasoning_content" not in delta:
                                    delta["reasoning_content"] = ""
                                delta["reasoning_content"] += part["text"]
                            else:
                                content_str += part["text"]
                        elif "thought" in part and isinstance(part["thought"], str):
                             # 兼容旧逻辑
                            if "reasoning_content" not in delta:
                                delta["reasoning_content"] = ""
                            delta["reasoning_content"] += part["thought"]
                        
                        if "functionCall" in part:
                            fc = part["functionCall"]
                            tool_calls.append({
                                "index": 0,
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {"name": fc["name"], "arguments": json.dumps(fc["args"])}
                            })
                    if content_str:
                        delta["content"] = content_str
                    # 注意：在流式 chunk 中，如果 content 为空通常不需要发送 content 字段，
                    # 除非客户端有特殊癖好。暂时保持原样，或者可以考虑发送 ""
                    
                    if tool_calls:
                        delta["tool_calls"] = tool_calls

                if candidate.get("finishReason") == "MAX_TOKENS": finish_reason = "length"
                elif candidate.get("finishReason") == "STOP": finish_reason = "stop"

                choices.append({"index": i, "delta": delta, "finish_reason": finish_reason})
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": choices
        }

    def openai_chunk_to_gemini_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """将OpenAI流式块转换为Gemini格式"""
        # OpenAI Chunk: {"choices": [{"delta": {"content": "..."}}]}
        # Gemini Chunk: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
        
        candidates = []
        if "choices" in chunk:
            for choice in chunk["choices"]:
                delta = choice.get("delta", {})
                parts = []
                
                content = delta.get("content")
                if content:
                    parts.append({"text": content})
                    
                if "reasoning_content" in delta:
                     parts.append({"thought": delta["reasoning_content"]}) # Custom extension for now

                # TODO: Handle tool_calls in stream if needed

                if parts:
                    candidates.append({
                        "content": {
                            "parts": parts,
                            "role": "model"
                        }
                    })
                    
                # finish_reason
                if choice.get("finish_reason"):
                     # 如果是结束块，Gemini 有时会在最后一个块发送 finishReason
                     # 但通常流式中间块不需要 finishReason，除非结束
                     finish_reason = "STOP"
                     if choice["finish_reason"] == "length": finish_reason = "MAX_TOKENS"
                     
                     # 查找或创建 candidate
                     if not candidates:
                         candidates.append({"content": {"parts": [], "role": "model"}})
                     
                     candidates[0]["finishReason"] = finish_reason

        return {"candidates": candidates}

    def claude_to_openai_chunk(self, chunk: Dict[str, Any], model: str) -> Dict[str, Any]:
        """将Claude流式块转换为OpenAI格式"""
        # Claude流式响应需要特殊处理
        # 例如: event: message_delta, data: {"type":"message_delta",...}
        delta_content = ""
        if chunk.get("type") == "content_block_delta":
            delta_content = chunk.get("delta", {}).get("text", "")

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": delta_content}, "finish_reason": None}]
        }
        

# 创建单例
universal_converter = UniversalConverter()