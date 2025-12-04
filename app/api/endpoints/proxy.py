import json
import time
import httpx
import logging
from typing import Any, List, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.api import deps
from app.models.user import User
from app.models.key import ExclusiveKey, OfficialKey
from app.models.preset import Preset
from app.models.regex import RegexRule
from app.models.preset_regex import PresetRegexRule
from app.models.log import Log
from app.models.system_config import SystemConfig
from app.schemas.openai import ChatCompletionRequest
from app.services.gemini_service import gemini_service
from app.services.universal_converter import universal_converter
from app.services.variable_service import variable_service
from app.services.regex_service import regex_service
from app.services.chat_processor import chat_processor
from app.services.proxy_service import proxy_service
from app.core.config import settings

router = APIRouter()

# Configure logger
logger = logging.getLogger(__name__)
current_log_level = "INFO"

async def get_log_level(db: AsyncSession):
    global current_log_level
    result = await db.execute(select(SystemConfig))
    config = result.scalars().first()
    if config and config.log_level:
        current_log_level = config.log_level
        return config.log_level
    current_log_level = "INFO"
    return "INFO"

def update_logger_level(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Ensure handler exists and set level for handler as well
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    for handler in logger.handlers:
        handler.setLevel(level)

def debug_log(message: str):
    """
    Wrapper for debug logging.
    """
    if current_log_level == "DEBUG":
        logger.debug(message)


@router.get("/v1/models")
async def list_models(
    key_info: tuple = Depends(deps.get_official_key_from_proxy)
):
    """
    处理 GET /v1/models 请求，通过代理到 Google API 列出可用模型。
    使用新的依赖项处理密钥。
    """
    official_key, _ = key_info

    # 2. 代理到 Google API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": official_key.key}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"请求 Google API 时出错: {e}")

    # 3. 转换响应
    try:
        gemini_response = response.json()
        models = gemini_response.get("models", [])
        
        openai_models = []
        for model in models:
            model_id = model.get("name", "").replace("models/", "")
            openai_models.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            })
            
        return {
            "object": "list",
            "data": openai_models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析或转换模型列表时出错: {e}")


@router.api_route("/v1beta/{path:path}", methods=["POST", "PUT", "DELETE", "GET"])
async def proxy_beta_requests(
    request: Request,
    path: str,
    db: AsyncSession = Depends(deps.get_db),
    key_info: tuple = Depends(deps.get_official_key_from_proxy)
):
    """
    通用代理，处理 /v1beta/ 下的所有请求，并交由 ProxyService 处理以实现日志记录。
    """
    official_key, user = key_info

    # 假设所有到 v1beta 的请求都意图发往 gemini
    return await proxy_service.smart_proxy_handler(
        request=request,
        db=db,
        path=path,
        official_key_obj=official_key,
        user=user,
        incoming_format="gemini"
    )

@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    db: AsyncSession = Depends(deps.get_db),
    key_info: tuple = Depends(deps.get_official_key_from_proxy)
):
    # 0. Configure Logging Level
    log_level = await get_log_level(db)
    update_logger_level(log_level)

    # 1. Auth & Key Validation
    official_key, user = key_info
    
    # 检查是否是专属密钥的逻辑现在由 get_official_key_from_proxy 处理
    # 如果 user 不为 None, 则说明是有效的专属密钥
    is_exclusive = user is not None
    exclusive_key = None
    if is_exclusive:
        # 为了日志记录，可能需要获取 exclusive_key 对象
        auth_header = request.headers.get("Authorization")
        client_key = auth_header.split(" ")[1] if auth_header and auth_header.startswith("Bearer ") else ""
        if client_key:
            result = await db.execute(select(ExclusiveKey).filter(ExclusiveKey.key == client_key))
            exclusive_key = result.scalars().first()
            debug_log(f"处理专属 Key 请求. Key ID: {exclusive_key.id}, 名称: {exclusive_key.name}")
    else:
        debug_log(f"处理官方 Key 请求.")

    # 2. Parse Request
    try:
        body = await request.json()
        openai_request = ChatCompletionRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    # 3. 如果是 gapi- key, 调用 ChatProcessor
    if is_exclusive and exclusive_key:
        # ChatProcessor 现在内部处理自己的日志记录，这里不再需要额外的日志代码
        result = await chat_processor.process_request(
            request=request,
            db=db,
            official_key=official_key,
            exclusive_key=exclusive_key,
            user=user,
            log_level=log_level,
            # For this endpoint, the client is always speaking the "openai" format
            original_format="openai"
        )
        
        # 根据结果类型返回响应
        if isinstance(result, AsyncGenerator):
            return StreamingResponse(result, media_type="text/event-stream")
        else:
            response_content, status_code, _ = result
            return JSONResponse(content=response_content, status_code=status_code)

    # --- 非 gapi- key 的新逻辑 (使用 ProxyService) ---
    
    return await proxy_service.smart_proxy_handler(
        request=request,
        db=db,
        path="chat/completions",
        official_key_obj=official_key,
        user=user, # user will be None for non-exclusive keys
        incoming_format="openai"
    )
