from typing import AsyncGenerator
from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, Response, JSONResponse
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.gemini_service import gemini_service
from app.api import deps
from app.core.database import get_db
from app.models.user import User
from sqlalchemy.future import select
from app.models.system_config import SystemConfig

from app.services.chat_processor import chat_processor
from app.models.key import ExclusiveKey
from app.services.proxy_service import proxy_service

router = APIRouter()

async def ensure_log_level(db: AsyncSession) -> str:
    """Ensure service logger level matches system config and return it"""
    result = await db.execute(select(SystemConfig))
    config = result.scalars().first()
    log_level = "INFO"
    if config and config.log_level:
        log_level = config.log_level
        gemini_service.update_log_level(log_level)
    return log_level

@router.api_route("/v1beta/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_v1beta(
    path: str,
    request: Request,
    key_info: tuple = Depends(deps.get_official_key_from_proxy),
    db: AsyncSession = Depends(get_db)
):
    log_level = await ensure_log_level(db)
    official_key, user = key_info
    
    # 判断是否为 gapi- key
    is_exclusive = user is not None
    # 强制将 path 转为 str，以防万一
    path_str = str(path)
    
    # 显式检查生成内容的端点
    is_generate_content = "generateContent" in path_str or "streamGenerateContent" in path_str

    # 如果是 gapi- key 并且是 chat completion 请求，则使用 ChatProcessor
    if is_exclusive and is_generate_content and request.method == "POST":
        # 获取 exclusive_key 对象
        # 修复：需要支持从 header 或 query params 中获取 key，与 deps 逻辑保持一致
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            client_key = auth_header.split(" ")[1]
        else:
            client_key = request.headers.get("x-goog-api-key") or request.query_params.get("key")
        
        result = await db.execute(select(ExclusiveKey).filter(ExclusiveKey.key == client_key))
        exclusive_key = result.scalars().first()
        
        if not exclusive_key:
            raise HTTPException(status_code=401, detail="Invalid exclusive key")

        # 尝试从路径中解析模型名称
        # path 示例: models/gemini-1.5-flash:streamGenerateContent
        model_override = None
        if "models/" in path_str and ":" in path_str:
            try:
                # 提取 models/ 和 : 之间的部分
                start_idx = path_str.find("models/") + len("models/")
                end_idx = path_str.find(":", start_idx)
                if end_idx > start_idx:
                    model_override = path_str[start_idx:end_idx]
            except Exception:
                # 解析失败也没关系，model_override 会是 None
                pass

        result = await chat_processor.process_request(
            request=request, db=db, official_key=official_key,
            exclusive_key=exclusive_key, user=user, log_level=log_level,
            model_override=model_override
        )
        
        if isinstance(result, AsyncGenerator):
            return StreamingResponse(result, media_type="text/event-stream")
        else:
            response_content, status_code, _ = result
            return JSONResponse(content=response_content, status_code=status_code)

    # --- 对于非 gapi- key 或非聊天请求，使用 ProxyService 进行智能透传/转换 ---
    
    # 注意：ProxyService 需要处理 v1beta 前缀问题。
    # 这里我们传入 path，ProxyService 会根据 target provider 决定是否加前缀。
    
    return await proxy_service.smart_proxy_handler(
        request=request,
        db=db,
        path=path,
        official_key_obj=official_key,
        user=user,
        incoming_format="gemini"
    )
