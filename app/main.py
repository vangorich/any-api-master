from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
from app.core.config import settings
from app.core.database import engine, Base
from app.models.system_config import SystemConfig
from app.core.database import get_db
from sqlalchemy import select
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure data directory exists
    if "sqlite" in settings.DATABASE_URL:
        db_path = settings.DATABASE_URL.split("///")[1]
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

    # # 自动执行数据库迁移 (已禁用，请手动执行 python migrate.py upgrade)
    # if False: # 自动迁移已禁用
    #     pass
    
    # Startup: Load site name from DB and set as app title
    db_session_gen = get_db()
    db = await anext(db_session_gen)
    try:
        stmt = select(SystemConfig)
        config = (await db.execute(stmt)).scalars().first()
        if config and config.site_name:
            app.title = config.site_name
        else:
            app.title = "Any API"
    except Exception as e:
        # 如果表不存在或其他数据库错误,提示用户运行迁移
        print(f"警告: 无法加载系统配置: {e}")
        # 提示用户运行迁移
        print("提示: 如果这是首次运行,请执行: python migrate.py upgrade")
    finally:
        await db.close()
            
    yield

app = FastAPI(
    title="Any API",
    version=settings.VERSION,
    openapi_url=f"{settings.VITE_API_STR}/openapi.json",
    lifespan=lifespan
)

# 注册全局异常处理器
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from app.core.exception_handlers import (
    api_exception_handler, 
    validation_exception_handler, 
    general_exception_handler
)

app.add_exception_handler(HTTPException, api_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

from app.api.api import api_router
from app.api.endpoints import generic_proxy
from app.api.endpoints import proxy
from app.api.endpoints import gemini_routes
from app.api.endpoints import claude_routes
from app.api.endpoints import universal_routes

app.include_router(api_router, prefix=settings.VITE_API_STR)

# 根路径路由挂载顺序至关重要

# 1. Gemini Native Routes (/v1beta...) - 优先匹配
app.include_router(gemini_routes.router)

# 2. OpenAI Compatible Routes (/v1/chat/completions, /v1/models) - 优先于通用/v1路由
# 必须放在 claude_routes 之前，因为 claude_routes 包含 /v1/{path} 通配符
app.include_router(proxy.router)

# 3. Claude Native Routes (/v1/messages, etc.)
app.include_router(claude_routes.router)

# 4. Universal Routes (/openai, /gemini, /claude) - 明确的前缀路由
# Move to later to avoid shadowing /v1beta etc if it catches too broadly (though it has prefix)
# But wait, universal_routes has path /{provider}/{path:path}.
# If provider is "v1beta", it matches! And v1beta is not in ["openai", "gemini", "claude"].
# So it MUST be after specific routes.
app.include_router(universal_routes.router)

# 5. Generic Proxy (Catch-all) - 最后匹配
app.include_router(generic_proxy.router, tags=["generic_proxy"])

@app.get("/")
async def root():
    return {"message": "Welcome to Any API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
