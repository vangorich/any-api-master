from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import logging
from app.core.logging import setup_logging
from app.core.config import settings
from app.core.database import engine, Base
from app.models.system_config import SystemConfig
from app.core.database import get_db
from sqlalchemy import select
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始配置日志 (使用默认设置，直到数据库加载完成)
    setup_logging()
    logging.info("应用启动，开始配置...")

    # Ensure data directory exists
    if "sqlite" in settings.DATABASE_URL:
        db_path = settings.DATABASE_URL.split("///")[1]
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    # Startup: Load site name from DB and set as app title
    db_session_gen = get_db()
    db = await anext(db_session_gen)
    try:
        stmt = select(SystemConfig)
        config = (await db.execute(stmt)).scalars().first()
        if config:
            # 加载站点名称
            if config.site_name:
                app.title = config.site_name
                logging.info(f"站点名称已加载: {app.title}")
            
            # 加载日志配置并重新应用
            if config.log_level:
                setup_logging(log_level=config.log_level)
                logging.info(f"已根据系统配置更新日志等级为: {config.log_level}")
        else:
            app.title = "Any API"
            logging.info("未找到系统配置，使用默认值")
    except Exception as e:
        logging.warning(f"无法加载系统配置: {e}")
        logging.warning("提示: 如果这是首次运行,请执行: python migrate.py upgrade")
    finally:
        await db.close()
            
    yield

    logging.info("应用关闭。")

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
# 注意：我们不再捕获通用的 Exception，因为它会覆盖掉 CancelledError
# app.add_exception_handler(Exception, general_exception_handler)

# 注册一个中间件来捕获 CancelledError
class CancelledRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await call_next(request)
        except asyncio.CancelledError:
            # 这里的日志是可选的，因为我们无法轻易访问到请求的详细信息
            # 主要目的是确保连接被正确关闭
            logging.info("请求被客户端取消。")
            # 返回一个标准响应表明客户端关闭了请求
            return Response(status_code=499)
        except Exception as exc:
            # 对于所有其他异常，我们仍然可以使用通用的处理器
            return await general_exception_handler(request, exc)

app.add_middleware(CancelledRequestMiddleware)


# 注册安全中间件 (添加CSP等安全响应头)
from app.middleware.security_middleware import SecurityHeadersMiddleware
app.add_middleware(SecurityHeadersMiddleware)

# --- 路由注册 ---

# 1. API 路由 (按照从最精确到最宽泛的顺序)
from app.api.api import api_router
from app.api.endpoints import generic_proxy, proxy, gemini_routes, claude_routes, universal_routes

app.include_router(api_router, prefix=settings.VITE_API_STR)

# 2. 静态文件服务 (必须在API路由之后,但在通配符路由之前)
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 3. 其他API路由
app.include_router(gemini_routes.router)
app.include_router(proxy.router)
app.include_router(claude_routes.router)
# app.include_router(universal_routes.router)
# app.include_router(generic_proxy.router, tags=["generic_proxy"])

# 4. SPA 前端 "后备" 路由 (必须在最后)
# 4. SPA 前端服务 (必须在所有API路由之后)
static_dir = "static" if os.path.exists("static") else "dist"
if os.path.exists(static_dir):
    # 显式挂载 `assets` 目录，这是最安全的做法
    app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")), name="assets")

    # 添加一个通用的后备路由来服务前端应用
    # 它会捕获所有未被API路由匹配到的请求
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        # 检查请求的路径是否对应一个真实存在的文件 (例如 /vite.svg)
        file_path = os.path.join(static_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        
        # 如果不是文件 (例如 /dashboard/system), 则返回 index.html
        return FileResponse(os.path.join(static_dir, "index.html"))

    # 为根路径单独添加一个路由，以防通配符路由出现问题
    @app.get("/", include_in_schema=False)
    async def serve_root():
        return FileResponse(os.path.join(static_dir, "index.html"))
else:
    logging.warning("静态文件目录 'static' 或 'dist' 未找到,前端将无法访问。")
    @app.get("/", include_in_schema=False)
    async def root_api_only():
        return {"message": "Welcome to Any API (Frontend not found)"}

# 这个启动块仅用于 python app/main.py 直接运行, uvicorn CLI 不会执行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
