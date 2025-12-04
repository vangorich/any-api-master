import logging
import logging.config
import sys
from app.core.config import settings

def setup_logging(log_level: str = None):
    """
    配置全局日志记录器。
    统一管理应用和 Uvicorn 日志，使用统一的格式：
    时间 | 日志等级 | 模块名 | 中文日志内容
    
    :param log_level: 可选的日志等级 (DEBUG, INFO, WARNING, ERROR, CRITICAL)。
                      如果未提供，则根据 settings.DEBUG 自动决定 (DEBUG/INFO)。
    """
    
    # 定义日志格式
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 确定日志等级
    if not log_level:
        log_level = "DEBUG" if settings.DEBUG else "INFO"
    
    log_level = log_level.upper()

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": date_format,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            # 应用根日志
            "app": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            # Uvicorn 访问日志
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            # Uvicorn 错误日志
            "uvicorn.error": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            # 根日志 (捕获其他库的日志)
            "": {
                "handlers": ["console"],
                "level": "INFO", # 默认 INFO，避免第三方库 DEBUG 刷屏
            },
        },
    }

    try:
        logging.config.dictConfig(logging_config)
        # 强制设置根 logger 级别，以防 dictConfig 没生效
        logging.getLogger().setLevel(log_level) 
        logging.info(f"日志系统初始化完成。当前等级: {log_level}")
    except Exception as e:
        # 如果配置失败，回退到简单配置
        logging.basicConfig(level=logging.INFO)
        logging.error(f"日志配置加载失败: {e}")
