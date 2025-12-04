import logging
import sys

def setup_logging():
    """
    配置全局日志记录器。
    """
    # 获取根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # 设置为DEBUG级别以显示所有日志

    # 移除所有现有的处理器，以避免重复日志
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # 创建一个新的流处理器
    stream_handler = logging.StreamHandler(sys.stdout)
    
    # 创建格式化器，加入级别和模块名
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 将格式化器添加到处理器
    stream_handler.setFormatter(formatter)
    
    # 将处理器添加到根记录器
    root_logger.addHandler(stream_handler)
