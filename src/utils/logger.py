# -*- coding: utf-8 -*-
"""
日志管理模块

提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(name: str = "NightVision", 
                 log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，如果为None则不写入文件
        level: 日志级别
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 如果已经有处理器，先清除
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "NightVision") -> logging.Logger:
    """获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    return logging.getLogger(name)

class LoggerMixin:
    """日志记录器混入类
    
    为类提供日志记录功能
    """
    
    @property
    def logger(self) -> logging.Logger:
        """获取类的日志记录器"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger