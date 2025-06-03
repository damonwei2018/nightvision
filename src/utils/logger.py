# -*- coding: utf-8 -*-
"""日志管理模块

提供统一的日志记录功能，包括控制台、文件和Qt界面日志输出
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import time
from logging.handlers import RotatingFileHandler

# 全局变量，存储所有日志消息
log_history: List[Dict[str, Any]] = []

# 全局变量，存储Qt日志处理器
qt_log_handlers = {}

class QtLogHandler(logging.Handler):
    """Qt日志处理器
    
    将日志消息转发到Qt界面
    """
    def __init__(self, signal=None):
        super().__init__()
        self.signal = signal
        self.log_buffer = []
        self.max_buffer_size = 1000  # 最大缓冲区大小
    
    def emit(self, record):
        try:
            # 格式化日志消息
            msg = self.format(record)
            
            # 创建日志条目
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'level': record.levelname,
                'message': record.getMessage(),
                'formatted': msg,
                'logger_name': record.name,
                'module': record.module,
                'line_no': record.lineno
            }
            
            # 添加到全局日志历史
            global log_history
            log_history.append(log_entry)
            
            # 限制全局日志历史大小
            if len(log_history) > 10000:  # 最多保留10000条日志
                log_history = log_history[-5000:]  # 保留最新的5000条
            
            # 添加到缓冲区
            self.log_buffer.append(log_entry)
            
            # 限制缓冲区大小
            if len(self.log_buffer) > self.max_buffer_size:
                self.log_buffer = self.log_buffer[-int(self.max_buffer_size/2):]  # 保留后半部分
            
            # 如果有信号，发送日志消息
            if self.signal is not None:
                self.signal.emit(log_entry)
                
        except Exception as e:
            # 避免日志处理器异常导致应用崩溃
            print(f"Qt日志处理器异常: {e}")
    
    def get_logs(self, count=None, level=None, search=None):
        """获取日志
        
        Args:
            count: 返回的日志数量，None表示全部
            level: 过滤的日志级别，None表示全部
            search: 搜索关键字，None表示不搜索
            
        Returns:
            过滤后的日志列表
        """
        filtered_logs = self.log_buffer
        
        # 按级别过滤
        if level is not None:
            filtered_logs = [log for log in filtered_logs if log['level'] == level]
        
        # 按关键字搜索
        if search is not None:
            filtered_logs = [log for log in filtered_logs if search.lower() in log['message'].lower()]
        
        # 限制数量
        if count is not None:
            filtered_logs = filtered_logs[-count:]
        
        return filtered_logs

def setup_logger(name: str = "NightVision", 
                 log_file: Optional[str] = None,
                 level: int = logging.INFO,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，如果为None则不写入文件
        level: 日志级别
        max_file_size: 日志文件最大大小（字节）
        backup_count: 保留的日志文件数量
        
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
        
        # 使用RotatingFileHandler代替FileHandler，支持日志文件轮转
        file_handler = RotatingFileHandler(
            log_path, 
            maxBytes=max_file_size, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def register_qt_handler(logger_name: str, signal) -> QtLogHandler:
    """注册Qt日志处理器
    
    Args:
        logger_name: 日志记录器名称
        signal: Qt信号，用于发送日志消息
        
    Returns:
        Qt日志处理器
    """
    logger = logging.getLogger(logger_name)
    
    # 创建Qt日志处理器
    handler = QtLogHandler(signal)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # 存储处理器引用
    global qt_log_handlers
    qt_log_handlers[logger_name] = handler
    
    return handler

def get_qt_handler(logger_name: str) -> Optional[QtLogHandler]:
    """获取Qt日志处理器
    
    Args:
        logger_name: 日志记录器名称
        
    Returns:
        Qt日志处理器，如果不存在则返回None
    """
    global qt_log_handlers
    return qt_log_handlers.get(logger_name)

def get_log_history(count: Optional[int] = None, 
                   level: Optional[str] = None, 
                   search: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取全局日志历史
    
    Args:
        count: 返回的日志数量，None表示全部
        level: 过滤的日志级别，None表示全部
        search: 搜索关键字，None表示不搜索
        
    Returns:
        过滤后的日志列表
    """
    global log_history
    filtered_logs = log_history
    
    # 按级别过滤
    if level is not None:
        filtered_logs = [log for log in filtered_logs if log['level'] == level]
    
    # 按关键字搜索
    if search is not None:
        filtered_logs = [log for log in filtered_logs if search.lower() in log['message'].lower()]
    
    # 限制数量
    if count is not None:
        filtered_logs = filtered_logs[-count:]
    
    return filtered_logs

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