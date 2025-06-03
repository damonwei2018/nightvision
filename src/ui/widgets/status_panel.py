# -*- coding: utf-8 -*-
"""
状态面板

负责显示处理进度、性能指标和系统状态
"""

import time
import psutil
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QProgressBar, QGroupBox, QFrame, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from ...utils.config import Config
from ...utils.logger import LoggerMixin, register_qt_handler

class StatusPanel(QWidget, LoggerMixin):
    """状态面板"""
    
    # 信号定义
    status_updated = pyqtSignal(str)  # 状态更新信号
    performance_updated = pyqtSignal(dict)  # 性能数据更新信号
    log_received = pyqtSignal(dict)  # 日志接收信号
    
    def __init__(self, config: Config, parent: Optional[QWidget] = None):
        """初始化状态面板
        
        Args:
            config: 配置对象
            parent: 父窗口
        """
        super().__init__(parent)
        self.config = config
        
        # 状态数据
        self.processing_start_time = None
        self.current_operation = ""
        self.performance_data = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'processing_time': 0.0,
            'fps': 0.0
        }
        
        self.init_ui()
        self.init_timers()
        
        # 注册Qt日志处理器
        self.qt_log_handler = register_qt_handler("NightVision", self.log_received)
        self.log_received.connect(self.on_log_received)
        
        self.logger.info("状态面板初始化完成")
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 进度区域
        self.create_progress_area(layout)
        
        # 性能监控区域
        self.create_performance_area(layout)
        
        # 日志区域
        self.create_log_area(layout)
        
        # 设置样式
        self.setStyleSheet(self.get_stylesheet())
    
    def create_progress_area(self, parent_layout: QVBoxLayout):
        """创建进度区域"""
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout(progress_group)
        
        # 当前操作标签
        self.operation_label = QLabel("就绪")
        self.operation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        self.operation_label.setFont(font)
        progress_layout.addWidget(self.operation_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        # 详细信息
        self.detail_label = QLabel("")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setStyleSheet("color: #888888; font-size: 11px;")
        progress_layout.addWidget(self.detail_label)
        
        parent_layout.addWidget(progress_group)
    
    def create_performance_area(self, parent_layout: QVBoxLayout):
        """创建性能监控区域"""
        perf_group = QGroupBox("性能监控")
        perf_layout = QVBoxLayout(perf_group)
        
        # 创建性能指标网格
        metrics_frame = QFrame()
        metrics_layout = QVBoxLayout(metrics_frame)
        
        # CPU使用率
        cpu_frame = QFrame()
        cpu_layout = QHBoxLayout(cpu_frame)
        cpu_layout.addWidget(QLabel("CPU:"))
        self.cpu_label = QLabel("0%")
        self.cpu_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        cpu_layout.addWidget(self.cpu_label)
        metrics_layout.addWidget(cpu_frame)
        
        # 内存使用率
        memory_frame = QFrame()
        memory_layout = QHBoxLayout(memory_frame)
        memory_layout.addWidget(QLabel("内存:"))
        self.memory_label = QLabel("0%")
        self.memory_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        memory_layout.addWidget(self.memory_label)
        metrics_layout.addWidget(memory_frame)
        
        # GPU使用率
        gpu_frame = QFrame()
        gpu_layout = QHBoxLayout(gpu_frame)
        gpu_layout.addWidget(QLabel("GPU:"))
        self.gpu_label = QLabel("N/A")
        self.gpu_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        gpu_layout.addWidget(self.gpu_label)
        metrics_layout.addWidget(gpu_frame)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        metrics_layout.addWidget(separator)
        
        # 处理时间
        time_frame = QFrame()
        time_layout = QHBoxLayout(time_frame)
        time_layout.addWidget(QLabel("处理时间:"))
        self.time_label = QLabel("0.0s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        time_layout.addWidget(self.time_label)
        metrics_layout.addWidget(time_frame)
        
        # FPS
        fps_frame = QFrame()
        fps_layout = QHBoxLayout(fps_frame)
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_label = QLabel("0.0")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        fps_layout.addWidget(self.fps_label)
        metrics_layout.addWidget(fps_frame)
        
        perf_layout.addWidget(metrics_frame)
        parent_layout.addWidget(perf_group)
    
    def create_log_area(self, parent_layout: QVBoxLayout):
        """创建日志区域"""
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout(log_group)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)  # 增加高度以显示更多日志
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setAcceptRichText(True)  # 支持富文本
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        parent_layout.addWidget(log_group)
    
    def init_timers(self):
        """初始化定时器"""
        # 性能监控定时器
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance)
        self.perf_timer.start(1000)  # 每秒更新一次
        
        # 处理时间更新定时器
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_processing_time)
    
    def get_stylesheet(self) -> str:
        """获取样式表"""
        return """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #3c3c3c;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 5px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QFrame {
            background-color: transparent;
            border: none;
        }
        
        QProgressBar {
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            text-align: center;
            background-color: #1e1e1e;
        }
        
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 2px;
        }
        
        QTextEdit {
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QLabel {
            color: #ffffff;
        }
        """
    
    def start_operation(self, operation: str, total_steps: int = 100):
        """开始操作
        
        Args:
            operation: 操作名称
            total_steps: 总步数
        """
        self.current_operation = operation
        self.processing_start_time = time.time()
        
        self.operation_label.setText(operation)
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(0)
        self.detail_label.setText("")
        
        # 开始处理时间更新
        self.time_timer.start(100)  # 每100ms更新一次
        
        self.add_log(f"开始操作: {operation}")
        self.status_updated.emit(f"开始: {operation}")
    
    def update_progress(self, current: int, detail: str = ""):
        """更新进度
        
        Args:
            current: 当前进度
            detail: 详细信息
        """
        self.progress_bar.setValue(current)
        if detail:
            self.detail_label.setText(detail)
        
        # 计算百分比
        if self.progress_bar.maximum() > 0:
            percentage = (current / self.progress_bar.maximum()) * 100
            self.status_updated.emit(f"{self.current_operation}: {percentage:.1f}%")
    
    def finish_operation(self, success: bool = True, message: str = ""):
        """完成操作
        
        Args:
            success: 是否成功
            message: 完成消息
        """
        self.time_timer.stop()
        
        if success:
            self.operation_label.setText("完成")
            self.progress_bar.setValue(self.progress_bar.maximum())
            status_msg = f"完成: {self.current_operation}"
        else:
            self.operation_label.setText("失败")
            status_msg = f"失败: {self.current_operation}"
        
        if message:
            self.detail_label.setText(message)
            status_msg += f" - {message}"
        
        # 计算总处理时间
        if self.processing_start_time:
            total_time = time.time() - self.processing_start_time
            self.performance_data['processing_time'] = total_time
            self.time_label.setText(f"{total_time:.2f}s")
            
            # 计算FPS（如果适用）
            if total_time > 0:
                fps = 1.0 / total_time
                self.performance_data['fps'] = fps
                self.fps_label.setText(f"{fps:.1f}")
        
        self.add_log(status_msg)
        self.status_updated.emit(status_msg)
        
        # 重置状态
        self.current_operation = ""
        self.processing_start_time = None
    
    def update_performance(self):
        """更新性能指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            self.performance_data['cpu_usage'] = cpu_percent
            self.cpu_label.setText(f"{cpu_percent:.1f}%")
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.performance_data['memory_usage'] = memory_percent
            self.memory_label.setText(f"{memory_percent:.1f}%")
            
            # GPU使用率（如果可用）
            gpu_usage = self.get_gpu_usage()
            if gpu_usage is not None:
                self.performance_data['gpu_usage'] = gpu_usage
                self.gpu_label.setText(f"{gpu_usage:.1f}%")
            else:
                self.gpu_label.setText("N/A")
                
        except Exception as e:
            self.logger.warning(f"性能监控更新失败: {e}")
    
    def get_gpu_usage(self) -> Optional[float]:
        """获取GPU使用率
        
        Returns:
            GPU使用率百分比，如果无法获取则返回None
        """
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"GPU使用率获取失败: {e}")
        
        return None
    
    def update_processing_time(self):
        """更新处理时间显示"""
        if self.processing_start_time:
            elapsed = time.time() - self.processing_start_time
            self.time_label.setText(f"{elapsed:.1f}s")
    
    def add_log(self, message: str):
        """添加本地日志消息（用于状态面板内部消息）
        
        Args:
            message: 日志消息
        """
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"<span style='color:#CCCCCC'>[{timestamp}] [STATUS] {message}</span>"
        
        # 添加到日志文本框
        self.log_text.append(formatted_message)
        
        # 限制日志行数
        document = self.log_text.document()
        if document.blockCount() > 100:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
        
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
        self.add_log("日志已清空")
    
    def get_performance_data(self) -> Dict[str, Any]:
        """获取性能数据
        
        Returns:
            性能数据字典
        """
        return self.performance_data.copy()
    
    def set_status(self, status: str):
        """设置状态
        
        Args:
            status: 状态文本
        """
        self.operation_label.setText(status)
        self.add_log(status)
        self.status_updated.emit(status)
    
    def reset_progress(self):
        """重置进度"""
        self.progress_bar.setValue(0)
        self.detail_label.setText("")
        self.operation_label.setText("就绪")
        
        if self.time_timer.isActive():
            self.time_timer.stop()
        
        self.current_operation = ""
        self.processing_start_time = None
    
    def show_error(self, error_message: str):
        """显示错误信息
        
        Args:
            error_message: 错误消息
        """
        self.operation_label.setText("错误")
        self.operation_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        self.detail_label.setText(error_message)
        self.detail_label.setStyleSheet("color: #ff6b6b;")
        
        self.add_log(f"错误: {error_message}")
        self.status_updated.emit(f"错误: {error_message}")
        
        # 3秒后恢复正常样式
        QTimer.singleShot(3000, self.reset_error_style)
    
    def reset_error_style(self):
        """重置错误样式"""
        self.operation_label.setStyleSheet("")
        self.detail_label.setStyleSheet("color: #888888; font-size: 11px;")
    
    def on_log_received(self, log_entry):
        """处理接收到的日志消息
        
        Args:
            log_entry: 日志条目，包含timestamp、level、message等字段
        """
        # 只显示INFO级别以上的日志（INFO、WARNING、ERROR、CRITICAL）
        if log_entry['level'] in ['INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            # 使用不同颜色显示不同级别的日志
            color = {
                'INFO': '#FFFFFF',      # 白色
                'WARNING': '#FFA500',   # 橙色
                'ERROR': '#FF6B6B',     # 红色
                'CRITICAL': '#FF0000'    # 亮红色
            }.get(log_entry['level'], '#FFFFFF')
            
            # 格式化日志消息
            timestamp = log_entry['timestamp'].split()[1]  # 只取时间部分
            message = log_entry['message']
            formatted_message = f"<span style='color:{color}'>[{timestamp}] [{log_entry['level']}] {message}</span>"
            
            # 添加到日志文本框
            self.log_text.append(formatted_message)
            
            # 限制日志行数
            document = self.log_text.document()
            if document.blockCount() > 100:
                cursor = self.log_text.textCursor()
                cursor.movePosition(cursor.MoveOperation.Start)
                cursor.select(cursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
            
            # 滚动到底部
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """关闭事件处理"""
        # 停止定时器
        if hasattr(self, 'perf_timer'):
            self.perf_timer.stop()
        if hasattr(self, 'time_timer'):
            self.time_timer.stop()
        
        super().closeEvent(event)