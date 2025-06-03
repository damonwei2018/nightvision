# -*- coding: utf-8 -*-
"""
主窗口模块

应用程序的主界面窗口
"""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QMenuBar, QStatusBar, QProgressBar,
    QLabel, QFileDialog, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence, QIcon

from ..utils.config import Config
from ..utils.logger import LoggerMixin
from .widgets.file_panel import FilePanel
from .widgets.control_panel import ControlPanel
from .widgets.preview_panel import PreviewPanel
from .widgets.status_panel import StatusPanel
from ..processing.image_processor import NightVisionProcessor

class ImageProcessingThread(QThread):
    """图像处理线程类"""
    
    progress_updated = pyqtSignal(int, str)  # 进度更新信号
    processing_finished = pyqtSignal(object)  # 处理完成信号
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, processor, image_path, params):
        """初始化图像处理线程
        
        Args:
            processor: 图像处理器
            image_path: 图像路径
            params: 处理参数
        """
        super().__init__()
        self.processor = processor
        self.image_path = image_path
        self.params = params
    
    def run(self):
        """线程执行函数"""
        try:
            def progress_callback(progress, message):
                self.progress_updated.emit(progress, message)
            
            # 处理输出尺寸参数
            if 'output_size' in self.params:
                output_size_text = self.params['output_size']
                output_size = None
                
                if output_size_text == "原始尺寸":
                    output_size = None
                elif output_size_text == "自定义":
                    if 'custom_width' in self.params and 'custom_height' in self.params:
                        width = self.params['custom_width']
                        height = self.params['custom_height']
                        output_size = (width, height)
                else:
                    # 处理如 "1920x1080" 格式的尺寸
                    try:
                        width, height = output_size_text.split('x')
                        output_size = (int(width), int(height))
                    except (ValueError, AttributeError):
                        output_size = None
                
                # 更新处理器的输出尺寸
                self.processor.output_size = output_size
            
            # 应用其他处理参数到处理器
            for key, value in self.params.items():
                if key != 'output_size' and hasattr(self.processor, key):
                    setattr(self.processor, key, value)
            
            # 调用处理方法
            result = self.processor.process_image(self.image_path, progress_callback)
            self.processing_finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class MainWindow(QMainWindow, LoggerMixin):
    """主窗口类"""
    
    # 信号定义
    image_loaded = pyqtSignal(str)  # 图像加载信号
    processing_started = pyqtSignal()  # 处理开始信号
    processing_finished = pyqtSignal(str)  # 处理完成信号
    
    def __init__(self, config: Config, parent: Optional[QWidget] = None):
        """初始化主窗口
        
        Args:
            config: 配置对象
            parent: 父窗口
        """
        super().__init__(parent)
        self.config = config
        self.current_image_path = None
        self.processor = None
        
        # 初始化UI
        self.init_ui()
        self.init_processor()
        self.connect_signals()
        
        # 恢复窗口状态
        self.restore_window_state()
        
        self.logger.info("主窗口初始化完成")
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("NightVision - 夜视图像处理")
        self.setMinimumSize(1000, 700)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 创建分割器
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # 创建面板
        self.file_panel = FilePanel(self.config)
        self.control_panel = ControlPanel(self.config)
        self.preview_panel = PreviewPanel(self.config)
        self.status_panel = StatusPanel(self.config)
        
        # 添加面板到分割器
        self.splitter.addWidget(self.file_panel)
        self.splitter.addWidget(self.preview_panel)
        self.splitter.addWidget(self.control_panel)
        
        # 设置分割器比例
        splitter_sizes = self.config.get('ui.splitter_sizes', [300, 600, 300])
        self.splitter.setSizes(splitter_sizes)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 设置样式
        self.setStyleSheet(self.get_custom_stylesheet())
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        # 打开文件
        open_action = QAction('打开图片(&O)', self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        # 保存文件
        save_action = QAction('保存结果(&S)', self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction('退出(&X)', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu('编辑(&E)')
        
        # 重置参数
        reset_action = QAction('重置参数(&R)', self)
        reset_action.triggered.connect(self.reset_parameters)
        edit_menu.addAction(reset_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图(&V)')
        
        # 全屏预览
        fullscreen_action = QAction('全屏预览(&F)', self)
        fullscreen_action.setShortcut(Qt.Key.Key_F11)
        fullscreen_action.triggered.connect(self.toggle_fullscreen_preview)
        view_menu.addAction(fullscreen_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        # 关于
        about_action = QAction('关于(&A)', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # 性能指标标签
        self.performance_label = QLabel("")
        self.status_bar.addPermanentWidget(self.performance_label)
    
    def init_processor(self):
        """初始化图像处理器"""
        try:
            self.processor = NightVisionProcessor(self.config)
            self.logger.info("图像处理器初始化成功")
        except Exception as e:
            self.logger.error(f"图像处理器初始化失败: {e}")
            QMessageBox.warning(self, "警告", f"图像处理器初始化失败: {e}")
    
    def connect_signals(self):
        """连接信号和槽 - 添加安全检查"""
        # 文件面板信号
        if hasattr(self.file_panel, 'file_selected'):
            self.file_panel.file_selected.connect(self.load_image)
        if hasattr(self.file_panel, 'batch_process_requested'):
            self.file_panel.batch_process_requested.connect(self.start_batch_processing)
        
        # 控制面板信号
        if hasattr(self.control_panel, 'parameter_changed'):
            self.control_panel.parameter_changed.connect(self.on_parameter_changed)
        if hasattr(self.control_panel, 'process_requested'):
            self.control_panel.process_requested.connect(self.process_current_image)
        
        # 预览面板信号
        if hasattr(self.preview_panel, 'zoom_changed'):
            self.preview_panel.zoom_changed.connect(self.on_zoom_changed)
        if hasattr(self.preview_panel, 'save_requested'):
            self.preview_panel.save_requested.connect(self.save_result)
        
        # 状态面板信号
        if hasattr(self.status_panel, 'performance_updated'):
            self.status_panel.performance_updated.connect(self.update_performance_display)
        if hasattr(self.status_panel, 'status_updated'):
            self.status_panel.status_updated.connect(self.update_status)
        
        # 主窗口信号
        if hasattr(self.preview_panel, 'show_original'):
            self.image_loaded.connect(self.preview_panel.show_original)
        if hasattr(self.status_panel, 'update_status'):
            self.processing_started.connect(lambda: self.status_panel.update_status({"status": "processing"}))
            self.processing_finished.connect(lambda _: self.status_panel.update_status({"status": "ready"}))
        
        # 性能监控定时器
        self.performance_timer = QTimer(self)
        self.performance_timer.timeout.connect(self.update_performance_metrics)
        self.performance_timer.start(1000)  # 每秒更新一次
    
    def get_custom_stylesheet(self) -> str:
        """获取自定义样式表"""
        return """
        QMainWindow {
            background-color: #2b2b2b;
        }
        
        QSplitter::handle {
            background-color: #3c3c3c;
            width: 2px;
        }
        
        QSplitter::handle:hover {
            background-color: #0078d4;
        }
        
        QMenuBar {
            background-color: #2b2b2b;
            color: #ffffff;
            border-bottom: 1px solid #3c3c3c;
        }
        
        QMenuBar::item {
            padding: 4px 8px;
            background-color: transparent;
        }
        
        QMenuBar::item:selected {
            background-color: #3c3c3c;
        }
        
        QStatusBar {
            background-color: #2b2b2b;
            color: #ffffff;
            border-top: 1px solid #3c3c3c;
        }
        """
    
    def open_image(self):
        """打开图片文件"""
        last_dir = self.config.get('paths.last_input_dir', '')
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片文件",
            last_dir,
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;所有文件 (*)"
        )
        
        if file_path:
            self.load_image(file_path)
            # 保存最后使用的目录
            self.config.set('paths.last_input_dir', str(Path(file_path).parent))
            self.config.save_config()
    
    def load_image(self, file_path: str):
        """加载图片
        
        Args:
            file_path: 图片文件路径
        """
        try:
            self.current_image_path = file_path
            self.preview_panel.load_image(file_path)
            self.status_label.setText(f"已加载: {Path(file_path).name}")
            self.image_loaded.emit(file_path)
            self.logger.info(f"图片加载成功: {file_path}")
        except Exception as e:
            self.logger.error(f"图片加载失败: {e}")
            QMessageBox.critical(self, "错误", f"图片加载失败: {e}")
    
    def save_result(self):
        """保存处理结果"""
        if not self.preview_panel.has_result():
            QMessageBox.information(self, "提示", "没有可保存的处理结果")
            return
        
        last_dir = self.config.get('paths.last_output_dir', '')
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存处理结果",
            last_dir,
            "PNG文件 (*.png);;JPEG文件 (*.jpg);;所有文件 (*)"
        )
        
        if file_path:
            try:
                self.preview_panel.save_result(file_path)
                self.status_label.setText(f"已保存: {Path(file_path).name}")
                # 保存最后使用的目录
                self.config.set('paths.last_output_dir', str(Path(file_path).parent))
                self.config.save_config()
                self.logger.info(f"结果保存成功: {file_path}")
            except Exception as e:
                self.logger.error(f"结果保存失败: {e}")
                QMessageBox.critical(self, "错误", f"结果保存失败: {e}")
    
    def on_parameter_changed(self, param_name: str, value):
        """参数变化处理
        
        Args:
            param_name: 参数名称
            value: 参数值
        """
        self.logger.debug(f"参数变化: {param_name} = {value}")
        
        # 如果启用了实时预览，则自动处理
        if self.control_panel.is_realtime_preview_enabled() and self.current_image_path:
            # 使用定时器延迟处理，避免频繁更新
            if not hasattr(self, 'preview_timer'):
                self.preview_timer = QTimer()
                self.preview_timer.setSingleShot(True)
                self.preview_timer.timeout.connect(self.process_current_image)
            
            delay = self.config.get('ui.preview_update_delay', 500)
            self.preview_timer.start(delay)
    
    def process_current_image(self):
        """处理当前图片 - 使用线程异步处理"""
        if not self.current_image_path:
            QMessageBox.information(self, "提示", "请先加载图片")
            return
        
        if not self.processor:
            QMessageBox.warning(self, "警告", "图像处理器未初始化")
            return
        
        # 显示进度
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)  # 设置进度范围
        self.progress_bar.setValue(0)
        self.status_label.setText("正在处理...")
        
        # 获取处理参数
        params = self.control_panel.get_parameters()
        
        # 开始处理
        self.processing_started.emit()
        
        # 创建并启动处理线程
        self.processing_thread = ImageProcessingThread(
            self.processor, 
            self.current_image_path, 
            params
        )
        
        # 连接线程信号
        self.processing_thread.progress_updated.connect(self.on_processing_progress)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        
        # 启动线程
        self.processing_thread.start()
    
    def on_processing_progress(self, progress, message):
        """处理进度更新
        
        Args:
            progress: 进度值(0-100)
            message: 进度消息
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def on_processing_finished(self, result_image):
        """处理完成
        
        Args:
            result_image: 处理结果图像
        """
        # 显示结果
        if hasattr(self.preview_panel, 'show_result'):
            self.preview_panel.show_result(result_image)
        
        # 隐藏进度
        self.progress_bar.setVisible(False)
        self.status_label.setText("处理完成")
        
        self.processing_finished.emit(self.current_image_path)
        
        # 清理线程
        if hasattr(self, 'processing_thread'):
            self.processing_thread.deleteLater()
    
    def on_processing_error(self, error_message):
        """处理错误
        
        Args:
            error_message: 错误消息
        """
        self.progress_bar.setVisible(False)
        self.status_label.setText("处理失败")
        self.logger.error(f"图像处理失败: {error_message}")
        QMessageBox.critical(self, "错误", f"图像处理失败: {error_message}")
        
        # 清理线程
        if hasattr(self, 'processing_thread'):
            self.processing_thread.deleteLater()
    
    def start_batch_processing(self, file_paths: list):
        """开始批量处理
        
        Args:
            file_paths: 文件路径列表
        """
        # TODO: 实现批量处理功能
        QMessageBox.information(self, "提示", "批量处理功能正在开发中")
    
    def reset_parameters(self):
        """重置参数"""
        self.control_panel.reset_parameters()
        self.status_label.setText("参数已重置")
    
    def toggle_fullscreen_preview(self):
        """切换全屏预览"""
        self.preview_panel.toggle_fullscreen()
    
    def on_zoom_changed(self, zoom_factor: float):
        """缩放变化处理
        
        Args:
            zoom_factor: 缩放因子
        """
        self.status_label.setText(f"缩放: {zoom_factor:.1%}")
    
    def update_performance_display(self, metrics: dict):
        """更新性能显示
        
        Args:
            metrics: 性能指标字典
        """
        if 'processing_time' in metrics:
            time_str = f"处理时间: {metrics['processing_time']:.2f}s"
            if 'memory_usage' in metrics:
                time_str += f" | 内存: {metrics['memory_usage']:.1f}MB"
            self.performance_label.setText(time_str)
    
    def update_status(self, status_info: dict):
        """更新状态信息
        
        Args:
            status_info: 状态信息字典
        """
        if 'status' in status_info:
            status = status_info['status']
            if status == 'ready':
                self.status_label.setText("就绪")
            elif status == 'processing':
                self.status_label.setText("处理中...")
            elif status == 'error':
                error_msg = status_info.get('message', '未知错误')
                self.status_label.setText(f"错误: {error_msg}")
            else:
                self.status_label.setText(status)
    
    def update_performance_metrics(self):
        """更新性能指标"""
        import psutil
        import os
        
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 获取内存使用情况
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # 转换为MB
        
        # 获取CPU使用率
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # 更新状态面板
        if hasattr(self.status_panel, 'performance_updated'):
            metrics = {
                'memory_usage': memory_mb,
                'cpu_percent': cpu_percent
            }
            self.status_panel.performance_updated.emit(metrics)
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于 NightVision",
            "<h3>NightVision 1.0.0</h3>"
            "<p>专业的夜视图像处理应用</p>"
            "<p>基于 Zero-DCE 和 DexiNed 深度学习模型</p>"
            "<p>© 2024 NightVision Team</p>"
        )
    
    def restore_window_state(self):
        """恢复窗口状态"""
        # 恢复窗口大小和位置
        size = self.config.get('ui.window_size', [1200, 800])
        position = self.config.get('ui.window_position', [100, 100])
        
        self.resize(size[0], size[1])
        self.move(position[0], position[1])
    
    def save_window_state(self):
        """保存窗口状态"""
        # 保存窗口大小和位置
        self.config.set('ui.window_size', [self.width(), self.height()])
        self.config.set('ui.window_position', [self.x(), self.y()])
        
        # 保存分割器状态
        self.config.set('ui.splitter_sizes', self.splitter.sizes())
        
        self.config.save_config()
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 保存窗口状态
        self.save_window_state()
        
        # 清理资源
        if self.processor:
            self.processor.cleanup()
        
        self.logger.info("应用程序关闭")
        event.accept()