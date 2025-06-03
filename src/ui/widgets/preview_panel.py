# -*- coding: utf-8 -*-
"""
预览面板

负责图像的预览显示和对比
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QScrollArea, QSplitter, QPushButton, QSlider,
    QGroupBox, QCheckBox, QComboBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QRect
from PyQt6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent

from ...utils.config import Config
from ...utils.logger import LoggerMixin

class ImageLabel(QLabel):
    """支持缩放和拖拽的图像标签"""
    
    zoom_changed = pyqtSignal(float)  # 缩放变化信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("border: 1px solid #3c3c3c; background-color: #1e1e1e;")
        
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        
        # 拖拽相关
        self.dragging = False
        self.last_pan_point = None
        self.pan_offset = [0, 0]
    
    def set_image(self, pixmap: QPixmap):
        """设置图像
        
        Args:
            pixmap: 图像像素图
        """
        self.original_pixmap = pixmap
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        # 自动缩放到适合窗口
        self.zoom_to_fit()
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        if self.original_pixmap is None:
            self.clear()
            return
        
        # 计算缩放后的尺寸
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.original_pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)
        self.zoom_changed.emit(self.zoom_factor)
    
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮事件 - 缩放"""
        if self.original_pixmap is None:
            return
        
        # 计算缩放因子
        delta = event.angleDelta().y()
        zoom_in = delta > 0
        zoom_step = 1.1 if zoom_in else 1.0 / 1.1
        
        new_zoom = self.zoom_factor * zoom_step
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.update_display()
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件 - 开始拖拽"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_pan_point = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件 - 拖拽"""
        if self.dragging and self.last_pan_point is not None:
            delta = event.position().toPoint() - self.last_pan_point
            self.pan_offset[0] += delta.x()
            self.pan_offset[1] += delta.y()
            self.last_pan_point = event.position().toPoint()
            # 这里可以实现图像平移逻辑
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件 - 结束拖拽"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def reset_zoom(self):
        """重置缩放"""
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.update_display()
    
    def zoom_to_fit(self):
        """缩放到适合窗口"""
        if self.original_pixmap is None:
            return
        
        widget_size = self.size()
        pixmap_size = self.original_pixmap.size()
        
        scale_x = widget_size.width() / pixmap_size.width()
        scale_y = widget_size.height() / pixmap_size.height()
        
        self.zoom_factor = min(scale_x, scale_y, 1.0)
        self.pan_offset = [0, 0]
        self.update_display()

class PreviewPanel(QWidget, LoggerMixin):
    """预览面板"""
    
    # 信号定义
    zoom_changed = pyqtSignal(float)  # 缩放变化信号
    
    def __init__(self, config: Config, parent: Optional[QWidget] = None):
        """初始化预览面板
        
        Args:
            config: 配置对象
            parent: 父窗口
        """
        super().__init__(parent)
        self.config = config
        self.current_image_path = None
        self.result_image = None
        
        self.init_ui()
        
        self.logger.info("预览面板初始化完成")
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 控制栏
        self.create_control_bar(layout)
        
        # 预览区域
        self.create_preview_area(layout)
        
        # 状态栏
        self.create_status_bar(layout)
        
        # 设置样式
        self.setStyleSheet(self.get_stylesheet())
    
    def create_control_bar(self, parent_layout: QVBoxLayout):
        """创建控制栏"""
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        # 显示模式选择
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["分屏对比", "仅原图", "仅结果", "叠加对比"])
        self.view_mode_combo.currentTextChanged.connect(self.change_view_mode)
        control_layout.addWidget(QLabel("显示模式:"))
        control_layout.addWidget(self.view_mode_combo)
        
        control_layout.addWidget(self.create_separator())
        
        # 缩放控制
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 500)  # 10% - 500%
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        control_layout.addWidget(QLabel("缩放:"))
        control_layout.addWidget(self.zoom_slider)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(40)
        control_layout.addWidget(self.zoom_label)
        
        control_layout.addWidget(self.create_separator())
        
        # 缩放按钮
        self.zoom_fit_btn = QPushButton("适合窗口")
        self.zoom_fit_btn.clicked.connect(self.zoom_to_fit)
        control_layout.addWidget(self.zoom_fit_btn)
        
        self.zoom_100_btn = QPushButton("100%")
        self.zoom_100_btn.clicked.connect(self.zoom_to_100)
        control_layout.addWidget(self.zoom_100_btn)
        
        control_layout.addWidget(self.create_separator())
        
        # 全屏按钮
        self.fullscreen_btn = QPushButton("全屏")
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        control_layout.addWidget(self.fullscreen_btn)
        
        control_layout.addStretch()
        
        parent_layout.addWidget(control_frame)
    
    def create_separator(self) -> QFrame:
        """创建分隔线"""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #3c3c3c;")
        return separator
    
    def create_preview_area(self, parent_layout: QVBoxLayout):
        """创建预览区域"""
        # 创建分割器
        self.preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 原图区域
        original_group = QGroupBox("原图")
        original_layout = QVBoxLayout(original_group)
        
        self.original_scroll = QScrollArea()
        self.original_scroll.setWidgetResizable(True)
        self.original_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.original_label = ImageLabel()
        self.original_label.zoom_changed.connect(self.on_original_zoom_changed)
        self.original_scroll.setWidget(self.original_label)
        
        original_layout.addWidget(self.original_scroll)
        
        # 结果区域
        result_group = QGroupBox("处理结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.result_label = ImageLabel()
        self.result_label.zoom_changed.connect(self.on_result_zoom_changed)
        self.result_scroll.setWidget(self.result_label)
        
        result_layout.addWidget(self.result_scroll)
        
        # 添加到分割器
        self.preview_splitter.addWidget(original_group)
        self.preview_splitter.addWidget(result_group)
        self.preview_splitter.setSizes([400, 400])
        
        parent_layout.addWidget(self.preview_splitter)
    
    def create_status_bar(self, parent_layout: QVBoxLayout):
        """创建状态栏"""
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 5, 5, 5)
        
        # 图像信息
        self.image_info_label = QLabel("未加载图像")
        status_layout.addWidget(self.image_info_label)
        
        status_layout.addStretch()
        
        # 鼠标位置和像素值
        self.pixel_info_label = QLabel("")
        status_layout.addWidget(self.pixel_info_label)
        
        parent_layout.addWidget(status_frame)
    
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
            background-color: #2b2b2b;
            border: 1px solid #3c3c3c;
        }
        
        QPushButton {
            background-color: #0078d4;
            border: none;
            color: white;
            padding: 4px 8px;
            border-radius: 3px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #106ebe;
        }
        
        QPushButton:pressed {
            background-color: #005a9e;
        }
        
        QComboBox {
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            padding: 3px;
            background-color: #1e1e1e;
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #3c3c3c;
            height: 4px;
            background: #1e1e1e;
            border-radius: 2px;
        }
        
        QSlider::handle:horizontal {
            background: #0078d4;
            border: 1px solid #0078d4;
            width: 12px;
            margin: -4px 0;
            border-radius: 6px;
        }
        
        QScrollArea {
            border: 1px solid #3c3c3c;
            background-color: #1e1e1e;
        }
        
        QSplitter::handle {
            background-color: #3c3c3c;
            width: 2px;
        }
        
        QSplitter::handle:hover {
            background-color: #0078d4;
        }
        """
    
    def load_image(self, file_path: str):
        """加载图像
        
        Args:
            file_path: 图像文件路径
        """
        try:
            self.current_image_path = file_path
            
            # 使用PIL加载图像
            pil_image = Image.open(file_path)
            
            # 转换为QPixmap
            pixmap = self.pil_to_qpixmap(pil_image)
            
            # 显示原图
            self.original_label.set_image(pixmap)
            
            # 清空结果
            self.result_label.clear()
            self.result_image = None
            
            # 更新图像信息
            self.update_image_info(pil_image)
            
            self.logger.info(f"图像加载成功: {file_path}")
            
        except Exception as e:
            self.logger.error(f"图像加载失败: {e}")
            raise
    
    def show_result(self, result_image):
        """显示处理结果
        
        Args:
            result_image: 处理结果图像 (PIL Image 或 numpy array)
        """
        try:
            self.result_image = result_image
            
            # 转换为PIL Image（如果需要）
            if isinstance(result_image, np.ndarray):
                if result_image.dtype != np.uint8:
                    result_image = (result_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(result_image)
            else:
                pil_image = result_image
            
            # 转换为QPixmap
            pixmap = self.pil_to_qpixmap(pil_image)
            
            # 显示结果
            self.result_label.set_image(pixmap)
            
            self.logger.debug("处理结果显示成功")
            
        except Exception as e:
            self.logger.error(f"结果显示失败: {e}")
            raise
    
    def pil_to_qpixmap(self, pil_image: Image.Image) -> QPixmap:
        """将PIL图像转换为QPixmap
        
        Args:
            pil_image: PIL图像
            
        Returns:
            QPixmap对象
        """
        # 确保图像为RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 转换为numpy数组
        np_array = np.array(pil_image)
        height, width, channel = np_array.shape
        bytes_per_line = 3 * width
        
        # 创建QImage
        q_image = QImage(
            np_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # 转换为QPixmap
        return QPixmap.fromImage(q_image)
    
    def update_image_info(self, pil_image: Image.Image):
        """更新图像信息显示
        
        Args:
            pil_image: PIL图像
        """
        width, height = pil_image.size
        mode = pil_image.mode
        
        # 计算文件大小
        if self.current_image_path:
            file_size = os.path.getsize(self.current_image_path)
            size_str = self.format_file_size(file_size)
            file_name = Path(self.current_image_path).name
            info_text = f"{file_name} | {width}×{height} | {mode} | {size_str}"
        else:
            info_text = f"{width}×{height} | {mode}"
        
        self.image_info_label.setText(info_text)
    
    def format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小
        
        Args:
            size_bytes: 字节数
            
        Returns:
            格式化的大小字符串
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    def change_view_mode(self, mode: str):
        """改变显示模式
        
        Args:
            mode: 显示模式
        """
        if mode == "分屏对比":
            self.preview_splitter.widget(0).setVisible(True)
            self.preview_splitter.widget(1).setVisible(True)
            self.preview_splitter.setOrientation(Qt.Orientation.Horizontal)
        elif mode == "仅原图":
            self.preview_splitter.widget(0).setVisible(True)
            self.preview_splitter.widget(1).setVisible(False)
        elif mode == "仅结果":
            self.preview_splitter.widget(0).setVisible(False)
            self.preview_splitter.widget(1).setVisible(True)
        elif mode == "叠加对比":
            # TODO: 实现叠加对比模式
            pass
    
    def on_zoom_slider_changed(self, value: int):
        """缩放滑块变化处理
        
        Args:
            value: 滑块值 (10-500)
        """
        zoom_factor = value / 100.0
        self.zoom_label.setText(f"{value}%")
        
        # 同步两个图像标签的缩放
        self.original_label.zoom_factor = zoom_factor
        self.original_label.update_display()
        
        self.result_label.zoom_factor = zoom_factor
        self.result_label.update_display()
        
        self.zoom_changed.emit(zoom_factor)
    
    def on_original_zoom_changed(self, zoom_factor: float):
        """原图缩放变化处理
        
        Args:
            zoom_factor: 缩放因子
        """
        # 防止循环调用
        if hasattr(self, '_updating_zoom') and self._updating_zoom:
            return
            
        self._updating_zoom = True
        try:
            # 同步缩放滑块
            self.zoom_slider.setValue(int(zoom_factor * 100))
            
            # 同步结果图缩放（不触发信号）
            self.result_label.zoom_factor = zoom_factor
            if self.result_label.original_pixmap is not None:
                scaled_size = self.result_label.original_pixmap.size() * zoom_factor
                scaled_pixmap = self.result_label.original_pixmap.scaled(
                    scaled_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.result_label.setPixmap(scaled_pixmap)
        finally:
            self._updating_zoom = False
    
    def on_result_zoom_changed(self, zoom_factor: float):
        """结果图缩放变化处理
        
        Args:
            zoom_factor: 缩放因子
        """
        # 防止循环调用
        if hasattr(self, '_updating_zoom') and self._updating_zoom:
            return
            
        self._updating_zoom = True
        try:
            # 同步缩放滑块
            self.zoom_slider.setValue(int(zoom_factor * 100))
            
            # 同步原图缩放（不触发信号）
            self.original_label.zoom_factor = zoom_factor
            if self.original_label.original_pixmap is not None:
                scaled_size = self.original_label.original_pixmap.size() * zoom_factor
                scaled_pixmap = self.original_label.original_pixmap.scaled(
                    scaled_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.original_label.setPixmap(scaled_pixmap)
        finally:
            self._updating_zoom = False
    
    def zoom_to_fit(self):
        """缩放到适合窗口"""
        self.original_label.zoom_to_fit()
        self.result_label.zoom_to_fit()
    
    def zoom_to_100(self):
        """缩放到100%"""
        self.original_label.reset_zoom()
        self.result_label.reset_zoom()
        self.zoom_slider.setValue(100)
    
    def toggle_fullscreen(self):
        """切换全屏模式"""
        # TODO: 实现全屏预览功能
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "提示", "全屏预览功能正在开发中")
    
    def has_result(self) -> bool:
        """检查是否有处理结果
        
        Returns:
            是否有处理结果
        """
        return self.result_image is not None
    
    def save_result(self, file_path: str):
        """保存处理结果
        
        Args:
            file_path: 保存路径
        """
        if not self.has_result():
            raise ValueError("没有可保存的处理结果")
        
        try:
            # 转换为PIL Image（如果需要）
            if isinstance(self.result_image, np.ndarray):
                if self.result_image.dtype != np.uint8:
                    result_image = (self.result_image * 255).astype(np.uint8)
                else:
                    result_image = self.result_image
                pil_image = Image.fromarray(result_image)
            else:
                pil_image = self.result_image
            
            # 保存图像
            pil_image.save(file_path)
            
            self.logger.info(f"处理结果保存成功: {file_path}")
            
        except Exception as e:
            self.logger.error(f"结果保存失败: {e}")
            raise
    
    def clear_images(self):
        """清空所有图像"""
        self.original_label.clear()
        self.result_label.clear()
        self.current_image_path = None
        self.result_image = None
        self.image_info_label.setText("未加载图像")
        self.pixel_info_label.setText("")