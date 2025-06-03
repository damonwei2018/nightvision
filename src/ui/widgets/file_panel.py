# -*- coding: utf-8 -*-
"""
文件操作面板

负责文件的打开、保存和批量处理操作
"""

import os
from pathlib import Path
from typing import List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QListWidget, QListWidgetItem, QLabel, QFileDialog,
    QGroupBox, QProgressBar, QCheckBox, QSpinBox,
    QComboBox, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QDragEnterEvent, QDropEvent

from ...utils.config import Config
from ...utils.logger import LoggerMixin

class FilePanel(QWidget, LoggerMixin):
    """文件操作面板"""
    
    # 信号定义
    file_selected = pyqtSignal(str)  # 文件选择信号
    batch_process_requested = pyqtSignal(list)  # 批量处理请求信号
    
    def __init__(self, config: Config, parent: Optional[QWidget] = None):
        """初始化文件面板
        
        Args:
            config: 配置对象
            parent: 父窗口
        """
        super().__init__(parent)
        self.config = config
        self.file_list = []
        
        self.init_ui()
        self.setup_drag_drop()
        
        self.logger.info("文件面板初始化完成")
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 打开文件按钮
        self.open_file_btn = QPushButton("打开文件")
        self.open_file_btn.clicked.connect(self.open_single_file)
        button_layout.addWidget(self.open_file_btn)
        
        # 打开文件夹按钮
        self.open_folder_btn = QPushButton("打开文件夹")
        self.open_folder_btn.clicked.connect(self.open_folder)
        button_layout.addWidget(self.open_folder_btn)
        
        file_layout.addLayout(button_layout)
        
        # 清空列表按钮
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.clear_file_list)
        file_layout.addWidget(self.clear_btn)
        
        layout.addWidget(file_group)
        
        # 文件列表组
        list_group = QGroupBox("文件列表")
        list_layout = QVBoxLayout(list_group)
        
        # 文件列表
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.on_file_item_clicked)
        self.file_list_widget.setAlternatingRowColors(True)
        list_layout.addWidget(self.file_list_widget)
        
        # 文件统计标签
        self.file_count_label = QLabel("文件数量: 0")
        list_layout.addWidget(self.file_count_label)
        
        layout.addWidget(list_group)
        
        # 批量处理组
        batch_group = QGroupBox("批量处理")
        batch_layout = QVBoxLayout(batch_group)
        
        # 输出设置
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("选择输出目录...")
        output_layout.addWidget(self.output_dir_edit)
        
        self.browse_output_btn = QPushButton("浏览")
        self.browse_output_btn.clicked.connect(self.browse_output_directory)
        output_layout.addWidget(self.browse_output_btn)
        
        batch_layout.addLayout(output_layout)
        
        # 输出格式
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("输出格式:"))
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG", "TIFF", "BMP"])
        self.format_combo.setCurrentText(self.config.get('processing.output_format', 'PNG').upper())
        format_layout.addWidget(self.format_combo)
        
        format_layout.addWidget(QLabel("质量:"))
        
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(self.config.get('processing.output_quality', 95))
        self.quality_spin.setSuffix("%")
        format_layout.addWidget(self.quality_spin)
        
        batch_layout.addLayout(format_layout)
        
        # 处理选项
        options_layout = QHBoxLayout()
        
        self.overwrite_check = QCheckBox("覆盖已存在文件")
        options_layout.addWidget(self.overwrite_check)
        
        self.preserve_structure_check = QCheckBox("保持目录结构")
        self.preserve_structure_check.setChecked(True)
        options_layout.addWidget(self.preserve_structure_check)
        
        batch_layout.addLayout(options_layout)
        
        # 批量处理按钮
        self.batch_process_btn = QPushButton("开始批量处理")
        self.batch_process_btn.clicked.connect(self.start_batch_processing)
        self.batch_process_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_process_btn)
        
        # 批量处理进度
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        batch_layout.addWidget(self.batch_progress)
        
        layout.addWidget(batch_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        # 设置样式
        self.setStyleSheet(self.get_stylesheet())
    
    def setup_drag_drop(self):
        """设置拖拽功能"""
        self.setAcceptDrops(True)
        self.file_list_widget.setAcceptDrops(True)
    
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
        
        QPushButton {
            background-color: #0078d4;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #106ebe;
        }
        
        QPushButton:pressed {
            background-color: #005a9e;
        }
        
        QPushButton:disabled {
            background-color: #3c3c3c;
            color: #888888;
        }
        
        QListWidget {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            background-color: #1e1e1e;
            alternate-background-color: #2d2d2d;
        }
        
        QListWidget::item {
            padding: 4px;
            border-bottom: 1px solid #3c3c3c;
        }
        
        QListWidget::item:selected {
            background-color: #0078d4;
        }
        
        QListWidget::item:hover {
            background-color: #2d2d2d;
        }
        
        QLineEdit {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 4px;
            background-color: #1e1e1e;
        }
        
        QComboBox {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 4px;
            background-color: #1e1e1e;
        }
        
        QSpinBox {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 4px;
            background-color: #1e1e1e;
        }
        """
    
    def open_single_file(self):
        """打开单个文件"""
        last_dir = self.config.get('paths.last_input_dir', '')
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片文件",
            last_dir,
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;所有文件 (*)"
        )
        
        if file_path:
            self.add_file(file_path)
            self.file_selected.emit(file_path)
            # 保存最后使用的目录
            self.config.set('paths.last_input_dir', str(Path(file_path).parent))
            self.config.save_config()
    
    def open_folder(self):
        """打开文件夹"""
        last_dir = self.config.get('paths.last_input_dir', '')
        
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择图片文件夹",
            last_dir
        )
        
        if folder_path:
            self.add_folder(folder_path)
            # 保存最后使用的目录
            self.config.set('paths.last_input_dir', folder_path)
            self.config.save_config()
    
    def add_file(self, file_path: str):
        """添加文件到列表
        
        Args:
            file_path: 文件路径
        """
        if file_path not in self.file_list:
            # 检查文件是否为支持的图片格式
            if self.is_supported_image(file_path):
                self.file_list.append(file_path)
                
                # 添加到列表控件
                item = QListWidgetItem(Path(file_path).name)
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                item.setToolTip(file_path)
                self.file_list_widget.addItem(item)
                
                self.update_file_count()
                self.update_batch_button_state()
                
                self.logger.debug(f"添加文件: {file_path}")
    
    def add_folder(self, folder_path: str):
        """添加文件夹中的所有图片
        
        Args:
            folder_path: 文件夹路径
        """
        folder = Path(folder_path)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
        added_count = 0
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                if str(file_path) not in self.file_list:
                    self.add_file(str(file_path))
                    added_count += 1
        
        if added_count > 0:
            self.logger.info(f"从文件夹添加了 {added_count} 个图片文件")
        else:
            QMessageBox.information(self, "提示", "文件夹中没有找到支持的图片文件")
    
    def is_supported_image(self, file_path: str) -> bool:
        """检查文件是否为支持的图片格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为支持的图片格式
        """
        supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        return Path(file_path).suffix.lower() in supported_extensions
    
    def clear_file_list(self):
        """清空文件列表"""
        self.file_list.clear()
        self.file_list_widget.clear()
        self.update_file_count()
        self.update_batch_button_state()
        self.logger.debug("文件列表已清空")
    
    def on_file_item_clicked(self, item: QListWidgetItem):
        """文件项点击处理
        
        Args:
            item: 列表项
        """
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path:
            self.file_selected.emit(file_path)
    
    def update_file_count(self):
        """更新文件数量显示"""
        count = len(self.file_list)
        self.file_count_label.setText(f"文件数量: {count}")
    
    def update_batch_button_state(self):
        """更新批量处理按钮状态"""
        has_files = len(self.file_list) > 0
        has_output_dir = bool(self.output_dir_edit.text().strip())
        self.batch_process_btn.setEnabled(has_files and has_output_dir)
    
    def browse_output_directory(self):
        """浏览输出目录"""
        last_dir = self.config.get('paths.last_output_dir', '')
        
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            last_dir
        )
        
        if folder_path:
            self.output_dir_edit.setText(folder_path)
            self.update_batch_button_state()
            # 保存最后使用的目录
            self.config.set('paths.last_output_dir', folder_path)
            self.config.save_config()
    
    def start_batch_processing(self):
        """开始批量处理"""
        if not self.file_list:
            QMessageBox.information(self, "提示", "请先添加要处理的文件")
            return
        
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.information(self, "提示", "请选择输出目录")
            return
        
        # 检查输出目录是否存在
        output_path = Path(output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法创建输出目录: {e}")
                return
        
        # 准备批量处理参数
        batch_params = {
            'files': self.file_list.copy(),
            'output_dir': output_dir,
            'output_format': self.format_combo.currentText().lower(),
            'output_quality': self.quality_spin.value(),
            'overwrite': self.overwrite_check.isChecked(),
            'preserve_structure': self.preserve_structure_check.isChecked()
        }
        
        # 发送批量处理请求信号
        self.batch_process_requested.emit(self.file_list)
        
        self.logger.info(f"开始批量处理 {len(self.file_list)} 个文件")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """拖拽放下事件"""
        urls = event.mimeData().urls()
        for url in urls:
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                self.add_file(file_path)
            elif os.path.isdir(file_path):
                self.add_folder(file_path)
        
        event.acceptProposedAction()
    
    def get_selected_files(self) -> List[str]:
        """获取选中的文件列表
        
        Returns:
            选中的文件路径列表
        """
        return self.file_list.copy()
    
    def set_batch_progress(self, current: int, total: int):
        """设置批量处理进度
        
        Args:
            current: 当前进度
            total: 总数
        """
        if total > 0:
            self.batch_progress.setVisible(True)
            self.batch_progress.setRange(0, total)
            self.batch_progress.setValue(current)
        else:
            self.batch_progress.setVisible(False)