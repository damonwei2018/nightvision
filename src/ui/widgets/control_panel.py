
# -*- coding: utf-8 -*-
"""
控制面板

负责图像处理参数的调节和控制
"""

from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel,
    QGroupBox, QPushButton, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QColorDialog, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QPalette

from ...utils.config import Config
from ...utils.logger import LoggerMixin

class ControlPanel(QWidget, LoggerMixin):
    """控制面板"""
    
    # 信号定义
    parameter_changed = pyqtSignal(str, object)  # 参数变化信号
    process_requested = pyqtSignal()  # 处理请求信号
    
    def __init__(self, config: Config, parent: Optional[QWidget] = None):
        """初始化控制面板
        
        Args:
            config: 配置对象
            parent: 父窗口
        """
        super().__init__(parent)
        self.config = config
        self.parameter_widgets = {}
        self.updating_ui = False  # 防止UI更新时触发信号
        
        self.init_ui()
        self.load_parameters()
        
        self.logger.info("控制面板初始化完成")
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # 创建主容器
        main_widget = QWidget()
        scroll_area.setWidget(main_widget)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        
        # 内容布局
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 实时预览控制
        preview_group = QGroupBox("预览控制")
        preview_layout = QVBoxLayout(preview_group)
        
        self.realtime_preview_check = QCheckBox("实时预览")
        self.realtime_preview_check.setChecked(True)
        self.realtime_preview_check.toggled.connect(self.on_realtime_preview_toggled)
        preview_layout.addWidget(self.realtime_preview_check)
        
        self.process_btn = QPushButton("处理图像")
        self.process_btn.clicked.connect(self.process_requested.emit)
        preview_layout.addWidget(self.process_btn)
        
        layout.addWidget(preview_group)
        
        # 图像增强参数
        self.create_enhancement_group(layout)
        
        # 边缘检测参数
        self.create_edge_detection_group(layout)
        
        # 描边风格参数
        self.create_edge_style_group(layout)
        
        # 输出设置
        self.create_output_settings_group(layout)
        
        # 预设和重置
        self.create_preset_group(layout)
        
        # 添加弹性空间
        layout.addStretch()
        
        # 设置样式
        self.setStyleSheet(self.get_stylesheet())
    
    def create_enhancement_group(self, parent_layout: QVBoxLayout):
        """创建图像增强参数组"""
        group = QGroupBox("图像增强 (Zero-DCE)")
        layout = QVBoxLayout(group)
        
        # 亮度调节
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("亮度:"))
        
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(50)
        self.brightness_slider.valueChanged.connect(
            lambda v: self.on_parameter_changed('brightness', v / 100.0)
        )
        brightness_layout.addWidget(self.brightness_slider)
        
        self.brightness_label = QLabel("0.50")
        self.brightness_label.setMinimumWidth(40)
        brightness_layout.addWidget(self.brightness_label)
        
        layout.addLayout(brightness_layout)
        
        # 对比度调节
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("对比度:"))
        
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(
            lambda v: self.on_parameter_changed('contrast', v / 100.0)
        )
        contrast_layout.addWidget(self.contrast_slider)
        
        self.contrast_label = QLabel("1.00")
        self.contrast_label.setMinimumWidth(40)
        contrast_layout.addWidget(self.contrast_label)
        
        layout.addLayout(contrast_layout)
        
        # 伽马校正
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("伽马:"))
        
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(50, 200)
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(
            lambda v: self.on_parameter_changed('gamma', v / 100.0)
        )
        gamma_layout.addWidget(self.gamma_slider)
        
        self.gamma_label = QLabel("1.00")
        self.gamma_label.setMinimumWidth(40)
        gamma_layout.addWidget(self.gamma_label)
        
        layout.addLayout(gamma_layout)
        
        # 曝光调节
        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(QLabel("曝光:"))
        
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(-100, 100)
        self.exposure_slider.setValue(0)
        self.exposure_slider.valueChanged.connect(
            lambda v: self.on_parameter_changed('exposure', v / 100.0)
        )
        exposure_layout.addWidget(self.exposure_slider)
        
        self.exposure_label = QLabel("0.00")
        self.exposure_label.setMinimumWidth(40)
        exposure_layout.addWidget(self.exposure_label)
        
        layout.addLayout(exposure_layout)
        
        # 保存控件引用
        self.parameter_widgets.update({
            'brightness': (self.brightness_slider, self.brightness_label),
            'contrast': (self.contrast_slider, self.contrast_label),
            'gamma': (self.gamma_slider, self.gamma_label),
            'exposure': (self.exposure_slider, self.exposure_label)
        })
        
        parent_layout.addWidget(group)
    
    def create_edge_detection_group(self, parent_layout: QVBoxLayout):
        """创建边缘检测参数组"""
        group = QGroupBox("边缘检测 (DexiNed)")
        layout = QVBoxLayout(group)
        
        # 启用边缘检测
        self.enable_edge_check = QCheckBox("启用边缘检测")
        self.enable_edge_check.setChecked(True)
        self.enable_edge_check.toggled.connect(
            lambda checked: self.on_parameter_changed('enable_edge_detection', checked)
        )
        layout.addWidget(self.enable_edge_check)
        
        # 边缘阈值
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("边缘阈值:"))
        
        self.edge_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_threshold_slider.setRange(1, 100)
        self.edge_threshold_slider.setValue(50)
        self.edge_threshold_slider.valueChanged.connect(
            lambda v: self.on_parameter_changed('edge_threshold', v / 100.0)
        )
        threshold_layout.addWidget(self.edge_threshold_slider)
        
        self.edge_threshold_label = QLabel("0.50")
        self.edge_threshold_label.setMinimumWidth(40)
        threshold_layout.addWidget(self.edge_threshold_label)
        
        layout.addLayout(threshold_layout)
        
        # 边缘宽度
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("边缘宽度:"))
        
        self.edge_width_spin = QSpinBox()
        self.edge_width_spin.setRange(1, 10)
        self.edge_width_spin.setValue(2)
        self.edge_width_spin.valueChanged.connect(
            lambda v: self.on_parameter_changed('edge_width', v)
        )
        width_layout.addWidget(self.edge_width_spin)
        
        width_layout.addWidget(QLabel("像素"))
        width_layout.addStretch()
        
        layout.addLayout(width_layout)
        
        # 保存控件引用
        self.parameter_widgets.update({
            'enable_edge_detection': self.enable_edge_check,
            'edge_threshold': (self.edge_threshold_slider, self.edge_threshold_label),
            'edge_width': self.edge_width_spin
        })
        
        parent_layout.addWidget(group)
    
    def create_edge_style_group(self, parent_layout: QVBoxLayout):
        """创建描边风格参数组"""
        group = QGroupBox("描边风格")
        layout = QVBoxLayout(group)
        
        # 风格选择
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("风格:"))
        
        self.edge_style_combo = QComboBox()
        self.edge_style_combo.addItems([
            "热成像绿色",
            "军用夜视黄绿色", 
            "经典夜视白色",
            "自定义颜色"
        ])
        self.edge_style_combo.currentTextChanged.connect(
            lambda text: self.on_edge_style_changed(text)
        )
        style_layout.addWidget(self.edge_style_combo)
        
        layout.addLayout(style_layout)
        
        # 颜色选择
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("颜色:"))
        
        self.edge_color_btn = QPushButton()
        self.edge_color_btn.setFixedSize(50, 30)
        self.edge_color_btn.clicked.connect(self.choose_edge_color)
        self.set_edge_color(QColor(0, 255, 0))  # 默认绿色
        color_layout.addWidget(self.edge_color_btn)
        
        color_layout.addStretch()
        
        layout.addLayout(color_layout)
        
        # 透明度
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("透明度:"))
        
        self.edge_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_alpha_slider.setRange(10, 100)
        self.edge_alpha_slider.setValue(100)
        self.edge_alpha_slider.valueChanged.connect(
            lambda v: self.on_parameter_changed('edge_alpha', v / 100.0)
        )
        alpha_layout.addWidget(self.edge_alpha_slider)
        
        self.edge_alpha_label = QLabel("100%")
        self.edge_alpha_label.setMinimumWidth(40)
        alpha_layout.addWidget(self.edge_alpha_label)
        
        layout.addLayout(alpha_layout)
        
        # 保存控件引用
        self.parameter_widgets.update({
            'edge_style': self.edge_style_combo,
            'edge_color': self.edge_color_btn,
            'edge_alpha': (self.edge_alpha_slider, self.edge_alpha_label)
        })
        
        parent_layout.addWidget(group)
    
    def create_output_settings_group(self, parent_layout: QVBoxLayout):
        """创建输出设置组"""
        group = QGroupBox("输出设置")
        layout = QVBoxLayout(group)
        
        # 输出尺寸
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("输出尺寸:"))
        
        self.output_size_combo = QComboBox()
        self.output_size_combo.addItems([
            "原始尺寸",
            "1920x1080",
            "1280x720",
            "自定义"
        ])
        self.output_size_combo.currentTextChanged.connect(
            lambda text: self.on_parameter_changed('output_size', text)
        )
        size_layout.addWidget(self.output_size_combo)
        
        layout.addLayout(size_layout)
        
        # 自定义尺寸
        custom_size_layout = QHBoxLayout()
        custom_size_layout.addWidget(QLabel("宽度:"))
        
        self.custom_width_spin = QSpinBox()
        self.custom_width_spin.setRange(100, 4096)
        self.custom_width_spin.setValue(1920)
        self.custom_width_spin.valueChanged.connect(
            lambda v: self.on_parameter_changed('custom_width', v)
        )
        custom_size_layout.addWidget(self.custom_width_spin)
        
        custom_size_layout.addWidget(QLabel("高度:"))
        
        self.custom_height_spin = QSpinBox()
        self.custom_height_spin.setRange(100, 4096)
        self.custom_height_spin.setValue(1080)
        self.custom_height_spin.valueChanged.connect(
            lambda v: self.on_parameter_changed('custom_height', v)
        )
        custom_size_layout.addWidget(self.custom_height_spin)
        
        layout.addLayout(custom_size_layout)
        
        # 质量设置
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("输出质量:"))
        
        self.output_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.output_quality_slider.setRange(50, 100)
        self.output_quality_slider.setValue(95)
        self.output_quality_slider.valueChanged.connect(
            lambda v: self.on_parameter_changed('output_quality', v)
        )
        quality_layout.addWidget(self.output_quality_slider)
        
        self.output_quality_label = QLabel("95%")
        self.output_quality_label.setMinimumWidth(40)
        quality_layout.addWidget(self.output_quality_label)
        
        layout.addLayout(quality_layout)
        
        # 保存控件引用
        self.parameter_widgets.update({
            'output_size': self.output_size_combo,
            'custom_width': self.custom_width_spin,
            'custom_height': self.custom_height_spin,
            'output_quality': (self.output_quality_slider, self.output_quality_label)
        })
        
        parent_layout.addWidget(group)
    
    def create_preset_group(self, parent_layout: QVBoxLayout):
        """创建预设和重置组"""
        group = QGroupBox("预设")
        layout = QVBoxLayout(group)
        
        # 预设选择
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设:"))
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "自定义",
            "夜视增强",
            "低光增强",
            "高对比度",
            "柔和增强"
        ])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo)
        
        layout.addLayout(preset_layout)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("重置参数")
        self.reset_btn.clicked.connect(self.reset_parameters)
        button_layout.addWidget(self.reset_btn)
        
        self.save_preset_btn = QPushButton("保存预设")
        self.save_preset_btn.clicked.connect(self.save_current_preset)
        button_layout.addWidget(self.save_preset_btn)
        
        layout.addLayout(button_layout)
        
        parent_layout.addWidget(group)
    
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
        
        QSlider::groove:horizontal {
            border: 1px solid #3c3c3c;
            height: 6px;
            background: #1e1e1e;
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal {
            background: #0078d4;
            border: 1px solid #0078d4;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        
        QSlider::handle:horizontal:hover {
            background: #106ebe;
        }
        
        QPushButton {
            background-color: #0078d4;
            border: none;
            color: white;
            padding: 6px 12px;
            border-radius: 4px;
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
        
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        
        QCheckBox::indicator:unchecked {
            border: 1px solid #3c3c3c;
            background-color: #1e1e1e;
            border-radius: 2px;
        }
        
        QCheckBox::indicator:checked {
            border: 1px solid #0078d4;
            background-color: #0078d4;
            border-radius: 2px;
        }
        """
    
    def on_parameter_changed(self, param_name: str, value):
        """参数变化处理
        
        Args:
            param_name: 参数名称
            value: 参数值
        """
        if self.updating_ui:
            return
        
        # 更新对应的标签显示
        if param_name in self.parameter_widgets:
            widget_info = self.parameter_widgets[param_name]
            if isinstance(widget_info, tuple) and len(widget_info) == 2:
                slider, label = widget_info
                if param_name in ['brightness', 'contrast', 'gamma', 'exposure']:
                    label.setText(f"{value:.2f}")
                elif param_name == 'edge_threshold':
                    label.setText(f"{value:.2f}")
                elif param_name in ['edge_alpha', 'output_quality']:
                    label.setText(f"{int(value * 100)}%" if param_name == 'edge_alpha' else f"{value}%")
        
        # 发送参数变化信号
        self.parameter_changed.emit(param_name, value)
        
        # 更新预设选择为自定义
        if not self.updating_ui:
            self.preset_combo.setCurrentText("自定义")
    
    def on_edge_style_changed(self, style_text: str):
        """边缘风格变化处理
        
        Args:
            style_text: 风格文本
        """
        style_colors = {
            "热成像绿色": QColor(0, 255, 0),
            "军用夜视黄绿色": QColor(173, 255, 47),
            "经典夜视白色": QColor(255, 255, 255),
            "自定义颜色": None
        }
        
        if style_text in style_colors and style_colors[style_text] is not None:
            self.set_edge_color(style_colors[style_text])
        
        self.on_parameter_changed('edge_style', style_text)
    
    def choose_edge_color(self):
        """选择边缘颜色"""
        current_color = self.edge_color_btn.palette().color(QPalette.ColorRole.Button)
        color = QColorDialog.getColor(current_color, self, "选择边缘颜色")
        
        if color.isValid():
            self.set_edge_color(color)
            self.edge_style_combo.setCurrentText("自定义颜色")
            self.on_parameter_changed('edge_color', color.name())
    
    def set_edge_color(self, color: QColor):
        """设置边缘颜色
        
        Args:
            color: 颜色对象
        """
        self.edge_color_btn.setStyleSheet(
            f"QPushButton {{ background-color: {color.name()}; border: 1px solid #3c3c3c; }}"
        )
    
    def on_realtime_preview_toggled(self, checked: bool):
        """实时预览切换处理
        
        Args:
            checked: 是否启用
        """
        self.process_btn.setEnabled(not checked)
        self.on_parameter_changed('realtime_preview', checked)
    
    def is_realtime_preview_enabled(self) -> bool:
        """检查是否启用实时预览
        
        Returns:
            是否启用实时预览
        """
        return self.realtime_preview_check.isChecked()
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取当前所有参数
        
        Returns:
            参数字典
        """
        params = {
            # 图像增强参数
            'brightness': self.brightness_slider.value() / 100.0,
            'contrast': self.contrast_slider.value() / 100.0,
            'gamma': self.gamma_slider.value() / 100.0,
            'exposure': self.exposure_slider.value() / 100.0,
            
            # 边缘检测参数
            'enable_edge_detection': self.enable_edge_check.isChecked(),
            'edge_threshold': self.edge_threshold_slider.value() / 100.0,
            'edge_width': self.edge_width_spin.value(),
            
            # 描边风格参数
            'edge_style': self.edge_style_combo.currentText(),
            'edge_color': self.edge_color_btn.palette().color(QPalette.ColorRole.Button).name(),
            'edge_alpha': self.edge_alpha_slider.value() / 100.0,
            
            # 输出设置
            'output_size': self.output_size_combo.currentText(),
            'custom_width': self.custom_width_spin.value(),
            'custom_height': self.custom_height_spin.value(),
            'output_quality': self.output_quality_slider.value(),
            
            # 其他设置
            'realtime_preview': self.realtime_preview_check.isChecked()
        }
        
        return params
    
    def set_parameters(self, params: Dict[str, Any]):
        """设置参数
        
        Args:
            params: 参数字典
        """
        self.updating_ui = True
        
        try:
            # 图像增强参数
            if 'brightness' in params:
                self.brightness_slider.setValue(int(params['brightness'] * 100))
            if 'contrast' in params:
                self.contrast_slider.setValue(int(params['contrast'] * 100))
            if 'gamma' in params:
                self.gamma_slider.setValue(int(params['gamma'] * 100))
            if 'exposure' in params:
                self.exposure_slider.setValue(int(params['exposure'] * 100))
            
            # 边缘检测参数
            if 'enable_edge_detection' in params:
                self.enable_edge_check.setChecked(params['enable_edge_detection'])
            if 'edge_threshold' in params:
                self.edge_threshold_slider.setValue(int(params['edge_threshold'] * 100))
            if 'edge_width' in params:
                self.edge_width_spin.setValue(params['edge_width'])
            
            # 描边风格参数
            if 'edge_style' in params:
                self.edge_style_combo.setCurrentText(params['edge_style'])
            if 'edge_color' in params:
                color = QColor(params['edge_color'])
                self.set_edge_color(color)
            if 'edge_alpha' in params:
                self.edge_alpha_slider.setValue(int(params['edge_alpha'] * 100))
            
            # 输出设置
            if 'output_size' in params:
                self.output_size_combo.setCurrentText(params['output_size'])
            if 'custom_width' in params:
                self.custom_width_spin.setValue(params['custom_width'])
            if 'custom_height' in params:
                self.custom_height_spin.setValue(params['custom_height'])
            if 'output_quality' in params:
                self.output_quality_slider.setValue(params['output_quality'])
            
            # 其他设置
            if 'realtime_preview' in params:
                self.realtime_preview_check.setChecked(params['realtime_preview'])
        
        finally:
            self.updating_ui = False
    
    def load_parameters(self):
        """从配置加载参数"""
        enhancement_params = {
            'brightness': self.config.get('enhancement.brightness', 0.5),
            'contrast': self.config.get('enhancement.contrast', 1.0),
            'gamma': self.config.get('enhancement.gamma', 1.0),
            'exposure': self.config.get('enhancement.exposure', 0.0)
        }
        
        edge_params = {
            'edge_threshold': self.config.get('edge_detection.threshold', 0.5),
            'edge_width': self.config.get('edge_detection.edge_width', 2),
            'edge_color': self.config.get('edge_detection.edge_color', '#00FF00'),
            'edge_style': self.config.get('edge_detection.edge_style', '热成像绿色')
        }
        
        all_params = {**enhancement_params, **edge_params}
        self.set_parameters(all_params)
    
    def save_parameters(self):
        """保存参数到配置"""
        params = self.get_parameters()
        
        # 保存图像增强参数
        self.config.set('enhancement.brightness', params['brightness'])
        self.config.set('enhancement.contrast', params['contrast'])
        self.config.set('enhancement.gamma', params['gamma'])
        self.config.set('enhancement.exposure', params['exposure'])
        
        # 保存边缘检测参数
        self.config.set('edge_detection.threshold', params['edge_threshold'])
        self.config.set('edge_detection.edge_width', params['edge_width'])
        self.config.set('edge_detection.edge_color', params['edge_color'])
        self.config.set('edge_detection.edge_style', params['edge_style'])
        
        self.config.save_config()
    
    def reset_parameters(self):
        """重置参数为默认值"""
        default_params = {
            'brightness': 0.5,
            'contrast': 1.0,
            'gamma': 1.0,
            'exposure': 0.0,
            'enable_edge_detection': True,
            'edge_threshold': 0.5,
            'edge_width': 2,
            'edge_style': '热成像绿色',
            'edge_color': '#00FF00',
            'edge_alpha': 1.0,
            'output_size': '原始尺寸',
            'output_quality': 95,
            'realtime_preview': True
        }
        
        self.set_parameters(default_params)
        self.preset_combo.setCurrentText("自定义")
    
    def apply_preset(self, preset_name: str):
        """应用预设
        
        Args:
            preset_name: 预设名称
        """
        if preset_name == "自定义":
            return
        
        presets = {
            "夜视增强": {
                'brightness': 0.7,
                'contrast': 1.2,
                'gamma': 0.8,
                'exposure': 0.1,
                'edge_style': '热成像绿色',
                'edge_threshold': 0.4
            },
            "低光增强": {
                'brightness': 0.8,
                'contrast': 1.1,
                'gamma': 0.9,
                'exposure': 0.2,
                'edge_style': '军用夜视黄绿色',
                'edge_threshold': 0.3
            },
            "高对比度": {
                'brightness': 0.6,
                'contrast': 1.5,
                'gamma': 1.0,
                'exposure': 0.0,
                'edge_style': '经典夜视白色',
                'edge_threshold': 0.6
            },
            "柔和增强": {
                'brightness': 0.6,
                'contrast': 1.1,
                'gamma': 1.1,
                'exposure': 0.05,
                'edge_style': '热成像绿色',
                'edge_threshold': 0.2
            }
        }
        
        if preset_name in presets:
            self.set_parameters(presets[preset_name])
    
    def save_current_preset(self):
        """保存当前设置为预设"""
        # TODO: 实现自定义预设保存功能
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "提示", "自定义预设保存功能正在开发中")