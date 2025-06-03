# -*- coding: utf-8 -*-
"""
配置管理模块

负责应用程序的配置加载、保存和管理
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """配置管理类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认路径
        """
        self.project_root = Path(__file__).parent.parent.parent
        
        if config_file is None:
            # 修改默认配置文件路径为项目根目录的config.json
            self.config_file = self.project_root / "config.json"
        else:
            self.config_file = Path(config_file)
        
        # 添加线程锁保证配置读写安全
        import threading
        self._lock = threading.RLock()
            
        # 默认配置
        self.default_config = {
            "ui": {
                "theme": "dark",
                "window_size": [1200, 800],
                "window_position": [100, 100],
                "splitter_sizes": [300, 600, 300],
                "preview_update_delay": 500  # 毫秒
            },
            "processing": {
                "use_gpu": True,
                "batch_size": 1,
                "num_workers": 4,
                "output_format": "png",
                "output_quality": 95
            },
            "models": {
                "use_gpu": True,
                "zero_dce_model_path": "models/zero_dce.onnx",
                "dexined_model_path": "models/dexined.onnx",
                "model_cache_dir": "models/cache",
                "zero_dce": {
                    "use_original_model": True,  # True: 使用原始enhance_net_nopool结构, False: 使用简化版结构
                    "num_iterations": 8,
                    "input_size": [512, 512]
                },
                "dexined": {
                     "input_size": [512, 512],
                     "threshold": 0.5
                 }
             },
             "enhancement": {
                "brightness": 0.5,
                "contrast": 1.0,
                "gamma": 1.0,
                "exposure": 0.0
            },
            "edge_detection": {
                "threshold": 0.5,
                "edge_width": 2,
                "edge_color": "#00FF00",  # 热成像绿色
                "edge_style": "thermal_green"
            },
            "paths": {
                "last_input_dir": "",
                "last_output_dir": "",
                "temp_dir": "temp"
            }
        }
        
        self.config = self.default_config.copy()
        self.load_config()
    
    def load_config(self) -> None:
        """从文件加载配置，使用线程锁保证线程安全"""
        with self._lock:
            try:
                if self.config_file.exists():
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                        # 递归更新配置，保留默认值
                        self._update_config_recursive(self.config, loaded_config)
                else:
                    # 如果配置文件不存在，创建默认配置文件
                    self.save_config()
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
    
    def save_config(self) -> None:
        """保存配置到文件，使用线程锁保证线程安全，实现原子性写入"""
        with self._lock:
            try:
                # 确保配置目录存在
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 创建临时文件进行原子性写入
                import tempfile
                import shutil
                import os
                
                # 使用临时文件写入配置
                fd, temp_path = tempfile.mkstemp(dir=self.config_file.parent)
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        json.dump(self.config, f, indent=4, ensure_ascii=False)
                    
                    # 在Windows上，可能需要先删除目标文件
                    if os.name == 'nt' and self.config_file.exists():
                        self.config_file.unlink()
                    
                    # 原子性地替换文件
                    shutil.move(temp_path, self.config_file)
                except Exception as e:
                    # 确保清理临时文件
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise e
            except Exception as e:
                print(f"保存配置文件失败: {e}")
    
    def _update_config_recursive(self, base_dict: Dict, update_dict: Dict) -> None:
        """递归更新配置字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_config_recursive(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值，使用线程锁保证线程安全
        
        Args:
            key_path: 配置键路径，使用点号分隔，如 'ui.theme'
            default: 默认值
            
        Returns:
            配置值
        """
        with self._lock:
            keys = key_path.split('.')
            value = self.config
            
            try:
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key_path: str, value: Any) -> None:
        """设置配置值，使用线程锁保证线程安全
        
        Args:
            key_path: 配置键路径，使用点号分隔
            value: 要设置的值
        """
        with self._lock:
            keys = key_path.split('.')
            config_dict = self.config
            
            # 导航到最后一级的父字典
            for key in keys[:-1]:
                if key not in config_dict:
                    config_dict[key] = {}
                config_dict = config_dict[key]
            
            # 设置最终值
            config_dict[keys[-1]] = value
    
    def get_model_path(self, model_name: str) -> Path:
        """获取模型文件路径，使用线程锁保证线程安全
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型文件的绝对路径
        """
        with self._lock:
            model_path = self.get(f'models.{model_name}_model_path')
            if model_path:
                path = Path(model_path)
                if not path.is_absolute():
                    path = self.project_root / path
                return path
            return None
    
    def get_temp_dir(self) -> Path:
        """获取临时目录路径，使用线程锁保证线程安全"""
        with self._lock:
            temp_dir = self.get('paths.temp_dir', 'temp')
            path = Path(temp_dir)
            if not path.is_absolute():
                path = self.project_root / path
            path.mkdir(parents=True, exist_ok=True)
            return path
    
    def reset_to_default(self) -> None:
        """重置为默认配置，使用线程锁保证线程安全"""
        with self._lock:
            self.config = self.default_config.copy()
            self.save_config()