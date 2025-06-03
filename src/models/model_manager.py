# -*- coding: utf-8 -*-
"""
模型管理器

负责深度学习模型的加载、管理和推理
"""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

import numpy as np
import torch
import onnxruntime as ort
from PIL import Image

from ..utils.config import Config
from ..utils.logger import LoggerMixin

class BaseModel(ABC, LoggerMixin):
    """模型基类"""
    
    def __init__(self, config: Config, model_name: str):
        """初始化模型
        
        Args:
            config: 配置对象
            model_name: 模型名称
        """
        self.config = config
        self.model_name = model_name
        self.model = None
        self.device = None
        self.input_size = None
        self.is_loaded = False
        
        # 性能统计
        self.inference_times = []
        self.total_inferences = 0
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> Any:
        """预处理输入图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的数据
        """
        pass
    
    @abstractmethod
    def inference(self, input_data: Any) -> Any:
        """模型推理
        
        Args:
            input_data: 预处理后的输入数据
            
        Returns:
            推理结果
        """
        pass
    
    @abstractmethod
    def postprocess(self, output: Any) -> Union[Image.Image, np.ndarray]:
        """后处理输出结果
        
        Args:
            output: 模型输出
            
        Returns:
            后处理后的图像
        """
        pass
    
    def predict(self, image: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        """完整的预测流程
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        if not self.is_loaded:
            raise RuntimeError(f"模型 {self.model_name} 未加载")
        
        start_time = time.time()
        
        try:
            # 预处理
            input_data = self.preprocess(image)
            
            # 推理
            output = self.inference(input_data)
            
            # 后处理
            result = self.postprocess(output)
            
            # 记录性能
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_inferences += 1
            
            # 保持最近100次推理时间
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            self.logger.debug(f"{self.model_name} 推理完成，耗时: {inference_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"{self.model_name} 推理失败: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计
        
        Returns:
            性能统计数据
        """
        if not self.inference_times:
            return {
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'total_inferences': self.total_inferences
            }
        
        return {
            'avg_time': np.mean(self.inference_times),
            'min_time': np.min(self.inference_times),
            'max_time': np.max(self.inference_times),
            'total_inferences': self.total_inferences
        }
    
    def unload_model(self):
        """卸载模型"""
        self.model = None
        self.is_loaded = False
        self.logger.info(f"模型 {self.model_name} 已卸载")

class ModelManager(LoggerMixin):
    """模型管理器"""
    
    def __init__(self, config: Config):
        """初始化模型管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.models = {}
        self.device_info = self._get_device_info()
        
        self.logger.info(f"模型管理器初始化完成，设备信息: {self.device_info}")
    
    def _get_device_info(self) -> Dict[str, Any]:
        """获取设备信息
        
        Returns:
            设备信息字典
        """
        device_info = {
            'cpu': True,
            'cuda': torch.cuda.is_available(),
            'cuda_devices': [],
            'onnx_providers': ort.get_available_providers()
        }
        
        if device_info['cuda']:
            device_info['cuda_devices'] = [
                {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory': torch.cuda.get_device_properties(i).total_memory
                }
                for i in range(torch.cuda.device_count())
            ]
        
        return device_info
    
    def register_model(self, model_name: str, model_instance: BaseModel):
        """注册模型
        
        Args:
            model_name: 模型名称
            model_instance: 模型实例
        """
        self.models[model_name] = model_instance
        self.logger.info(f"模型 {model_name} 已注册")
    
    def load_model(self, model_name: str, model_path: str) -> bool:
        """加载模型
        
        Args:
            model_name: 模型名称
            model_path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        if model_name not in self.models:
            self.logger.error(f"未注册的模型: {model_name}")
            return False
        
        if not os.path.exists(model_path):
            self.logger.error(f"模型文件不存在: {model_path}")
            return False
        
        try:
            success = self.models[model_name].load_model(model_path)
            if success:
                self.logger.info(f"模型 {model_name} 加载成功")
            else:
                self.logger.error(f"模型 {model_name} 加载失败")
            return success
            
        except Exception as e:
            self.logger.error(f"模型 {model_name} 加载异常: {e}")
            return False
    
    def unload_model(self, model_name: str):
        """卸载模型
        
        Args:
            model_name: 模型名称
        """
        if model_name in self.models:
            self.models[model_name].unload_model()
            self.logger.info(f"模型 {model_name} 已卸载")
    
    def unload_all_models(self):
        """卸载所有模型"""
        for model_name in self.models:
            self.unload_model(model_name)
        self.logger.info("所有模型已卸载")
    
    def predict(self, model_name: str, image: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        """使用指定模型进行预测
        
        Args:
            model_name: 模型名称
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        if model_name not in self.models:
            raise ValueError(f"未注册的模型: {model_name}")
        
        return self.models[model_name].predict(image)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否已加载
        """
        if model_name not in self.models:
            return False
        return self.models[model_name].is_loaded
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息字典
        """
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        return {
            'name': model_name,
            'is_loaded': model.is_loaded,
            'device': str(model.device) if model.device else None,
            'input_size': model.input_size,
            'performance': model.get_performance_stats()
        }
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型信息
        
        Returns:
            所有模型信息字典
        """
        return {
            model_name: self.get_model_info(model_name)
            for model_name in self.models
        }
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息
        
        Returns:
            设备信息字典
        """
        return self.device_info.copy()
    
    def get_recommended_device(self) -> str:
        """获取推荐的设备
        
        Returns:
            推荐的设备名称
        """
        if self.device_info['cuda'] and self.device_info['cuda_devices']:
            return 'cuda'
        else:
            return 'cpu'
    
    def cleanup(self):
        """清理资源"""
        self.unload_all_models()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("模型管理器资源清理完成")

def get_onnx_providers(use_gpu: bool = True) -> list:
    """获取ONNX运行时提供者
    
    Args:
        use_gpu: 是否使用GPU
        
    Returns:
        提供者列表
    """
    providers = []
    
    if use_gpu:
        # CUDA提供者
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        
        # DirectML提供者（Windows）
        if 'DmlExecutionProvider' in ort.get_available_providers():
            providers.append('DmlExecutionProvider')
    
    # CPU提供者（总是可用）
    providers.append('CPUExecutionProvider')
    
    return providers

def optimize_onnx_session(model_path: str, use_gpu: bool = True) -> ort.InferenceSession:
    """创建优化的ONNX推理会话
    
    Args:
        model_path: 模型文件路径
        use_gpu: 是否使用GPU
        
    Returns:
        ONNX推理会话
    """
    # 会话选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 设置线程数
    sess_options.intra_op_num_threads = 0  # 使用所有可用线程
    sess_options.inter_op_num_threads = 0
    
    # 获取提供者
    providers = get_onnx_providers(use_gpu)
    
    # 创建会话
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers
    )
    
    return session