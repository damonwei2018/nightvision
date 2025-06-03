# -*- coding: utf-8 -*-
"""
DexiNed边缘检测模型实现

直接使用官方DexiNed实现，确保与预训练权重完全兼容
"""

import os
from typing import Union, Optional

import numpy as np
import torch
from PIL import Image

from .model_manager import BaseModel
from ..utils.config import Config

class DexiNedModel(BaseModel):
    """DexiNed边缘检测模型 - 使用官方实现"""
    
    def __init__(self, config: Config):
        """初始化DexiNed模型
        
        Args:
            config: 配置对象
        """
        super().__init__(config, "DexiNed")
        
        # 模型参数
        self.input_size = (352, 352)  # 官方模型使用352x352
        self.use_onnx = False  # 官方模型只支持PyTorch
        
        # 边缘检测参数
        self.threshold = 0.5
        self.edge_width = 1
        self.use_nms = True
        self.output_index = -1  # 使用融合输出（最后一个）
    
    def load_model(self, model_path: str) -> bool:
        """加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        # 始终使用官方模型实现
        self.logger.info("加载官方DexiNed模型")
        try:
            from .dexined_official import DexiNedOfficialModel
            
            # 创建官方模型实例
            self._official_model = DexiNedOfficialModel(self.config)
            
            # 加载模型
            success = self._official_model.load_model(model_path)
            
            if success:
                self.is_loaded = True
                self.use_onnx = False
                
                # 代理官方模型的所有方法
                self.preprocess = self._official_model.preprocess
                self.inference = self._official_model.inference
                self.postprocess = self._official_model.postprocess
                self.predict = self._official_model.predict
                self.set_edge_params = self._official_model.set_edge_params
                self.get_all_outputs = self._official_model.get_all_outputs
                
                self.logger.info("官方DexiNed模型加载成功")
            
            return success
            
        except Exception as e:
            self.logger.error(f"加载官方DexiNed模型失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def get_edge_params(self) -> dict:
        """获取当前边缘检测参数
        
        Returns:
            边缘检测参数字典
        """
        if hasattr(self, '_official_model'):
            return {
                'threshold': self._official_model.threshold,
                'edge_width': self._official_model.edge_width,
                'use_nms': self._official_model.use_nms,
                'output_index': self._official_model.output_index
            }
        else:
            return {
                'threshold': self.threshold,
                'edge_width': self.edge_width,
                'use_nms': self.use_nms,
                'output_index': self.output_index
            }
    
    def reset_edge_params(self):
        """重置边缘检测参数为默认值"""
        if hasattr(self, '_official_model'):
            self._official_model.set_edge_params(
                threshold=0.5,
                edge_width=1,
                use_nms=True,
                output_index=-1
            )
        else:
            self.threshold = 0.5
            self.edge_width = 1
            self.use_nms = True
            self.output_index = -1
        
        self.logger.debug("边缘检测参数已重置为默认值")
    
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """预处理输入图像（默认实现，将被官方模型方法替换）"""
        if hasattr(self, '_official_model'):
            return self._official_model.preprocess(image)
        else:
            raise NotImplementedError("模型未加载")
    
    def inference(self, input_data) -> Union[np.ndarray, torch.Tensor]:
        """模型推理（默认实现，将被官方模型方法替换）"""
        if hasattr(self, '_official_model'):
            return self._official_model.inference(input_data)
        else:
            raise NotImplementedError("模型未加载")
    
    def postprocess(self, output: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """后处理输出结果（默认实现，将被官方模型方法替换）"""
        if hasattr(self, '_official_model'):
            return self._official_model.postprocess(output)
        else:
            raise NotImplementedError("模型未加载")