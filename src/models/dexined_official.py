# -*- coding: utf-8 -*-
"""
官方DexiNed边缘检测模型包装器

使用官方实现的DexiNed模型
"""

import os
import sys
from typing import Union, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

# 添加官方DexiNed路径
project_root = Path(__file__).parent.parent.parent
third_party_path = project_root / 'third_party' / 'DexiNed'
if third_party_path.exists():
    sys.path.insert(0, str(third_party_path))

try:
    from model import DexiNed as OfficialDexiNed
except ImportError:
    print("警告: 无法导入官方DexiNed模型，请确保third_party/DexiNed目录存在")
    OfficialDexiNed = None

from .model_manager import BaseModel
from ..utils.config import Config

class DexiNedOfficialModel(BaseModel):
    """官方DexiNed边缘检测模型"""
    
    def __init__(self, config: Config):
        """初始化官方DexiNed模型
        
        Args:
            config: 配置对象
        """
        super().__init__(config, "DexiNed-Official")
        
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
        if OfficialDexiNed is None:
            self.logger.error("无法导入官方DexiNed模型")
            return False
            
        try:
            # 设置设备
            if self.config.get('models.use_gpu', True) and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
            
            # 创建官方模型
            self.logger.info("创建官方DexiNed模型...")
            self.model = OfficialDexiNed().to(self.device)
            
            # 加载权重
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                self.logger.info(f"加载PyTorch权重文件: {model_path}")
                
                # 加载state_dict
                state_dict = torch.load(model_path, map_location=self.device)
                
                # 如果是字典格式，提取state_dict
                if isinstance(state_dict, dict):
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']
                
                # 加载权重
                self.model.load_state_dict(state_dict)
                self.logger.info("官方DexiNed模型权重加载成功")
            else:
                raise ValueError(f"不支持的模型格式: {model_path}")
            
            # 设置为评估模式
            self.model.eval()
            
            self.is_loaded = True
            
            # 统计参数
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"模型参数总量: {total_params:,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"官方DexiNed模型加载失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """预处理输入图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的数据
        """
        # 转换为numpy数组
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image.copy()
        
        # 确保是RGB格式
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # RGB图像
            pass
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # RGBA图像，去除alpha通道
            image_array = image_array[:, :, :3]
        elif len(image_array.shape) == 2:
            # 灰度图像，转换为RGB
            image_array = np.stack([image_array] * 3, axis=2)
        else:
            raise ValueError(f"不支持的图像格式: {image_array.shape}")
        
        # 保存原始尺寸
        self.original_size = (image_array.shape[1], image_array.shape[0])  # (W, H)
        
        # 调整尺寸到352x352（官方模型的输入尺寸）
        if image_array.shape[:2] != (self.input_size[1], self.input_size[0]):
            image_array = cv2.resize(
                image_array, 
                self.input_size, 
                interpolation=cv2.INTER_LINEAR
            )
        
        # 归一化到[0, 1]
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0
        
        # 官方模型使用的均值（BGR格式）
        mean_bgr = [103.939, 116.779, 123.68]
        
        # 转换为BGR并减去均值
        image_bgr = image_array[:, :, ::-1]  # RGB -> BGR
        image_bgr = image_bgr * 255.0  # [0,1] -> [0,255]
        image_bgr = image_bgr - np.array(mean_bgr)
        
        # 转换为模型输入格式 [1, 3, H, W]
        input_data = torch.from_numpy(image_bgr).permute(2, 0, 1).unsqueeze(0).float()
        input_data = input_data.to(self.device)
        
        return input_data
    
    def inference(self, input_data) -> Union[np.ndarray, torch.Tensor]:
        """模型推理
        
        Args:
            input_data: 预处理后的输入数据
            
        Returns:
            推理结果
        """
        with torch.no_grad():
            outputs = self.model(input_data)
        
        # 返回指定的输出
        return outputs[self.output_index]
    
    def postprocess(self, output: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """后处理输出结果
        
        Args:
            output: 模型输出
            
        Returns:
            后处理后的边缘图像
        """
        # 转换为numpy数组
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        
        # 移除batch维度
        if len(output.shape) == 4:
            output = output[0, 0]  # [B, 1, H, W] -> [H, W]
        elif len(output.shape) == 3:
            output = output[0]  # [1, H, W] -> [H, W]
        
        # 应用sigmoid激活（官方模型输出未激活）
        output = 1.0 / (1.0 + np.exp(-output))
        
        # 确保值在[0, 1]范围内
        output = np.clip(output, 0.0, 1.0)
        
        # 应用阈值
        if self.threshold > 0:
            output = (output > self.threshold).astype(np.float32)
        
        # 非极大值抑制
        if self.use_nms:
            output = self._apply_nms(output)
        
        # 边缘宽度调整
        if self.edge_width > 1:
            output = self._adjust_edge_width(output)
        
        # 调整回原始尺寸
        if hasattr(self, 'original_size') and output.shape != (self.original_size[1], self.original_size[0]):
            output = cv2.resize(
                output, 
                self.original_size, 
                interpolation=cv2.INTER_LINEAR
            )
        
        # 转换为uint8格式
        output = (output * 255).astype(np.uint8)
        
        return output
    
    def _apply_nms(self, edge_map: np.ndarray) -> np.ndarray:
        """应用非极大值抑制
        
        Args:
            edge_map: 边缘图
            
        Returns:
            处理后的边缘图
        """
        # 简化版NMS
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype=np.float32)
        response = cv2.filter2D(edge_map, -1, kernel)
        
        # 保留局部最大值
        suppressed = np.zeros_like(edge_map)
        suppressed[response > 0] = edge_map[response > 0]
        
        return suppressed
    
    def _adjust_edge_width(self, edge_map: np.ndarray) -> np.ndarray:
        """调整边缘宽度
        
        Args:
            edge_map: 边缘图
            
        Returns:
            调整后的边缘图
        """
        if self.edge_width <= 1:
            return edge_map
        
        # 使用形态学操作增加边缘宽度
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.edge_width, self.edge_width)
        )
        
        dilated = cv2.dilate(edge_map, kernel, iterations=1)
        
        return dilated
    
    def set_edge_params(self, 
                       threshold: float = 0.5,
                       edge_width: int = 1,
                       use_nms: bool = True,
                       output_index: int = -1):
        """设置边缘检测参数
        
        Args:
            threshold: 边缘阈值 (0.0 - 1.0)
            edge_width: 边缘宽度 (1 - 10)
            use_nms: 是否使用非极大值抑制
            output_index: 使用哪个输出 (-1为融合输出，0-5为侧输出)
        """
        self.threshold = max(0.0, min(1.0, threshold))
        self.edge_width = max(1, min(10, edge_width))
        self.use_nms = use_nms
        self.output_index = max(-1, min(5, output_index))
        
        self.logger.debug(
            f"边缘检测参数更新: 阈值={self.threshold:.2f}, "
            f"宽度={self.edge_width}, NMS={self.use_nms}, "
            f"输出索引={self.output_index}"
        )
    
    def get_all_outputs(self, image: Union[Image.Image, np.ndarray]) -> list:
        """获取所有输出
        
        Args:
            image: 输入图像
            
        Returns:
            所有输出的列表
        """
        if not self.is_loaded:
            raise RuntimeError(f"模型 {self.model_name} 未加载")
        
        # 预处理
        input_data = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_data)
        
        # 后处理所有输出
        processed_outputs = []
        for output in outputs:
            processed = self.postprocess(output)
            processed_outputs.append(processed)
        
        return processed_outputs 