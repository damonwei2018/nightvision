# -*- coding: utf-8 -*-
"""
Zero-DCE模型实现

用于低光照图像增强的深度学习模型
"""

import os
from typing import Union, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

from .model_manager import BaseModel, optimize_onnx_session
from ..utils.config import Config

class DCENet(nn.Module):
    """Zero-DCE网络结构 - 原始enhance_net_nopool实现"""
    
    def __init__(self, num_iterations: int = 8):
        """初始化网络
        
        Args:
            num_iterations: 迭代次数
        """
        super(DCENet, self).__init__()
        self.num_iterations = num_iterations
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
        # 特征提取网络 - 参考原始enhance_net_nopool结构
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)  # 8次迭代 * 3通道 = 24
        
        # 池化和上采样（虽然名字叫nopool，但保留这些层以备用）
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self, x):
        """前向传播 - 参考原始实现
        
        Args:
            x: 输入图像 [B, 3, H, W]
            
        Returns:
            enhance_image_1: 中间增强结果
            enhance_image: 最终增强结果
            r: 增强曲线参数
        """
        # 特征提取 - 参考原始enhance_net_nopool
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        
        # 特征融合
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        
        # 生成增强曲线参数
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        
        # 应用增强曲线 - 完全按照原始实现
        enhanced = x
        enhanced = enhanced + r1 * (torch.pow(enhanced, 2) - enhanced)
        enhanced = enhanced + r2 * (torch.pow(enhanced, 2) - enhanced)
        enhanced = enhanced + r3 * (torch.pow(enhanced, 2) - enhanced)
        enhance_image_1 = enhanced + r4 * (torch.pow(enhanced, 2) - enhanced)
        enhanced = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        enhanced = enhanced + r6 * (torch.pow(enhanced, 2) - enhanced)
        enhanced = enhanced + r7 * (torch.pow(enhanced, 2) - enhanced)
        enhance_image = enhanced + r8 * (torch.pow(enhanced, 2) - enhanced)
        
        # 合并所有曲线参数
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        
        return enhance_image_1, enhance_image, r

class SimpleDCENet(nn.Module):
    """简化版Zero-DCE网络结构 - 原来的实现"""
    
    def __init__(self, num_iterations: int = 8):
        """初始化网络
        
        Args:
            num_iterations: 迭代次数
        """
        super(SimpleDCENet, self).__init__()
        self.num_iterations = num_iterations
        
        # 特征提取网络
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        
        # 输出层 - 生成增强曲线参数
        self.conv5 = nn.Conv2d(32, 3 * num_iterations, 3, padding=1)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            
        Returns:
            增强后的图像
        """
        # 特征提取
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        
        # 生成增强曲线参数
        curves = self.tanh(self.conv5(x4))
        
        # 应用增强曲线
        enhanced = x
        for i in range(self.num_iterations):
            # 获取当前迭代的曲线参数
            curve = curves[:, i*3:(i+1)*3, :, :]
            
            # 应用曲线增强
            enhanced = enhanced + curve * (torch.pow(enhanced, 2) - enhanced)
        
        return enhanced

class ZeroDCEModel(BaseModel):
    """Zero-DCE模型"""
    
    def __init__(self, config: Config):
        """初始化Zero-DCE模型
        
        Args:
            config: 配置对象
        """
        super().__init__(config, "Zero-DCE")
        
        # 模型参数
        self.input_size = (512, 512)  # 默认输入尺寸
        self.num_iterations = 8
        self.use_onnx = True  # 优先使用ONNX
        
        # 增强参数
        self.brightness_factor = 1.0
        self.contrast_factor = 1.0
        self.gamma_factor = 1.0
        self.exposure_factor = 1.0
    
    def load_model(self, model_path: str) -> bool:
        """加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        try:
            if model_path.endswith('.onnx'):
                return self._load_onnx_model(model_path)
            else:
                return self._load_pytorch_model(model_path)
                
        except Exception as e:
            self.logger.error(f"Zero-DCE模型加载失败: {e}")
            return False
    
    def _load_onnx_model(self, model_path: str) -> bool:
        """加载ONNX模型
        
        Args:
            model_path: ONNX模型路径
            
        Returns:
            是否加载成功
        """
        try:
            # 检查GPU可用性
            use_gpu = self.config.get('models.use_gpu', True) and torch.cuda.is_available()
            
            # 创建ONNX会话
            self.model = optimize_onnx_session(model_path, use_gpu)
            
            # 获取输入输出信息
            input_info = self.model.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape
            
            output_info = self.model.get_outputs()[0]
            self.output_name = output_info.name
            
            # 设置输入尺寸
            if len(self.input_shape) == 4 and self.input_shape[2] > 0 and self.input_shape[3] > 0:
                self.input_size = (self.input_shape[3], self.input_shape[2])  # (W, H)
            
            self.device = "ONNX"
            self.use_onnx = True
            self.is_loaded = True
            
            self.logger.info(f"ONNX模型加载成功，输入尺寸: {self.input_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX模型加载失败: {e}")
            return False
    
    def _load_pytorch_model(self, model_path: str) -> bool:
        """加载PyTorch模型
        
        Args:
            model_path: PyTorch模型路径
            
        Returns:
            是否加载成功
        """
        try:
            # 设置设备
            if self.config.get('models.use_gpu', True) and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
            
            # 创建模型 - 根据配置选择模型类型
            use_original_model = self.config.get('models.zero_dce.use_original_model', True)
            
            if use_original_model:
                self.logger.info("使用原始Zero-DCE模型结构")
                self.model = DCENet(self.num_iterations)
                self.use_original_model = True
            else:
                self.logger.info("使用简化版Zero-DCE模型结构")
                self.model = SimpleDCENet(self.num_iterations)
                self.use_original_model = False
            
            # 加载权重
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                self.logger.info(f"加载PyTorch权重文件: {model_path}")
                # 首先尝试直接加载
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                    # 如果是完整的checkpoint
                    if isinstance(state_dict, dict):
                        if 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                        elif 'model' in state_dict:
                            state_dict = state_dict['model']
                    
                    # 处理键名前缀问题
                    new_state_dict = {}
                    model_state_dict = self.model.state_dict()
                    
                    for k, v in state_dict.items():
                        # 移除可能的module前缀
                        if k.startswith('module.'):
                            k = k[7:]
                        
                        # 检查键是否存在于模型中
                        if k in model_state_dict:
                            # 检查形状是否匹配
                            if v.shape == model_state_dict[k].shape:
                                new_state_dict[k] = v
                            else:
                                self.logger.warning(f"形状不匹配，跳过键 {k}: 期望 {model_state_dict[k].shape}, 实际 {v.shape}")
                        else:
                            # 尝试匹配相似的键名
                            matched = False
                            for model_key in model_state_dict.keys():
                                if model_key.endswith(k.split('.')[-1]) and v.shape == model_state_dict[model_key].shape:
                                    new_state_dict[model_key] = v
                                    matched = True
                                    self.logger.debug(f"匹配键 {k} -> {model_key}")
                                    break
                            if not matched:
                                self.logger.debug(f"未找到匹配的键: {k}")
                    
                    # 加载权重
                    missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
                    
                    if missing_keys:
                        self.logger.warning(f"缺失的键: {missing_keys}")
                    if unexpected_keys:
                        self.logger.warning(f"意外的键: {unexpected_keys}")
                    
                    # 检查是否成功加载了足够的权重
                    loaded_ratio = len(new_state_dict) / len(model_state_dict)
                    if loaded_ratio < 0.5:
                        self.logger.warning(f"只加载了 {loaded_ratio*100:.1f}% 的权重，可能模型结构不匹配")
                    else:
                        self.logger.info(f"成功加载了 {loaded_ratio*100:.1f}% 的权重")
                    
                except Exception as e:
                    self.logger.error(f"加载权重时出错: {e}")
                    # 继续使用随机初始化的权重
                    self.logger.warning("使用随机初始化的权重继续")
            else:
                raise ValueError(f"不支持的模型格式: {model_path}")
            
            # 移动到设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()
            
            self.use_onnx = False
            self.is_loaded = True
            
            self.logger.info(f"PyTorch模型加载完成，设备: {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"PyTorch模型加载失败: {e}")
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
        
        # 调整尺寸
        if image_array.shape[:2] != (self.input_size[1], self.input_size[0]):
            image_array = cv2.resize(
                image_array, 
                self.input_size, 
                interpolation=cv2.INTER_LINEAR
            )
        
        # 归一化到[0, 1]
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0
        
        # 转换为模型输入格式 [1, 3, H, W]
        if self.use_onnx:
            input_data = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
            input_data = np.expand_dims(input_data, axis=0)  # 添加batch维度
        else:
            input_data = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
            input_data = input_data.to(self.device)
        
        return input_data
    
    def inference(self, input_data) -> Union[np.ndarray, torch.Tensor]:
        """模型推理
        
        Args:
            input_data: 预处理后的输入数据
            
        Returns:
            推理结果
        """
        if self.use_onnx:
            # ONNX推理
            output = self.model.run([self.output_name], {self.input_name: input_data})[0]
        else:
            # PyTorch推理
            with torch.no_grad():
                if hasattr(self, 'use_original_model') and self.use_original_model:
                    # 原始DCENet返回三个值：中间结果、最终结果和增强曲线参数
                    _, output, _ = self.model(input_data)
                else:
                    # 简化版DCENet只返回最终结果
                    output = self.model(input_data)
        
        return output
    
    def postprocess(self, output: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """后处理输出结果
        
        Args:
            output: 模型输出
            
        Returns:
            后处理后的图像
        """
        # 转换为numpy数组
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        
        # 移除batch维度并转换为HWC格式
        if len(output.shape) == 4:
            output = output[0]  # 移除batch维度
        
        if output.shape[0] == 3:  # CHW格式
            output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
        
        # 确保值在[0, 1]范围内
        output = np.clip(output, 0.0, 1.0)
        
        # 应用额外的增强参数
        output = self._apply_enhancement_params(output)
        
        # 调整回原始尺寸
        if hasattr(self, 'original_size') and output.shape[:2] != (self.original_size[1], self.original_size[0]):
            output = cv2.resize(
                output, 
                self.original_size, 
                interpolation=cv2.INTER_LINEAR
            )
        
        # 转换为uint8格式
        output = (output * 255).astype(np.uint8)
        
        return output
    
    def _apply_enhancement_params(self, image: np.ndarray) -> np.ndarray:
        """应用增强参数
        
        Args:
            image: 输入图像 [0, 1]
            
        Returns:
            增强后的图像
        """
        enhanced = image.copy()
        
        # 亮度调整
        if self.brightness_factor != 1.0:
            enhanced = enhanced * self.brightness_factor
        
        # 对比度调整
        if self.contrast_factor != 1.0:
            mean = np.mean(enhanced)
            enhanced = (enhanced - mean) * self.contrast_factor + mean
        
        # Gamma校正
        if self.gamma_factor != 1.0:
            enhanced = np.power(enhanced, 1.0 / self.gamma_factor)
        
        # 曝光调整
        if self.exposure_factor != 1.0:
            enhanced = enhanced * (2.0 ** self.exposure_factor)
        
        # 确保值在[0, 1]范围内
        enhanced = np.clip(enhanced, 0.0, 1.0)
        
        return enhanced
    
    def set_enhancement_params(self, 
                             brightness: float = 1.0,
                             contrast: float = 1.0,
                             gamma: float = 1.0,
                             exposure: float = 1.0):
        """设置增强参数
        
        Args:
            brightness: 亮度因子 (0.1 - 3.0)
            contrast: 对比度因子 (0.1 - 3.0)
            gamma: Gamma因子 (0.1 - 3.0)
            exposure: 曝光因子 (-2.0 - 2.0)
        """
        self.brightness_factor = max(0.1, min(3.0, brightness))
        self.contrast_factor = max(0.1, min(3.0, contrast))
        self.gamma_factor = max(0.1, min(3.0, gamma))
        self.exposure_factor = max(-2.0, min(2.0, exposure))
        
        self.logger.debug(
            f"增强参数更新: 亮度={self.brightness_factor:.2f}, "
            f"对比度={self.contrast_factor:.2f}, "
            f"Gamma={self.gamma_factor:.2f}, "
            f"曝光={self.exposure_factor:.2f}"
        )
    
    def get_enhancement_params(self) -> dict:
        """获取当前增强参数
        
        Returns:
            增强参数字典
        """
        return {
            'brightness': self.brightness_factor,
            'contrast': self.contrast_factor,
            'gamma': self.gamma_factor,
            'exposure': self.exposure_factor
        }
    
    def reset_enhancement_params(self):
        """重置增强参数为默认值"""
        self.brightness_factor = 1.0
        self.contrast_factor = 1.0
        self.gamma_factor = 1.0
        self.exposure_factor = 1.0
        
        self.logger.debug("增强参数已重置为默认值")
    
    def create_dummy_model(self, save_path: str):
        """创建虚拟模型用于测试
        
        Args:
            save_path: 保存路径
        """
        try:
            # 创建虚拟模型
            model = DCENet(self.num_iterations)
            
            # 保存模型
            torch.save(model.state_dict(), save_path)
            
            self.logger.info(f"虚拟Zero-DCE模型已创建: {save_path}")
            
        except Exception as e:
            self.logger.error(f"虚拟模型创建失败: {e}")
            raise

def download_pretrained_model(save_dir: str) -> str:
    """下载预训练模型
    
    Args:
        save_dir: 保存目录
        
    Returns:
        模型文件路径
    """
    import urllib.request
    from pathlib import Path
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / "zero_dce.pth"
    
    # 如果模型已存在，直接返回
    if model_path.exists():
        return str(model_path)
    
    # 这里应该是实际的模型下载URL
    # 由于没有实际的预训练模型，我们创建一个虚拟模型
    try:
        model = DCENet()
        torch.save(model.state_dict(), model_path)
        print(f"虚拟Zero-DCE模型已创建: {model_path}")
        return str(model_path)
        
    except Exception as e:
        print(f"模型下载/创建失败: {e}")
        raise