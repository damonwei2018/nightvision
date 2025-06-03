#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载真正的预训练模型脚本
"""

import os
import requests
from pathlib import Path
import gdown
import torch
import numpy as np

def download_file(url, save_path):
    """下载文件"""
    print(f"正在下载: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"下载完成: {save_path}")

def download_zero_dce_model():
    """下载Zero-DCE预训练模型"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Zero-DCE模型地址（从Hugging Face Spaces）
    # 使用直接下载链接
    url = "https://huggingface.co/spaces/IanNathaniel/Zero-DCE/resolve/main/Epoch99.pth"
    save_path = models_dir / "zero_dce_original.pth"
    
    if not save_path.exists():
        try:
            print("下载Zero-DCE模型...")
            # 使用wget命令下载
            import subprocess
            subprocess.run(["wget", "-O", str(save_path), url], check=True)
            print("Zero-DCE模型下载成功")
            return save_path
        except Exception as e:
            print(f"下载Zero-DCE模型失败: {e}")
            return None
    else:
        print(f"Zero-DCE模型已存在: {save_path}")
        return save_path

def download_dexined_model():
    """下载DexiNed预训练模型"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # DexiNed checkpoint下载链接（从Google Drive）
    # 注意：这是原始仓库提供的模型
    gdrive_id = "1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu"
    save_path = models_dir / "dexined_original.pth"
    
    if not save_path.exists():
        try:
            print("下载DexiNed模型...")
            gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", str(save_path), quiet=False)
            print("DexiNed模型下载成功")
            return save_path
        except Exception as e:
            print(f"下载DexiNed模型失败: {e}")
            return None
    else:
        print(f"DexiNed模型已存在: {save_path}")
        return save_path

def create_dummy_onnx_model(model_name, output_path):
    """创建一个有效的虚拟ONNX模型用于测试"""
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
            
        def forward(self, x):
            return self.conv(x)
    
    # 创建模型
    model = DummyModel()
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # 导出为ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
    )
    
    print(f"创建虚拟ONNX模型: {output_path}")

def main():
    """主函数"""
    print("开始下载真正的预训练模型...")
    
    # 确保模型目录存在
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 下载模型
    zero_dce_path = download_zero_dce_model()
    dexined_path = download_dexined_model()
    
    # 如果下载失败，创建有效的虚拟ONNX模型
    if not zero_dce_path:
        print("\n由于无法下载Zero-DCE模型，创建虚拟ONNX模型...")
        create_dummy_onnx_model("zero_dce", models_dir / "zero_dce.onnx")
    
    if not dexined_path:
        print("\n由于无法下载DexiNed模型，创建虚拟ONNX模型...")
        create_dummy_onnx_model("dexined", models_dir / "dexined.onnx")
    
    print("\n模型准备完成！")
    
    # 如果成功下载了PyTorch模型，提示用户需要修改代码
    if zero_dce_path or dexined_path:
        print("\n注意：已下载PyTorch格式的模型，需要修改代码以支持.pth文件")
        print("或者使用转换脚本将.pth转换为.onnx格式")

if __name__ == "__main__":
    main() 