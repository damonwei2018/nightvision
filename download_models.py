#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本

自动下载Zero-DCE和DexiNed预训练模型
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

# 模型信息配置
MODEL_INFO = {
    'zero_dce': {
        'name': 'Zero-DCE ONNX Model',
        'filename': 'zero_dce.onnx',
        'url': 'https://github.com/Li-Chongyi/Zero-DCE/raw/master/model/Epoch99.pth',
        'sha256': '7c427bd4be38b0cc32b03d4e90b8862957d5c92d7f2b2b2d596b924c3c29c9b0',
        'size_mb': 0.25,
        'description': 'Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement',
        'format': 'pytorch',
        'onnx_output': 'zero_dce.onnx'
    },
    'dexined': {
        'name': 'DexiNed ONNX Model',
        'filename': 'dexined.onnx',
        'url': 'https://github.com/xavysp/DexiNed/raw/master/checkpoints/10_model.pth',
        'sha256': 'placeholder_hash_for_dexined_model',
        'size_mb': 124,
        'description': 'Dense Extreme Inception Network for Edge Detection',
        'format': 'pytorch',
        'onnx_output': 'dexined.onnx'
    }
}

# 备用下载源
ALTERNATIVE_SOURCES = {
    'zero_dce': [
        'https://huggingface.co/datasets/nightvision/models/resolve/main/zero_dce.pth',
        'https://drive.google.com/uc?export=download&id=1XxaHcwkbdpJ8dGj6F9POw0ZVheFKpgBT'
    ],
    'dexined': [
        'https://huggingface.co/datasets/nightvision/models/resolve/main/dexined.pth',
        'https://drive.google.com/uc?export=download&id=1MRUlg_mRwDiBiQWwn5ufIXjCn6zoyLDo'
    ],
    # 直接提供ONNX格式的备用链接
    'zero_dce_onnx': [
        'https://huggingface.co/datasets/nightvision/models/resolve/main/zero_dce.onnx',
        'https://drive.google.com/uc?export=download&id=1YQadMQHXZZ9zJwQjbpMw7wWwXnWwGJKj'
    ],
    'dexined_onnx': [
        'https://huggingface.co/datasets/nightvision/models/resolve/main/dexined.onnx',
        'https://drive.google.com/uc?export=download&id=1ZkBFhY_L1qE_kYRjw7-NH_OZYDjdLZSj'
    ]
}

class ModelDownloader:
    """模型下载器"""
    
    def __init__(self, models_dir: str = 'models'):
        """初始化下载器
        
        Args:
            models_dir: 模型保存目录
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # 创建临时下载目录
        self.temp_dir = self.models_dir / 'temp'
        self.temp_dir.mkdir(exist_ok=True)
    
    def download_file(self, url: str, filepath: Path, expected_size: Optional[int] = None) -> bool:
        """下载文件
        
        Args:
            url: 下载链接
            filepath: 保存路径
            expected_size: 期望文件大小（字节）
            
        Returns:
            是否下载成功
        """
        try:
            print(f"正在下载: {url}")
            
            # 发送请求
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            # 检查文件大小
            if expected_size and total_size > 0 and abs(total_size - expected_size) > expected_size * 0.1:
                print(f"警告: 文件大小不匹配 (期望: {expected_size}, 实际: {total_size})")
            
            # 下载文件
            with open(filepath, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=filepath.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"下载完成: {filepath}")
            return True
            
        except Exception as e:
            print(f"下载失败: {e}")
            if filepath.exists():
                filepath.unlink()  # 删除不完整的文件
            return False
    
    def verify_file(self, filepath: Path, expected_hash: str) -> bool:
        """验证文件完整性
        
        Args:
            filepath: 文件路径
            expected_hash: 期望的SHA256哈希值
            
        Returns:
            是否验证通过
        """
        if not filepath.exists():
            return False
        
        # 跳过占位符哈希值的验证
        if expected_hash.startswith('placeholder_'):
            print(f"跳过哈希验证: {filepath.name} (使用占位符哈希)")
            return True
        
        try:
            print(f"验证文件完整性: {filepath.name}")
            
            sha256_hash = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            file_hash = sha256_hash.hexdigest()
            
            if file_hash == expected_hash:
                print(f"文件验证通过: {filepath.name}")
                return True
            else:
                print(f"文件验证失败: {filepath.name}")
                print(f"期望哈希: {expected_hash}")
                print(f"实际哈希: {file_hash}")
                return False
                
        except Exception as e:
            print(f"文件验证出错: {e}")
            return False
    
    def download_model(self, model_name: str, force_download: bool = False) -> bool:
        """下载指定模型
        
        Args:
            model_name: 模型名称
            force_download: 是否强制重新下载
            
        Returns:
            是否下载成功
        """
        if model_name not in MODEL_INFO:
            print(f"未知模型: {model_name}")
            return False
        
        model_info = MODEL_INFO[model_name]
        model_path = self.models_dir / model_info['filename']
        
        # 检查文件是否已存在
        if model_path.exists() and not force_download:
            if self.verify_file(model_path, model_info['sha256']):
                print(f"模型已存在且验证通过: {model_path}")
                return True
            else:
                print(f"模型文件损坏，重新下载: {model_path}")
        
        print(f"\n开始下载 {model_info['name']}")
        print(f"描述: {model_info['description']}")
        print(f"大小: {model_info['size_mb']:.1f} MB")
        
        # 尝试主要下载源
        temp_path = self.temp_dir / model_info['filename']
        expected_size = int(model_info['size_mb'] * 1024 * 1024)
        
        success = self.download_file(model_info['url'], temp_path, expected_size)
        
        # 如果主要源失败，尝试备用源
        if not success and model_name in ALTERNATIVE_SOURCES:
            print("主要下载源失败，尝试备用源...")
            for alt_url in ALTERNATIVE_SOURCES[model_name]:
                print(f"尝试备用源: {alt_url}")
                success = self.download_file(alt_url, temp_path, expected_size)
                if success:
                    break
        
        if not success:
            print(f"所有下载源都失败: {model_name}")
            return False
        
        # 验证下载的文件
        if not self.verify_file(temp_path, model_info['sha256']):
            print(f"下载的文件验证失败: {model_name}")
            temp_path.unlink()
            return False
        
        # 移动到最终位置
        if model_path.exists():
            model_path.unlink()
        temp_path.rename(model_path)
        
        print(f"模型下载成功: {model_path}")
        return True
    
    def download_all_models(self, force_download: bool = False) -> Dict[str, bool]:
        """下载所有模型
        
        Args:
            force_download: 是否强制重新下载
            
        Returns:
            下载结果字典
        """
        results = {}
        
        print("开始下载所有模型...")
        print(f"模型保存目录: {self.models_dir.absolute()}")
        
        for model_name in MODEL_INFO.keys():
            print(f"\n{'='*50}")
            results[model_name] = self.download_model(model_name, force_download)
        
        # 显示下载结果摘要
        print(f"\n{'='*50}")
        print("下载结果摘要:")
        for model_name, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            print(f"  {model_name}: {status}")
        
        return results
    
    def create_dummy_models(self):
        """创建虚拟模型文件用于测试"""
        print("创建虚拟模型文件用于测试...")
        
        for model_name, model_info in MODEL_INFO.items():
            model_path = self.models_dir / model_info['filename']
            
            if not model_path.exists():
                # 创建一个小的虚拟ONNX文件
                dummy_content = b'\x08\x01\x12\x04test\x1a\x04test"\x04test'
                with open(model_path, 'wb') as f:
                    f.write(dummy_content)
                
                print(f"创建虚拟模型: {model_path}")
        
        print("虚拟模型创建完成")
    
    def list_models(self):
        """列出所有可用模型"""
        print("可用模型列表:")
        print(f"{'模型名称':<15} {'文件名':<20} {'大小':<10} {'状态':<10}")
        print("-" * 60)
        
        for model_name, model_info in MODEL_INFO.items():
            model_path = self.models_dir / model_info['filename']
            status = "已下载" if model_path.exists() else "未下载"
            size_str = f"{model_info['size_mb']:.1f} MB"
            
            print(f"{model_name:<15} {model_info['filename']:<20} {size_str:<10} {status:<10}")
    
    def cleanup_temp(self):
        """清理临时文件"""
        if self.temp_dir.exists():
            for file in self.temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
            print("临时文件清理完成")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Night Vision 模型下载器')
    parser.add_argument('--models-dir', default='models', help='模型保存目录')
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    parser.add_argument('--list', action='store_true', help='列出所有模型')
    parser.add_argument('--dummy', action='store_true', help='创建虚拟模型用于测试')
    parser.add_argument('--model', help='下载指定模型')
    parser.add_argument('--cleanup', action='store_true', help='清理临时文件')
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = ModelDownloader(args.models_dir)
    
    try:
        if args.list:
            downloader.list_models()
        elif args.dummy:
            downloader.create_dummy_models()
        elif args.cleanup:
            downloader.cleanup_temp()
        elif args.model:
            success = downloader.download_model(args.model, args.force)
            sys.exit(0 if success else 1)
        else:
            # 下载所有模型
            results = downloader.download_all_models(args.force)
            
            # 检查是否有失败的下载
            failed_models = [name for name, success in results.items() if not success]
            if failed_models:
                print(f"\n警告: 以下模型下载失败: {', '.join(failed_models)}")
                print("你可以稍后重试，或使用 --dummy 参数创建虚拟模型进行测试")
                sys.exit(1)
            else:
                print("\n所有模型下载成功！")
    
    except KeyboardInterrupt:
        print("\n下载被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n下载过程中出现错误: {e}")
        sys.exit(1)
    finally:
        downloader.cleanup_temp()

if __name__ == '__main__':
    main()