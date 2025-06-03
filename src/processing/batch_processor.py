# -*- coding: utf-8 -*-
"""
批量处理器

实现多文件并行处理和进度管理
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from queue import Queue

from PIL import Image
import numpy as np

from .image_processor import NightVisionProcessor
from ..utils.config import Config
from ..utils.logger import LoggerMixin

class BatchProcessor(LoggerMixin):
    """批量处理器"""
    
    def __init__(self, config: Config, max_workers: int = 4):
        """初始化批量处理器
        
        Args:
            config: 配置对象
            max_workers: 最大工作线程数
        """
        self.config = config
        self.max_workers = max_workers
        
        # 创建图像处理器
        self.processor = NightVisionProcessor(config)
        
        # 处理状态
        self.is_processing = False
        self.is_cancelled = False
        self.current_task = None
        
        # 进度跟踪
        self.progress_queue = Queue()
        self.results_queue = Queue()
        
        # 支持的图像格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        self.logger.info(f"批量处理器初始化完成，最大工作线程数: {max_workers}")
    
    def load_models(self, zero_dce_path: Optional[str] = None, dexined_path: Optional[str] = None) -> bool:
        """加载模型
        
        Args:
            zero_dce_path: Zero-DCE模型路径
            dexined_path: DexiNed模型路径
            
        Returns:
            是否加载成功
        """
        return self.processor.load_models(zero_dce_path, dexined_path)
    
    def process_files(self,
                     input_files: List[str],
                     output_dir: str,
                     options: Dict[str, Any],
                     progress_callback: Optional[Callable] = None,
                     completion_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """批量处理文件
        
        Args:
            input_files: 输入文件列表
            output_dir: 输出目录
            options: 处理选项
            progress_callback: 进度回调函数 (progress, message)
            completion_callback: 完成回调函数 (results)
            
        Returns:
            处理结果
        """
        if self.is_processing:
            raise RuntimeError("批量处理正在进行中")
        
        self.is_processing = True
        self.is_cancelled = False
        
        try:
            # 验证输入文件
            valid_files = self._validate_input_files(input_files)
            if not valid_files:
                raise ValueError("没有有效的输入文件")
            
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 应用处理选项
            self._apply_processing_options(options)
            
            # 开始处理
            results = self._process_files_parallel(
                valid_files, 
                output_path, 
                options,
                progress_callback
            )
            
            # 调用完成回调
            if completion_callback:
                completion_callback(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"批量处理失败: {e}")
            raise
        finally:
            self.is_processing = False
    
    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         options: Dict[str, Any],
                         recursive: bool = True,
                         progress_callback: Optional[Callable] = None,
                         completion_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """处理目录中的所有图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            options: 处理选项
            recursive: 是否递归处理子目录
            progress_callback: 进度回调函数
            completion_callback: 完成回调函数
            
        Returns:
            处理结果
        """
        # 扫描目录中的图像文件
        input_files = self._scan_directory(input_dir, recursive)
        
        if not input_files:
            raise ValueError(f"目录 {input_dir} 中没有找到支持的图像文件")
        
        self.logger.info(f"在目录 {input_dir} 中找到 {len(input_files)} 个图像文件")
        
        # 处理文件
        return self.process_files(
            input_files,
            output_dir,
            options,
            progress_callback,
            completion_callback
        )
    
    def _validate_input_files(self, input_files: List[str]) -> List[str]:
        """验证输入文件
        
        Args:
            input_files: 输入文件列表
            
        Returns:
            有效的文件列表
        """
        valid_files = []
        
        for file_path in input_files:
            path = Path(file_path)
            
            # 检查文件是否存在
            if not path.exists():
                self.logger.warning(f"文件不存在: {file_path}")
                continue
            
            # 检查文件格式
            if path.suffix.lower() not in self.supported_formats:
                self.logger.warning(f"不支持的文件格式: {file_path}")
                continue
            
            # 检查文件是否可读
            try:
                with Image.open(file_path) as img:
                    img.verify()
                valid_files.append(file_path)
            except Exception as e:
                self.logger.warning(f"文件损坏或无法读取: {file_path}, 错误: {e}")
        
        self.logger.info(f"验证完成，有效文件数: {len(valid_files)}/{len(input_files)}")
        return valid_files
    
    def _scan_directory(self, directory: str, recursive: bool = True) -> List[str]:
        """扫描目录中的图像文件
        
        Args:
            directory: 目录路径
            recursive: 是否递归扫描
            
        Returns:
            图像文件路径列表
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"目录不存在或不是有效目录: {directory}")
        
        image_files = []
        
        # 扫描模式
        pattern = '**/*' if recursive else '*'
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def _apply_processing_options(self, options: Dict[str, Any]):
        """应用处理选项
        
        Args:
            options: 处理选项字典
        """
        # 设置夜视风格
        if 'style' in options:
            style_name = options['style']
            custom_params = options.get('custom_style_params')
            self.processor.set_night_vision_style(style_name, custom_params)
        
        # 设置增强参数
        if 'enhancement_params' in options:
            self.processor.set_enhancement_params(**options['enhancement_params'])
        
        # 设置边缘检测参数
        if 'edge_params' in options:
            self.processor.set_edge_params(**options['edge_params'])
        
        # 设置处理选项
        processing_options = {
            'enable_enhancement': options.get('enable_enhancement', True),
            'enable_edge_detection': options.get('enable_edge_detection', True),
            'output_size': options.get('output_size'),
            'output_quality': options.get('output_quality', 95)
        }
        self.processor.set_processing_options(**processing_options)
    
    def _process_files_parallel(self,
                               input_files: List[str],
                               output_dir: Path,
                               options: Dict[str, Any],
                               progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """并行处理文件
        
        Args:
            input_files: 输入文件列表
            output_dir: 输出目录
            options: 处理选项
            progress_callback: 进度回调函数
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        results = {
            'total': len(input_files),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'failed_files': [],
            'skipped_files': [],
            'processing_time': 0,
            'output_files': []
        }
        
        # 检查是否保留目录结构
        preserve_structure = options.get('preserve_structure', False)
        overwrite = options.get('overwrite', False)
        output_format = options.get('output_format', 'same')  # 'same', 'jpg', 'png'
        
        # 使用线程池处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_file = {}
            for i, input_file in enumerate(input_files):
                if self.is_cancelled:
                    break
                
                # 生成输出文件路径
                output_file = self._generate_output_path(
                    input_file, output_dir, preserve_structure, output_format
                )
                
                # 检查是否跳过已存在的文件
                if output_file.exists() and not overwrite:
                    results['skipped'] += 1
                    results['skipped_files'].append(str(input_file))
                    if progress_callback:
                        progress = int((i + 1) / len(input_files) * 100)
                        progress_callback(progress, f"跳过已存在文件: {output_file.name}")
                    continue
                
                # 提交处理任务
                future = executor.submit(
                    self._process_single_file,
                    input_file,
                    output_file,
                    options
                )
                future_to_file[future] = (input_file, output_file)
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_file):
                if self.is_cancelled:
                    break
                
                input_file, output_file = future_to_file[future]
                completed += 1
                
                try:
                    success = future.result()
                    if success:
                        results['success'] += 1
                        results['output_files'].append(str(output_file))
                        self.logger.debug(f"处理成功: {input_file} -> {output_file}")
                    else:
                        results['failed'] += 1
                        results['failed_files'].append(str(input_file))
                        
                except Exception as e:
                    results['failed'] += 1
                    results['failed_files'].append(str(input_file))
                    self.logger.error(f"处理失败: {input_file}, 错误: {e}")
                
                # 更新进度
                if progress_callback:
                    progress = int(completed / len(input_files) * 100)
                    progress_callback(
                        progress, 
                        f"已处理 {completed}/{len(input_files)} 个文件"
                    )
        
        results['processing_time'] = time.time() - start_time
        
        self.logger.info(
            f"批量处理完成: 成功 {results['success']}, "
            f"失败 {results['failed']}, 跳过 {results['skipped']}, "
            f"耗时 {results['processing_time']:.2f}s"
        )
        
        return results
    
    def _process_single_file(self,
                            input_file: str,
                            output_file: Path,
                            options: Dict[str, Any]) -> bool:
        """处理单个文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            options: 处理选项
            
        Returns:
            是否处理成功
        """
        try:
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 处理图像
            result_image = self.processor.process_image(input_file)
            
            # 保存结果
            result_pil = Image.fromarray(result_image)
            
            # 设置保存参数
            save_kwargs = {'quality': options.get('output_quality', 95)}
            if output_file.suffix.lower() == '.png':
                save_kwargs.pop('quality', None)  # PNG不支持quality参数
            
            result_pil.save(output_file, **save_kwargs)
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理文件失败: {input_file}, 错误: {e}")
            return False
    
    def _generate_output_path(self,
                             input_file: str,
                             output_dir: Path,
                             preserve_structure: bool,
                             output_format: str) -> Path:
        """生成输出文件路径
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            preserve_structure: 是否保留目录结构
            output_format: 输出格式
            
        Returns:
            输出文件路径
        """
        input_path = Path(input_file)
        
        # 确定输出文件名
        if output_format == 'same':
            output_name = f"{input_path.stem}_nightvision{input_path.suffix}"
        elif output_format == 'jpg':
            output_name = f"{input_path.stem}_nightvision.jpg"
        elif output_format == 'png':
            output_name = f"{input_path.stem}_nightvision.png"
        else:
            output_name = f"{input_path.stem}_nightvision{input_path.suffix}"
        
        # 确定输出路径
        if preserve_structure:
            # 保留相对路径结构
            relative_dir = input_path.parent
            output_path = output_dir / relative_dir / output_name
        else:
            # 直接放在输出目录
            output_path = output_dir / output_name
        
        return output_path
    
    def cancel_processing(self):
        """取消当前处理"""
        if self.is_processing:
            self.is_cancelled = True
            self.logger.info("批量处理已取消")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """获取处理状态
        
        Returns:
            处理状态信息
        """
        return {
            'is_processing': self.is_processing,
            'is_cancelled': self.is_cancelled,
            'max_workers': self.max_workers,
            'supported_formats': list(self.supported_formats)
        }
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式
        
        Returns:
            支持的格式列表
        """
        return list(self.supported_formats)
    
    def estimate_processing_time(self, file_count: int, avg_file_size_mb: float = 2.0) -> float:
        """估算处理时间
        
        Args:
            file_count: 文件数量
            avg_file_size_mb: 平均文件大小（MB）
            
        Returns:
            估算的处理时间（秒）
        """
        # 基于经验的估算公式
        # 假设每MB需要0.5秒处理时间，考虑并行处理的加速比
        base_time_per_mb = 0.5
        parallel_factor = min(self.max_workers, file_count) / self.max_workers
        
        estimated_time = (file_count * avg_file_size_mb * base_time_per_mb) / parallel_factor
        
        return max(estimated_time, file_count * 0.1)  # 最少每个文件0.1秒
    
    def cleanup(self):
        """清理资源"""
        self.cancel_processing()
        self.processor.cleanup()
        self.logger.info("批量处理器资源清理完成")
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass