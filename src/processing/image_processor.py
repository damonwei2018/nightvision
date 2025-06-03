# -*- coding: utf-8 -*-
"""
图像处理器

整合Zero-DCE和DexiNed模型，实现完整的夜视效果处理
"""

import time
from typing import Union, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from ..models.model_manager import ModelManager
from ..models.zero_dce import ZeroDCEModel
from ..models.dexined import DexiNedModel
from ..utils.config import Config
from ..utils.logger import LoggerMixin

class NightVisionProcessor(LoggerMixin):
    """夜视效果处理器"""
    
    # 预设的夜视风格
    NIGHT_VISION_STYLES = {
        'thermal_green': {
            'name': '热成像绿色',
            'edge_color': (0, 255, 0),  # 绿色
            'background_tint': (0, 40, 0),  # 深绿色背景
            'edge_alpha': 0.8,
            'background_alpha': 0.3
        },
        'military_yellow': {
            'name': '军用夜视黄绿色',
            'edge_color': (0, 255, 128),  # 黄绿色
            'background_tint': (0, 30, 15),  # 深黄绿色背景
            'edge_alpha': 0.9,
            'background_alpha': 0.2
        },
        'classic_white': {
            'name': '经典夜视白色',
            'edge_color': (255, 255, 255),  # 白色
            'background_tint': (20, 20, 20),  # 深灰色背景
            'edge_alpha': 1.0,
            'background_alpha': 0.1
        },
        'custom': {
            'name': '自定义',
            'edge_color': (0, 255, 0),
            'background_tint': (0, 40, 0),
            'edge_alpha': 0.8,
            'background_alpha': 0.3
        }
    }
    
    def __init__(self, config: Config):
        """初始化夜视处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.model_manager = ModelManager(config)
        
        # 初始化模型
        self.zero_dce = ZeroDCEModel(config)
        self.dexined = DexiNedModel(config)
        
        # 注册模型
        self.model_manager.register_model('zero_dce', self.zero_dce)
        self.model_manager.register_model('dexined', self.dexined)
        
        # 处理参数
        self.current_style = 'thermal_green'
        self.enable_enhancement = True
        self.enable_edge_detection = True
        
        # 输出设置
        self.output_size = None  # None表示保持原始尺寸
        self.output_quality = 95
        
        self.logger.info("夜视处理器初始化完成")
    
    def load_models(self, zero_dce_path: Optional[str] = None, dexined_path: Optional[str] = None) -> bool:
        """加载模型
        
        Args:
            zero_dce_path: Zero-DCE模型路径
            dexined_path: DexiNed模型路径
            
        Returns:
            是否加载成功
        """
        success = True
        
        # 加载Zero-DCE模型
        if zero_dce_path and Path(zero_dce_path).exists():
            if not self.model_manager.load_model('zero_dce', zero_dce_path):
                self.logger.warning("Zero-DCE模型加载失败")
                success = False
        else:
            self.logger.warning("Zero-DCE模型路径无效或文件不存在")
            # 创建虚拟模型用于测试
            if zero_dce_path:
                try:
                    self.zero_dce.create_dummy_model(zero_dce_path)
                    self.model_manager.load_model('zero_dce', zero_dce_path)
                except Exception as e:
                    self.logger.error(f"创建虚拟Zero-DCE模型失败: {e}")
                    success = False
        
        # 加载DexiNed模型
        if dexined_path and Path(dexined_path).exists():
            if not self.model_manager.load_model('dexined', dexined_path):
                self.logger.warning("DexiNed模型加载失败")
                success = False
        else:
            self.logger.warning("DexiNed模型路径无效或文件不存在")
            # 创建虚拟模型用于测试
            if dexined_path:
                try:
                    self.dexined.create_dummy_model(dexined_path)
                    self.model_manager.load_model('dexined', dexined_path)
                except Exception as e:
                    self.logger.error(f"创建虚拟DexiNed模型失败: {e}")
                    success = False
        
        return success
    
    def process_image(self, 
                     image: Union[str, Image.Image, np.ndarray],
                     progress_callback: Optional[callable] = None) -> np.ndarray:
        """处理图像
        
        Args:
            image: 输入图像（路径、PIL图像或numpy数组）
            progress_callback: 进度回调函数
            
        Returns:
            处理后的图像
        """
        start_time = time.time()
        processing_result = {
            'success': False,
            'stages': {
                'loading': False,
                'enhancement': False,
                'edge_detection': False,
                'night_vision': False
            },
            'errors': []
        }
        
        # 输入验证
        if image is None:
            self.logger.error("输入图像为None")
            error_image = np.zeros((400, 600, 3), dtype=np.uint8)
            try:
                import cv2
                cv2.putText(error_image, "Input image is None", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except:
                pass
            return error_image
            
        try:
            # 输入验证
            if isinstance(image, str):
                # 检查文件是否存在
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"图像文件不存在: {image}")
                if not image_path.is_file():
                    raise ValueError(f"指定路径不是文件: {image}")
                    
                # 检查文件大小
                file_size_mb = image_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 100:  # 超过100MB的图像可能导致内存问题
                    self.logger.warning(f"图像文件较大 ({file_size_mb:.1f} MB)，可能导致内存不足")
                
                try:
                    # 尝试打开图像文件
                    pil_image = Image.open(image)
                    # 验证图像格式
                    pil_image.verify()  # 验证图像数据
                    # 重新打开图像，因为verify后图像文件指针已移动
                    pil_image = Image.open(image)
                except Exception as e:
                    raise ValueError(f"无法打开或验证图像文件: {e}")
            elif isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, np.ndarray):
                if image.size == 0:
                    raise ValueError("空的numpy数组")
                try:
                    pil_image = Image.fromarray(image)
                except Exception as e:
                    raise ValueError(f"无法从numpy数组创建图像: {e}")
            else:
                raise ValueError(f"不支持的图像类型: {type(image)}")
            
            # 检查图像尺寸
            width, height = pil_image.size
            if width * height > 25000000:  # 超过2500万像素可能导致内存问题
                self.logger.warning(f"图像尺寸较大 ({width}x{height})，可能导致内存不足")
                
                # 监控内存使用
                try:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_usage_mb = memory_info.rss / (1024 * 1024)
                    available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
                    
                    self.logger.info(f"当前内存使用: {memory_usage_mb:.1f} MB, 可用内存: {available_memory_mb:.1f} MB")
                    
                    # 如果可用内存不足，尝试缩小图像
                    estimated_memory_needed = width * height * 4 * 3 / (1024 * 1024)  # 粗略估计处理过程中需要的内存
                    if estimated_memory_needed > available_memory_mb * 0.7:  # 如果预计使用超过70%的可用内存
                        scale_factor = min(1.0, (available_memory_mb * 0.5) / estimated_memory_needed)
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        self.logger.warning(f"内存不足，自动缩小图像至 {new_width}x{new_height}")
                        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                except ImportError:
                    self.logger.warning("无法导入psutil模块，跳过内存监控")
                except Exception as e:
                    self.logger.warning(f"内存监控失败: {e}")
            
            # 确保是RGB格式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 标记加载阶段成功
            processing_result['stages']['loading'] = True
            
            if progress_callback:
                progress_callback(10, "图像加载完成")
            
            # 图像增强
            enhanced_image = pil_image
            enhancement_success = False
            if self.enable_enhancement:
                if self.model_manager.is_model_loaded('zero_dce'):
                    self.logger.debug("开始图像增强")
                    try:
                        enhanced_image = self._enhance_image(pil_image)
                        enhancement_success = True
                        processing_result['stages']['enhancement'] = True
                    except Exception as e:
                        self.logger.error(f"图像增强失败: {e}")
                        # 使用原图继续处理
                        enhanced_image = pil_image
                        processing_result['errors'].append(f"图像增强失败: {str(e)}")
                else:
                    self.logger.warning("Zero-DCE模型未加载，跳过图像增强")
                    processing_result['stages']['enhancement'] = True
                
                if progress_callback:
                    progress_callback(50, "图像增强" + ("完成" if enhancement_success else "失败，使用原图"))
            else:
                processing_result['stages']['enhancement'] = True
            
            # 边缘检测
            edge_map = None
            edge_detection_success = False
            if self.enable_edge_detection:
                if self.model_manager.is_model_loaded('dexined'):
                    self.logger.debug("开始边缘检测")
                    try:
                        edge_map = self._detect_edges(enhanced_image)
                        if edge_map is not None:
                            edge_detection_success = True
                            processing_result['stages']['edge_detection'] = True
                    except Exception as e:
                        self.logger.error(f"边缘检测失败: {e}")
                        processing_result['errors'].append(f"边缘检测失败: {str(e)}")
                else:
                    self.logger.warning("DexiNed模型未加载，跳过边缘检测")
                    processing_result['stages']['edge_detection'] = True
                
                if progress_callback:
                    progress_callback(80, "边缘检测" + ("完成" if edge_detection_success else "失败"))
            else:
                processing_result['stages']['edge_detection'] = True
            
            # 应用夜视效果
            try:
                result = self._apply_night_vision_effect(enhanced_image, edge_map)
                processing_result['stages']['night_vision'] = True
            except Exception as e:
                error_msg = f"应用夜视效果失败: {str(e)}"
                self.logger.error(error_msg)
                processing_result['errors'].append(error_msg)
                # 如果夜视效果失败，返回增强后的图像
                result = np.array(enhanced_image)
            
            if progress_callback:
                progress_callback(100, "处理完成")
            
            # 标记整体处理成功
            processing_result['success'] = True
            
            # 记录处理时间和结果
            processing_time = time.time() - start_time
            self.logger.info(
                f"图像处理完成，耗时: {processing_time:.2f}s, "
                f"增强: {'成功' if enhancement_success else '失败'}, "
                f"边缘检测: {'成功' if edge_detection_success else '失败'}, "
                f"错误数: {len(processing_result['errors'])}"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"图像处理失败: {e}, 耗时: {processing_time:.2f}s")
            # 记录详细错误信息
            import traceback
            error_details = traceback.format_exc()
            self.logger.debug(f"错误详情:\n{error_details}")
            processing_result['errors'].append(f"图像处理失败: {str(e)}")
            
            # 如果图像已加载，返回原始图像
            if processing_result['stages']['loading']:
                return np.array(pil_image)
            else:
                # 如果连图像加载都失败了，创建一个错误图像
                error_image = np.zeros((400, 600, 3), dtype=np.uint8)
                # 添加错误文本
                try:
                    import cv2
                    cv2.putText(error_image, "Image Processing Failed", (50, 200), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                except:
                    pass  # 如果cv2导入失败，就返回纯黑图像
                return error_image
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """图像增强
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        try:
            # 使用Zero-DCE模型进行增强
            enhanced_array = self.model_manager.predict('zero_dce', image)
            
            # 转换回PIL图像
            if isinstance(enhanced_array, np.ndarray):
                enhanced_image = Image.fromarray(enhanced_array)
            else:
                enhanced_image = enhanced_array
            
            return enhanced_image
            
        except Exception as e:
            self.logger.warning(f"图像增强失败，使用原图: {e}")
            return image
    
    def _detect_edges(self, image: Image.Image) -> Optional[np.ndarray]:
        """边缘检测
        
        Args:
            image: 输入图像
            
        Returns:
            边缘图像
        """
        try:
            # 使用DexiNed模型进行边缘检测
            edge_map = self.model_manager.predict('dexined', image)
            
            # 确保是numpy数组
            if isinstance(edge_map, Image.Image):
                edge_map = np.array(edge_map)
            
            return edge_map
            
        except Exception as e:
            self.logger.warning(f"边缘检测失败: {e}")
            return None
    
    def _apply_night_vision_effect(self, 
                                  enhanced_image: Image.Image, 
                                  edge_map: Optional[np.ndarray]) -> np.ndarray:
        """应用夜视效果
        
        Args:
            enhanced_image: 增强后的图像
            edge_map: 边缘图像
            
        Returns:
            夜视效果图像
        """
        # 转换为numpy数组
        image_array = np.array(enhanced_image)
        
        # 获取当前风格参数
        style = self.NIGHT_VISION_STYLES[self.current_style]
        
        # 创建结果图像
        result = image_array.copy().astype(np.float32)
        
        # 应用背景色调
        if style['background_alpha'] > 0:
            background_tint = np.array(style['background_tint'], dtype=np.float32)
            result = result * (1 - style['background_alpha']) + background_tint * style['background_alpha']
        
        # 应用边缘效果
        if edge_map is not None and style['edge_alpha'] > 0:
            # 确保边缘图是单通道
            if len(edge_map.shape) == 3:
                edge_map = cv2.cvtColor(edge_map, cv2.COLOR_RGB2GRAY)
            
            # 调整边缘图尺寸
            if edge_map.shape != image_array.shape[:2]:
                edge_map = cv2.resize(edge_map, (image_array.shape[1], image_array.shape[0]))
            
            # 归一化边缘图
            edge_map = edge_map.astype(np.float32) / 255.0
            
            # 创建彩色边缘
            edge_color = np.array(style['edge_color'], dtype=np.float32)
            colored_edges = np.zeros_like(result)
            for i in range(3):
                colored_edges[:, :, i] = edge_map * edge_color[i]
            
            # 混合边缘和原图
            edge_mask = np.expand_dims(edge_map, axis=2)
            result = result * (1 - edge_mask * style['edge_alpha']) + colored_edges * style['edge_alpha']
        
        # 确保值在有效范围内
        result = np.clip(result, 0, 255)
        
        # 调整输出尺寸
        if self.output_size is not None:
            result = cv2.resize(result, self.output_size, interpolation=cv2.INTER_LINEAR)
        
        return result.astype(np.uint8)
    
    def set_enhancement_params(self, **params):
        """设置增强参数
        
        Args:
            **params: 增强参数
        """
        if self.model_manager.is_model_loaded('zero_dce'):
            self.zero_dce.set_enhancement_params(**params)
            self.logger.debug(f"增强参数已更新: {params}")
    
    def set_edge_params(self, **params):
        """设置边缘检测参数
        
        Args:
            **params: 边缘检测参数
        """
        if self.model_manager.is_model_loaded('dexined'):
            self.dexined.set_edge_params(**params)
            self.logger.debug(f"边缘检测参数已更新: {params}")
    
    def set_night_vision_style(self, style_name: str, custom_params: Optional[Dict] = None):
        """设置夜视风格
        
        Args:
            style_name: 风格名称
            custom_params: 自定义参数（当style_name为'custom'时使用）
        """
        if style_name in self.NIGHT_VISION_STYLES:
            self.current_style = style_name
            
            # 如果是自定义风格，更新参数
            if style_name == 'custom' and custom_params:
                self.NIGHT_VISION_STYLES['custom'].update(custom_params)
            
            self.logger.debug(f"夜视风格已设置为: {self.NIGHT_VISION_STYLES[style_name]['name']}")
        else:
            self.logger.warning(f"未知的夜视风格: {style_name}")
    
    def set_processing_options(self, 
                             enable_enhancement: bool = True,
                             enable_edge_detection: bool = True,
                             output_size: Optional[Tuple[int, int]] = None,
                             output_quality: int = 95):
        """设置处理选项
        
        Args:
            enable_enhancement: 是否启用图像增强
            enable_edge_detection: 是否启用边缘检测
            output_size: 输出尺寸 (width, height)
            output_quality: 输出质量 (1-100)
        """
        self.enable_enhancement = enable_enhancement
        self.enable_edge_detection = enable_edge_detection
        self.output_size = output_size
        self.output_quality = max(1, min(100, output_quality))
        
        self.logger.debug(
            f"处理选项已更新: 增强={enable_enhancement}, "
            f"边缘检测={enable_edge_detection}, "
            f"输出尺寸={output_size}, 质量={output_quality}"
        )
    
    def get_available_styles(self) -> Dict[str, str]:
        """获取可用的夜视风格
        
        Returns:
            风格字典 {key: name}
        """
        return {key: style['name'] for key, style in self.NIGHT_VISION_STYLES.items()}
    
    def get_current_style(self) -> Dict[str, Any]:
        """获取当前风格参数
        
        Returns:
            当前风格参数
        """
        return self.NIGHT_VISION_STYLES[self.current_style].copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'models': self.model_manager.get_all_models_info(),
            'device_info': self.model_manager.get_device_info(),
            'processing_options': {
                'enable_enhancement': self.enable_enhancement,
                'enable_edge_detection': self.enable_edge_detection,
                'output_size': self.output_size,
                'output_quality': self.output_quality
            },
            'current_style': self.current_style
        }
    
    def batch_process(self, 
                     input_paths: list,
                     output_dir: str,
                     progress_callback: Optional[callable] = None,
                     continue_on_error: bool = True,
                     max_retries: int = 1) -> Dict[str, Any]:
        """批量处理图像
        
        Args:
            input_paths: 输入图像路径列表
            output_dir: 输出目录
            progress_callback: 进度回调函数
            continue_on_error: 遇到错误时是否继续处理其他图像
            max_retries: 处理失败时的最大重试次数
            
        Returns:
            处理结果统计
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证输入路径列表
        valid_paths = []
        invalid_paths = []
        for path in input_paths:
            if not isinstance(path, (str, Path)):
                invalid_paths.append(str(path))
                continue
                
            path_obj = Path(path)
            if not path_obj.exists():
                invalid_paths.append(str(path))
                self.logger.warning(f"文件不存在: {path}")
                continue
                
            if not path_obj.is_file():
                invalid_paths.append(str(path))
                self.logger.warning(f"不是文件: {path}")
                continue
                
            # 检查是否是支持的图像格式
            try:
                with Image.open(path) as img:
                    img.verify()  # 验证图像数据
                valid_paths.append(path)
            except Exception as e:
                invalid_paths.append(str(path))
                self.logger.warning(f"无效的图像文件: {path}, 错误: {e}")
        
        results = {
            'total': len(input_paths),
            'valid': len(valid_paths),
            'invalid': len(invalid_paths),
            'success': 0,
            'failed': 0,
            'failed_files': [],
            'skipped_files': invalid_paths,
            'processing_time': 0,
            'retry_count': 0
        }
        
        if len(valid_paths) == 0:
            self.logger.error("没有有效的图像文件可处理")
            return results
        
        start_time = time.time()
        
        # 监控内存使用
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)
            self.logger.info(f"批处理开始，初始内存使用: {initial_memory:.1f} MB")
        except ImportError:
            self.logger.warning("无法导入psutil模块，跳过内存监控")
        except Exception as e:
            self.logger.warning(f"内存监控初始化失败: {e}")
        
        for i, input_path in enumerate(valid_paths):
            retry_count = 0
            success = False
            
            while not success and retry_count <= max_retries:
                if retry_count > 0:
                    self.logger.info(f"重试处理 ({retry_count}/{max_retries}): {input_path}")
                    
                try:
                    # 处理单张图像
                    result_image = self.process_image(input_path)
                    
                    # 生成输出文件名
                    input_file = Path(input_path)
                    output_file = output_dir / f"{input_file.stem}_nightvision{input_file.suffix}"
                    
                    # 保存结果
                    result_pil = Image.fromarray(result_image)
                    
                    # 确定输出格式和质量
                    save_format = self.config.get('processing.output_format', 'png').upper()
                    save_quality = self.output_quality
                    
                    # 使用临时文件进行原子性写入
                    import tempfile
                    import shutil
                    import os
                    
                    # 创建临时文件
                    fd, temp_path = tempfile.mkstemp(dir=output_dir)
                    os.close(fd)
                    
                    try:
                        # 保存到临时文件
                        if save_format == 'JPEG' or save_format == 'JPG':
                            result_pil.save(temp_path, format='JPEG', quality=save_quality)
                        elif save_format == 'PNG':
                            result_pil.save(temp_path, format='PNG')
                        elif save_format == 'WEBP':
                            result_pil.save(temp_path, format='WEBP', quality=save_quality)
                        else:  # 默认使用PNG
                            result_pil.save(temp_path, format='PNG')
                        
                        # 原子性地替换文件
                        if os.path.exists(output_file):
                            os.unlink(output_file)
                        shutil.move(temp_path, output_file)
                        
                        results['success'] += 1
                        success = True
                        self.logger.info(f"处理完成: {input_path} -> {output_file}")
                    except Exception as e:
                        # 确保清理临时文件
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                        raise e
                    
                except Exception as e:
                    retry_count += 1
                    results['retry_count'] += 1
                    
                    if retry_count > max_retries:
                        results['failed'] += 1
                        results['failed_files'].append(str(input_path))
                        self.logger.error(f"处理失败 (已重试 {retry_count-1} 次): {input_path}, 错误: {e}")
                        
                        # 如果不继续处理，则中断批处理
                        if not continue_on_error:
                            self.logger.error("由于错误且设置了不继续处理，批处理中断")
                            break
            
            # 更新进度
            if progress_callback:
                progress = int((i + 1) / len(valid_paths) * 100)
                status_text = f"已处理 {i + 1}/{len(valid_paths)} 张图像"
                if results['failed'] > 0:
                    status_text += f" (失败: {results['failed']})"
                progress_callback(progress, status_text)
            
            # 每处理10张图像，检查一次内存使用情况
            if (i + 1) % 10 == 0:
                try:
                    import psutil
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    self.logger.info(f"已处理 {i + 1} 张图像，当前内存使用: {current_memory:.1f} MB")
                    
                    # 如果内存使用过高，尝试手动触发垃圾回收
                    if current_memory > initial_memory * 2:  # 内存使用超过初始值的2倍
                        import gc
                        gc.collect()
                        self.logger.info("触发垃圾回收以释放内存")
                except Exception:
                    pass
        
        results['processing_time'] = time.time() - start_time
        
        # 最终内存使用情况
        try:
            import psutil
            final_memory = process.memory_info().rss / (1024 * 1024)
            self.logger.info(f"批处理结束，最终内存使用: {final_memory:.1f} MB (增加: {final_memory - initial_memory:.1f} MB)")
        except Exception:
            pass
        
        self.logger.info(
            f"批量处理完成: 成功 {results['success']}, "
            f"失败 {results['failed']}, "
            f"无效 {results['invalid']}, "
            f"重试 {results['retry_count']} 次, "
            f"耗时 {results['processing_time']:.2f}s"
        )
        
        return results
    
    def cleanup(self):
        """清理资源"""
        self.model_manager.cleanup()
        self.logger.info("夜视处理器资源清理完成")
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass