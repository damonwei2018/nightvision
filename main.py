#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NightVision - 夜视图像处理应用
主程序入口

作者: AI Assistant
版本: 1.0.0
描述: 基于Zero-DCE和DexiNed的夜视图像增强和描边处理应用
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QDir
from PyQt6.QtGui import QIcon
import qdarkstyle

from src.ui.main_window import MainWindow
from src.utils.logger import setup_logger
from src.utils.config import Config

def setup_application():
    """设置应用程序基本配置"""
    # 创建应用程序实例
    app = QApplication(sys.argv)
    app.setApplicationName("NightVision")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("NightVision Team")
    
    # 设置应用程序图标
    icon_path = project_root / "assets" / "icons" / "app_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # 应用深色主题
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    
    return app

def main():
    """主函数"""
    try:
        # 设置日志
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "nightvision.log"
        
        # 使用RotatingFileHandler，设置为DEBUG级别以捕获所有日志
        logger = setup_logger(
            log_file=str(log_file),
            level=10,  # DEBUG级别
            max_file_size=10 * 1024 * 1024,  # 10MB
            backup_count=5  # 保留5个备份文件
        )
        logger.info("启动 NightVision 应用程序")
        logger.debug("日志系统初始化完成，级别：DEBUG")
        
        # 加载配置
        config = Config()
        logger.debug("配置加载完成")
        
        # 创建应用程序
        app = setup_application()
        logger.debug("应用程序实例创建完成")
        
        # 创建主窗口
        logger.debug("开始创建主窗口...")
        main_window = MainWindow(config)
        logger.debug("主窗口创建完成")
        
        main_window.show()
        logger.info("应用程序界面已启动")
        
        # 运行应用程序
        sys.exit(app.exec())
        
    except Exception as e:
        if 'logger' in locals():
            logger.critical(f"应用程序启动失败: {e}", exc_info=True)
        print(f"应用程序启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()