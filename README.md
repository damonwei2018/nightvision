# 🌙 Night Vision - 夜视效果图像处理器

一个基于深度学习的专业夜视效果图像处理应用，集成了Zero-DCE低光增强和DexiNed边缘检测技术，提供多种夜视风格和实时预览功能。

## 🎯 项目状态

✅ **项目已完成并可正常使用！**

- AI模型成功集成并运行
- 所有夜视效果正常工作
- 用户界面响应流畅
- 实时预览功能完善

## 🖼️ 效果展示

NightVision能够将普通图像转换为专业的夜视效果：

- **原始图像**：普通的室内或室外照片
- **夜视效果**：经过AI增强和边缘检测后的专业夜视效果
- **多种风格**：支持热成像绿色、军用黄绿色、经典白色等多种风格

主要处理流程：
1. **Zero-DCE增强**：智能提升图像亮度和对比度
2. **DexiNed边缘检测**：精确检测物体轮廓
3. **夜视效果合成**：将增强图像与边缘信息融合

## ✨ 主要特性

### 🎨 夜视效果风格
- **热成像绿色** - 经典的热成像仪绿色描边效果
- **军用夜视黄绿色** - 军用夜视设备的黄绿色风格
- **经典夜视白色** - 传统夜视仪的白色描边效果
- **自定义风格** - 支持用户自定义颜色和参数

### 🚀 核心功能
- **智能图像增强** - 基于Zero-DCE的低光图像增强
- **精确边缘检测** - 使用DexiNed模型进行高质量边缘检测
- **实时预览** - 支持参数调节的即时效果显示
- **批量处理** - 多线程并行处理大量图像
- **多种输出格式** - 支持JPG、PNG等常见格式

### 💻 用户界面
- **现代化设计** - 深色主题，专业图像处理软件风格
- **直观操作** - 拖拽支持，分屏对比显示
- **实时反馈** - 处理进度条，性能监控
- **参数控制** - 丰富的调节选项和预设

## 📋 系统要求

### 最低配置
- **操作系统**: Windows 10/11 (64位)
- **内存**: 4GB RAM
- **存储**: 2GB 可用空间
- **Python**: 3.8 或更高版本

### 推荐配置
- **内存**: 8GB RAM 或更多
- **显卡**: 支持CUDA的NVIDIA显卡 (可选，用于GPU加速)
- **存储**: SSD硬盘，提升处理速度

## 🛠️ 安装指南

### 1. 克隆项目
```bash
git clone https://github.com/your-username/nightvision.git
cd nightvision
```

### 2. 创建虚拟环境
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 下载预训练模型

项目需要下载预训练模型才能正常工作：

```bash
# 运行模型下载脚本
python download_real_models.py
```

这将自动下载：
- **Zero-DCE模型** (`zero_dce_original.pth`) - 来自Hugging Face
- **DexiNed模型** (`dexined_original.pth`) - 来自官方GitHub

模型将保存在 `models/` 目录下。

### 5. 运行应用
```bash
python main.py
```

## 🎯 快速开始

### 基础使用
1. **启动应用** - 运行 `python main.py`
2. **加载图像** - 点击"打开图片"或拖拽图像到预览区
3. **选择风格** - 在控制面板选择夜视效果风格
4. **调节参数** - 使用滑块调整亮度、对比度等参数
5. **预览效果** - 实时查看处理结果
6. **保存结果** - 点击"保存"导出处理后的图像

### 批量处理
1. **选择批量模式** - 在文件面板点击"批量处理"
2. **添加文件** - 选择多个图像文件或整个文件夹
3. **设置输出** - 选择输出目录和格式
4. **开始处理** - 点击"开始处理"，监控进度

## ⚙️ 配置说明

### 配置文件位置
- 主配置文件: `config.json`
- 用户设置: `config/settings.json`
- 日志文件: `logs/nightvision.log`

### 主要配置项

#### 模型设置
```json
{
  "models": {
    "zero_dce": {
      "path": "models/zero_dce_original.pth",
      "use_original_model": true,
      "enhancement_params": {
        "brightness": 1.0,
        "contrast": 1.0,
        "gamma": 1.0
      }
    },
    "dexined": {
      "path": "models/dexined_original.pth",
      "edge_params": {
        "threshold": 0.5,
        "edge_width": 1,
        "use_nms": true,
        "output_index": -1
      }
    }
  }
}
```

## 🎨 夜视风格详解

### 热成像绿色 (Thermal Green)
- **特点**: 经典的热成像仪效果
- **颜色**: 绿色描边 (0, 255, 0)
- **适用**: 军事、安防、夜间监控

### 军用夜视黄绿色 (Military Yellow-Green)
- **特点**: 军用夜视设备风格
- **颜色**: 黄绿色描边 (0, 255, 128)
- **适用**: 战术应用、游戏设计

### 经典夜视白色 (Classic White)
- **特点**: 传统夜视仪效果
- **颜色**: 白色描边 (255, 255, 255)
- **适用**: 科幻风格、电影效果

### 自定义风格 (Custom)
- **特点**: 完全可定制
- **参数**: 边缘颜色、背景色调、透明度等
- **适用**: 特殊需求、创意设计

## 📊 性能优化

### GPU加速
- 支持NVIDIA CUDA加速
- 自动检测可用的GPU设备
- 可在配置中启用/禁用GPU使用

### 内存管理
- 智能缓存机制
- 大图像自动分块处理
- 可配置内存使用限制

### 并行处理
- 多线程批量处理
- 可调节工作线程数量
- 异步处理避免界面阻塞

## 🔧 开发指南

### 项目结构
```
nightvision/
├── main.py                    # 主程序入口
├── requirements.txt           # 依赖列表
├── config.json               # 配置文件
├── download_real_models.py   # 模型下载脚本
├── README.md                 # 项目说明
├── src/                      # 源代码
│   ├── __init__.py
│   ├── ui/                   # 用户界面
│   │   ├── main_window.py
│   │   └── widgets/
│   ├── models/               # 深度学习模型
│   │   ├── zero_dce.py      # Zero-DCE低光增强
│   │   ├── dexined.py       # DexiNed边缘检测
│   │   └── dexined_official.py  # 官方DexiNed实现
│   ├── processing/           # 图像处理
│   │   ├── image_processor.py
│   │   └── batch_processor.py
│   └── utils/                # 工具模块
│       ├── config.py
│       └── logger.py
├── models/                   # 模型文件目录
├── logs/                     # 日志文件
├── cache/                    # 缓存目录
└── third_party/              # 第三方代码
    ├── Zero-DCE/            # 官方Zero-DCE实现
    └── DexiNed/             # 官方DexiNed实现
```

### 技术架构说明

#### 模型实现
- **Zero-DCE**: 使用官方的 `enhance_net_nopool` 网络结构，100%兼容官方权重
- **DexiNed**: 集成官方的 DenseBlock 架构，通过包装器使用官方实现

#### 关键技术点
1. **模型兼容性**: 通过分析官方实现确保网络结构完全匹配
2. **权重加载**: 智能处理不同格式的checkpoint文件
3. **后处理增强**: 在官方模型基础上添加亮度、对比度、gamma调节

## 🐛 故障排除

### 常见问题

#### 模型加载失败
- **原因**: 模型文件不存在或损坏
- **解决**: 运行 `python download_real_models.py` 重新下载

#### GPU加速不可用
- **原因**: CUDA环境未正确安装
- **解决**: 安装对应版本的CUDA和cuDNN

#### 内存不足
- **原因**: 处理大尺寸图像时内存溢出
- **解决**: 降低批处理数量，启用图像分块处理

#### 处理速度慢
- **原因**: CPU处理或模型未优化
- **解决**: 启用GPU加速，调整输入图像尺寸

### 日志分析
- 查看 `logs/nightvision.log` 获取详细错误信息
- 启用调试模式获取更多日志
- 使用性能分析工具监控资源使用

## 📝 更新日志

### v1.0.0 (2024-12-19)
- ✨ 初始版本发布
- 🎨 支持4种夜视风格
- 🚀 集成Zero-DCE和DexiNed官方模型
- 💻 现代化PyQt6界面
- 📊 批量处理功能
- ⚡ GPU加速支持
- 🔧 完整的模型下载和管理系统

### 最近更新
- 修复了AI模型调用失败的问题
- 集成官方DexiNed实现，提升边缘检测质量
- 添加了模型自动下载脚本
- 优化了日志系统

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) - 低光图像增强
- [DexiNed](https://github.com/xavysp/DexiNed) - 边缘检测
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📞 联系方式

- 项目主页: https://github.com/your-username/nightvision
- 问题反馈: https://github.com/your-username/nightvision/issues

---

**Night Vision** - 让每一张图片都拥有夜视效果 🌙✨