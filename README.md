# RoRD: 基于 AI 的集成电路版图识别

[//]: # (徽章占位符：您可以根据需要添加构建状态、版本号等徽章)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)

## 📖 描述

本项目实现了 **RoRD (Rotation-Robust Descriptors)** 模型，这是一种先进的局部特征匹配方法，专用于集成电路（IC）版图的识别。

IC 版图在匹配时可能出现多种方向（0°、90°、180°、270° 及其镜像），RoRD 模型通过其旋转鲁棒性设计，能够有效应对这一挑战。 项目采用自监督学习和随机旋转的数据增强策略，旨在解决 IC 版图识别中常见的数据稀缺性、几何多变性、动态扩展性和结构复杂性等问题。

### ✨ 主要功能

* **模型实现**：基于 D2-Net 架构，使用 PyTorch 实现了适用于 IC 版图的 RoRD 模型。
* **数据加载**：提供了自定义的 `ICLayoutDataset` 类，用于加载光栅化的 IC 版图图像。
* **训练脚本**：通过随机旋转生成训练对，以自监督的方式训练模型，确保其旋转鲁棒性。
* **评估脚本**：可在验证集上评估模型性能，计算精确率、召回率和 F1 分数。
* **匹配工具**：使用训练好的模型进行模板匹配，支持多实例检测和匹配结果的可视化。

## 🛠️ 安装

### 环境要求

* Python 3.8 或更高版本
* CUDA (可选, 推荐用于 GPU 加速)

### 依赖安装

**使用 uv（推荐）：**
```bash
# 安装 uv（如果尚未安装）
pip install uv

# 安装项目依赖
uv sync
```

**使用 pip：**
```bash
pip install -e .
```

## 🚀 使用方法

### 📁 项目结构

```
ic_layout_recognition/
├── data/
│   └── ic_dataset.py
├── models/
│   └── rord.py
├── utils/
│   └── transforms.py
├── train.py
├── evaluate.py
├── match.py
├── LICENSE.txt
└── README.md
```

## 🚀 使用方法

### 📋 训练准备清单

在开始训练前，请确保完成以下准备：

#### 1. 数据准备
- **训练数据**：准备PNG格式的布局图像（如电路板布局、建筑平面图等）
- **数据目录结构**：
  ```
  your_data_directory/
  ├── image1.png
  ├── image2.png
  └── ...
  ```

#### 2. 配置文件修改
编辑 `config.py` 文件，修改以下路径：
```python
# 必需修改的路径
LAYOUT_DIR = '你的布局图像目录路径'        # 训练数据目录
SAVE_DIR = '你的模型和日志保存路径'        # 输出目录
```

#### 3. 环境检查
确保已正确安装所有依赖：
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

### 🎯 开始训练

#### 基础训练
```bash
python train.py --data_dir /path/to/your/layouts --save_dir /path/to/your/models
```

#### 自定义训练参数
```bash
python train.py \
    --data_dir /path/to/your/layouts \
    --save_dir /path/to/your/models \
    --epochs 50 \
    --batch_size 8 \
    --lr 5e-5
```

#### 查看所有可用参数
```bash
python train.py --help
```

### 📊 训练监控
训练过程中会在 `SAVE_DIR` 目录下生成：
- 日志文件：`training_YYYYMMDD_HHMMSS.log`
- 最佳模型：`rord_model_best.pth`
- 最终模型：`rord_model_final.pth`

### 🚀 快速开始示例
```bash
# 1. 安装依赖
uv sync

# 2. 修改配置文件
# 编辑 config.py 中的 LAYOUT_DIR 和 SAVE_DIR

# 3. 开始训练
python train.py --data_dir ./data/layouts --save_dir ./output

# 4. 使用训练好的模型进行匹配
python match.py --model_path ./output/rord_model_final.pth \
                --layout ./test/layout.png \
                --template ./test/template.png \
                --output ./result.png
```

### 4. 模板匹配
```bash
python match.py --model_path /path/to/your/models/rord_model_final.pth \
                --layout /path/to/layout.png \
                --template /path/to/template.png \
                --output /path/to/result.png
```

### 5. 评估模型
```bash
python evaluate.py --model_path /path/to/your/models/rord_model_final.pth \
                   --val_dir /path/to/val/images \
                   --annotations_dir /path/to/val/annotations \
                   --templates_dir /path/to/templates
```

## 📦 数据准备

### 训练数据

* **格式**: PNG 格式的 IC 版图图像，可从 GDSII 或 OASIS 文件光栅化得到。
* **要求**: 数据集应包含多个版图图像，建议分辨率适中（例如 1024x1024）。
* **存储**: 将所有训练图像存放在一个目录中（例如 `path/to/layouts`）。

### 验证数据

* **图像**: PNG 格式的验证集图像，存储在指定目录（例如 `path/to/val/images`）。
* **模板**: 所有模板图像存储在单独的目录中（例如 `path/to/templates`）。
* **标注**: 真实标注信息以 JSON 格式提供，文件名需与对应的验证图像一致，并存储在指定目录（例如 `path/to/val/annotations`）。

JSON 标注文件示例：
```json
{
    "boxes": [
        {"template": "template1.png", "x": 100, "y": 200, "width": 50, "height": 50},
        {"template": "template2.png", "x": 300, "y": 400, "width": 60, "height": 60}
    ]
}
```

## 🧠 模型架构

RoRD 模型基于 D2-Net 架构，并使用 VGG-16 作为其骨干网络。

* **检测头**: 用于检测关键点，输出一个概率图。
* **描述子头**: 生成 128 维的旋转鲁棒描述子，专门为 IC 版图的 8 个离散旋转方向进行了适配。

模型通过自监督学习进行训练，利用 0° 到 360° 的随机旋转生成训练对，以同时优化关键点的检测重复性和描述子的相似性。

## 📊 结果

[待补充：请在此处添加预训练模型的链接或基准测试结果。]

* **预训练模型**: [链接待补充]
* **验证集评估指标**:
    * 精确率: X
    * 召回率: Y
    * F1 分数: Z

## 📄 许可协议

本项目根据 [Apache License 2.0](LICENSE.txt) 授权。