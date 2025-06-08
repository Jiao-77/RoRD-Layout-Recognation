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

推荐使用 `uv` 进行安装：
```bash
uv add torch torchvision opencv-python numpy Pillow
uv lock
uv sync
```

或者，您也可以使用 `pip`：
```bash
pip install torch torchvision opencv-python numpy Pillow
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

### 1. 训练模型

使用以下命令启动模型训练。训练过程采用自监督学习，通过对图像应用随机旋转来生成训练对，从而优化关键点检测和描述子生成。

```bash
python train.py --data_dir path/to/layouts --save_dir path/to/save
```

| 参数 | 描述 |
| :--- | :--- |
| `--data_dir` | **[必需]** 包含 PNG 格式 IC 版图图像的目录。 |
| `--save_dir` | **[必需]** 训练好的模型权重保存目录。 |

### 2. 评估模型

使用以下命令在验证集上评估模型的性能。评估脚本会计算基于 IoU 阈值的精确率、召回率和 F1 分数。

```bash
python evaluate.py --model_path path/to/model.pth --val_dir path/to/val/images --annotations_dir path/to/val/annotations --templates path/to/templates
```

| 参数 | 描述 |
| :--- | :--- |
| `--model_path` | **[必需]** 训练好的模型权重 (`.pth`) 文件路径。 |
| `--val_dir` | **[必需]** 验证集图像目录。 |
| `--annotations_dir` | **[必需]** 包含真实标注的 JSON 文件目录。 |
| `--templates` | **[必需]** 模板图像的路径列表。 |

### 3. 进行模板匹配

使用以下命令将模板图像与指定的版图图像进行匹配。匹配过程利用 RoRD 模型提取关键点和描述子，通过互最近邻（MNN）匹配和 RANSAC 几何验证来定位模板。

```bash
python match.py --model_path path/to/model.pth --layout_path path/to/layout.png --template_path path/to/template.png --output_path path/to/output.png
```

| 参数 | 描述 |
| :--- | :--- |
| `--model_path` | **[必需]** 训练好的模型权重 (`.pth`) 文件路径。 |
| `--layout_path` | **[必需]** 待匹配的版图图像路径。 |
| `--template_path` | **[必需]** 模板图像路径。 |
| `--output_path` | **[可选]** 保存可视化匹配结果的路径。 |

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