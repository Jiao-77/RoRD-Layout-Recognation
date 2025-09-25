# RoRD: 基于 AI 的集成电路版图识别

[//]: # (徽章占位符：您可以根据需要添加构建状态、版本号等徽章)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)

## 📖 描述

本项目实现了 **RoRD (Rotation-Robust Descriptors)** 模型，这是一种先进的局部特征匹配方法，专用于集成电路（IC）版图的识别。

IC 版图在匹配时可能出现多种方向（0°、90°、180°、270° 及其镜像），RoRD 模型通过其**几何感知损失函数**和**曼哈顿结构优化**的设计，能够有效应对这一挑战。项目采用**几何结构学习**而非纹理学习的训练策略，专门针对 IC 版图的二值化、稀疏性、重复结构和曼哈顿几何特征进行了深度优化。

### ✨ 主要功能

* **模型实现**：基于 D2-Net 架构，使用 PyTorch 实现了适用于 IC 版图的 RoRD 模型，**专门针对几何结构学习优化**。
* **数据加载**：提供了自定义的 `ICLayoutDataset` 类，用于加载光栅化的 IC 版图图像，支持**曼哈顿几何感知采样**。
* **训练脚本**：通过**几何感知损失函数**训练模型，学习**几何结构描述子**而非纹理特征，确保对二值化、稀疏性、重复结构的鲁棒性。
* **评估脚本**：可在验证集上评估模型性能，**专门针对IC版图特征**计算几何一致性指标。
* **匹配工具**：使用训练好的模型进行**几何结构匹配**，有效区分重复图形并支持多实例检测。
* **灵活配置与日志**：引入 OmegaConf 驱动的 YAML 配置 (`configs/*.yaml`)，配合 `utils.config_loader` 与 TensorBoard 监控实现参数/路径集中管理。

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
RoRD-Layout-Recognation/
├── configs/
│   └── base_config.yaml          # YAML 配置入口
├── data/
│   └── ic_dataset.py             # 数据集与数据接口
├── docs/
│   ├── data_description.md
│   ├── feature_work.md
│   ├── loss_function.md
│   └── NextStep.md
├── models/
│   └── rord.py                   # RoRD 模型定义
├── utils/
│   ├── config_loader.py          # YAML 配置加载与路径转换
│   ├── data_utils.py
│   └── transforms.py
├── losses.py                     # 几何感知损失集合
├── train.py                      # 训练脚本（YAML + TensorBoard）
├── evaluate.py                   # 评估脚本
├── match.py                      # 模板匹配脚本
├── config.py                     # 兼容旧流程的 YAML 读取 shim
├── pyproject.toml
└── README.md
```

### 🧩 配置与模块化更新

- **YAML 配置中心**：所有路径与超参数集中存放在 `configs/*.yaml`，通过 `utils.config_loader.load_config` 统一解析；CLI 的 `--config` 参数可切换实验配置，`to_absolute_path` 则保证相对路径相对配置文件解析。
- **旧配置兼容**：`config.py` 现在仅作为兼容层，将 YAML 配置转换成原有的 Python 常量，便于逐步迁移历史代码。
- **损失与数据解耦**：`losses.py` 汇总几何感知损失，`data/ic_dataset.py` 与 `utils/data_utils.py` 分离数据准备逻辑，便于扩展新的采样策略或损失项。
- **日志体系**：`logging` 配置节配合 TensorBoard 集成，`train.py`、`evaluate.py`、`match.py` 可统一写入 `log_dir/子任务/experiment_name`。

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
项目默认从 `configs/base_config.yaml` 读取训练、评估与日志参数。建议复制该文件并按实验命名，例如：

```bash
cp configs/base_config.yaml configs/exp_ic_baseline.yaml
```

在 YAML 中修改路径与关键参数：

```yaml
paths:
  layout_dir: "数据集/训练图像目录"
  save_dir: "输出目录（模型与日志）"
  val_img_dir: "验证集图像目录"
  val_ann_dir: "验证集标注目录"
  template_dir: "模板图像目录"

training:
  num_epochs: 50
  batch_size: 8
  learning_rate: 5.0e-5

logging:
  use_tensorboard: true
  log_dir: "runs"
  experiment_name: "baseline"
```

> 保留 `config.py` 仅用于兼容旧版脚本；新流程全部通过 YAML + `utils.config_loader` 载入配置。

#### 3. 环境检查
确保已正确安装所有依赖：
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

### 🎯 开始训练

#### 基础训练
```bash
uv run python train.py --config configs/exp_ic_baseline.yaml
```

上述命令将读取 `configs/exp_ic_baseline.yaml` 中的路径和训练参数；若未指定 `--config`，脚本会回落到 `configs/base_config.yaml`。

#### 自定义训练参数
```bash
uv run python train.py \
  --config configs/exp_ic_baseline.yaml \
  --data_dir /override/layouts \
  --save_dir /override/models \
  --epochs 60 \
  --batch_size 16 \
  --lr 1e-4
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

### � TensorBoard 实验追踪

`configs/base_config.yaml` 中新增的 `logging` 区块用于控制 TensorBoard：

```yaml
logging:
  use_tensorboard: true          # 是否启用 TensorBoard 记录
  log_dir: "runs"                # 日志根目录（相对/绝对路径均可）
  experiment_name: "default"    # 实验名称，将作为子目录名
```

需要临时覆盖时，可在命令行传入参数（以下命令均可用 `uv run` 直接执行）：

```bash
uv run python train.py --log_dir logs --experiment_name exp001
uv run python evaluate.py --log_dir logs --experiment_name exp001
uv run python match.py --tb_log_matches --log_dir logs --experiment_name exp001
uv run python train.py --disable_tensorboard            # 如需关闭记录
```

执行训练、评估或模板匹配后，通过下列命令启动 TensorBoard：

```bash
uv run tensorboard --logdir runs
```

TensorBoard 中将展示：

- `train.py`：损失、学习率、梯度范数等随时间变化曲线；
- `evaluate.py`：精确率 / 召回率 / F1 分数；
- `match.py`（配合 `--tb_log_matches`）：每个匹配实例的内点数量、尺度和总检测数量。

### �🚀 快速开始示例
```bash
# 1. 安装依赖
uv sync

# 2. 复制并编辑 YAML 配置
cp configs/base_config.yaml configs/exp_ic_baseline.yaml
# 根据数据路径与实验需求调整 paths/training/logging 字段

# 3. 开始训练
uv run python train.py --config configs/exp_ic_baseline.yaml

# 4. 使用训练好的模型进行匹配
uv run python match.py --config configs/exp_ic_baseline.yaml \
                       --model_path ./output/rord_model_final.pth \
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

## 🧠 模型架构 - IC版图专用优化版

RoRD 模型基于 D2-Net 架构，使用 VGG-16 作为骨干网络，**专门针对IC版图的几何特征进行了深度优化**。

### 网络结构创新
* **检测头**: 用于检测**几何边界关键点**，输出二值化概率图，专门针对IC版图的黑白边界优化
* **描述子头**: 生成 128 维的**几何结构描述子**，而非纹理描述子，具有以下特性：
  - **曼哈顿几何感知**: 专门针对水平和垂直结构优化
  - **重复结构区分**: 能有效区分相同图形的不同实例
  - **二值化鲁棒性**: 对光照变化完全不变
  - **稀疏特征优化**: 专注于真实几何结构而非噪声

### 核心创新 - 几何感知损失函数
**专为IC版图特征设计**：
- **曼哈顿一致性损失**: 确保90度旋转下的几何一致性
- **稀疏性正则化**: 适应IC版图稀疏特征分布
- **二值化特征距离**: 强化几何边界特征，弱化灰度变化
- **几何感知困难负样本**: 基于结构相似性而非像素相似性选择负样本

## 🔎 推理与匹配（FPN 路径与 NMS）

项目已支持通过 FPN 单次推理产生多尺度特征，并在匹配阶段引入半径 NMS 去重以减少冗余关键点：

在 `configs/base_config.yaml` 中启用 FPN 与 NMS：

```yaml
model:
  fpn:
    enabled: true
    out_channels: 256
    levels: [2, 3, 4]

matching:
  use_fpn: true
  nms:
    enabled: true
    radius: 4
    score_threshold: 0.5
```

运行匹配并将过程写入 TensorBoard：

```bash
uv run python match.py \
  --config configs/base_config.yaml \
  --layout /path/to/layout.png \
  --template /path/to/template.png \
  --tb_log_matches
```

如需回退旧“图像金字塔”路径，将 `matching.use_fpn` 设为 `false` 即可。

也可使用 CLI 快捷开关临时覆盖：

```bash
# 关闭 FPN（等同 matching.use_fpn=false）
uv run python match.py --config configs/base_config.yaml --fpn_off \
  --layout /path/to/layout.png --template /path/to/template.png

# 关闭关键点去重（NMS）
uv run python match.py --config configs/base_config.yaml --no_nms \
  --layout /path/to/layout.png --template /path/to/template.png
```

### 训练策略 - 几何结构学习
模型通过**几何结构学习**策略进行训练：
- **曼哈顿变换生成训练对**: 利用90度旋转等曼哈顿变换
- **几何感知采样**: 优先采样水平和垂直方向的边缘点
- **结构一致性优化**: 学习几何结构描述子而非纹理特征
- **重复结构鲁棒性**: 有效处理IC版图中的大量重复图形

**关键区别**: 传统方法学习纹理特征，我们的方法**学习几何结构特征**，完美适应IC版图的二值化、稀疏性、重复结构和曼哈顿几何特征。

## 📊 结果

[待补充：请在此处添加预训练模型的链接或基准测试结果。]

* **预训练模型**: [链接待补充]
* **验证集评估指标**:
    * 精确率: X
    * 召回率: Y
    * F1 分数: Z

## 📄 许可协议

本项目根据 [Apache License 2.0](LICENSE.txt) 授权。