# RoRD: 基于 AI 的集成电路版图识别

![Python Version](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red)
![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)
![Tests](https://img.shields.io/badge/Tests-111%20passed-brightgreen)
![Code Quality](https://img.shields.io/badge/Code%20Quality-A-green)

## ⚡ Quick Start（含合成数据与H校验）

```bash
# 一键生成→渲染→预览→H校验→写回配置（开启合成混采与 Elastic）
uv run python tools/synth_pipeline.py \
  --out_root data/synthetic \
  --num 50 \
  --dpi 600 \
  --config configs/base_config.yaml \
  --ratio 0.3 \
  --enable_elastic \
  --validate_h --validate_n 6
```

提示：zsh 下使用反斜杠续行时，确保每行末尾只有一个 `\` 且下一行不要粘连参数（避免如 `6uv` 这样的粘连）。

可选：为 KLayout 渲染指定图层配色/线宽/背景（示例：金属层绿色、过孔红色、黑底）
```bash
uv run python tools/layout2png.py \
  --in data/synthetic/gds --out data/synthetic/png --dpi 800 \
  --layermap '1/0:#00FF00,2/0:#FF0000' --line_width 2 --bgcolor '#000000'
```

## 📖 描述

本项目实现了 **RoRD (Rotation-Robust Descriptors)** 模型，这是一种先进的局部特征匹配方法，专用于集成电路（IC）版图的识别。

IC 版图在匹配时可能出现多种方向（0°、90°、180°、270° 及其镜像），RoRD 模型通过其**几何感知损失函数**和**曼哈顿结构优化**的设计，能够有效应对这一挑战。项目采用**几何结构学习**而非纹理学习的训练策略，专门针对 IC 版图的二值化、稀疏性、重复结构和曼哈顿几何特征进行了深度优化。

👉 增量报告与性能分析见：`docs/reports/Increment_Report_2025-10-20.md`

👉 代码审查报告见：`docs/codereview/2026-03-17_code_review_report.md`

### ✨ 主要功能

* **模型实现**：基于 D2-Net 思路，使用 PyTorch 实现了适用于 IC 版图的 RoRD 模型，**专门针对几何结构学习优化**；支持可切换骨干（`vgg16` / `resnet34` / `efficientnet_b0`）。
* **数据加载**：提供了自定义的 `ICLayoutDataset` 类，用于加载光栅化的 IC 版图图像，支持**曼哈顿几何感知采样**。
* **训练脚本**：通过**几何感知损失函数**训练模型，学习**几何结构描述子**而非纹理特征，确保对二值化、稀疏性、重复结构的鲁棒性。
* **评估脚本**：可在验证集上评估模型性能，**专门针对IC版图特征**计算几何一致性指标。
* **匹配工具**：支持 FPN 多尺度推理与滑窗两种路径，并提供**KD-Tree优化的半径NMS**去重；可直接输出多实例匹配结果。
* **统一配置系统**：使用 dataclass 定义类型安全的配置结构，支持从 YAML 文件加载。
* **完整测试套件**：111 个单元测试，覆盖模型、损失函数、数据集和匹配逻辑。

## 🛠️ 安装

### 环境要求

* Python 3.12 或更高版本
* CUDA (可选, 推荐用于 GPU 加速)

### 依赖安装

**使用 uv（推荐）：**
```bash
# 安装 uv（如果尚未安装）
pip install uv

# 安装项目依赖
uv sync

# 安装开发依赖（包含测试工具）
uv sync --all-extras
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
│   ├── codereview/               # 代码审查报告
│   ├── data_description.md
│   ├── feature_work.md
│   ├── loss_function.md
│   └── NextStep.md
├── models/
│   └── rord.py                   # RoRD 模型与 FPN，多骨干支持
├── utils/
│   ├── config.py                 # 统一配置系统（dataclass）
│   ├── config_loader.py          # YAML 配置加载（已废弃）
│   ├── data_utils.py
│   └── transforms.py
├── losses.py                     # 几何感知损失集合
├── train.py                      # 训练脚本（YAML + TensorBoard）
├── evaluate.py                   # 评估脚本
├── match.py                      # 模板匹配脚本（FPN / 滑窗 + KD-Tree NMS）
├── tests/
│   ├── test_losses.py            # 损失函数单元测试
│   ├── test_dataset.py           # 数据集单元测试
│   ├── test_model.py             # 模型单元测试
│   ├── test_match.py             # 匹配逻辑单元测试
│   ├── benchmark_fpn.py          # FPN vs 滑窗性能对标
│   ├── benchmark_backbones.py    # 多骨干 A/B 前向基准
│   ├── benchmark_attention.py    # 注意力 none/se/cbam A/B 基准
│   └── benchmark_grid.py         # 三维基准：Backbone × Attention × Single/FPN
├── config.py                     # 兼容旧流程的 YAML 读取 shim（已废弃）
├── pyproject.toml
└── README.md
```

### 🧩 配置系统

项目使用统一的配置系统，基于 dataclass 实现类型安全的配置访问：

```python
from utils.config import RoRDConfig, ModelConfig, BackboneConfig

# 从 YAML 加载配置
cfg = RoRDConfig.from_yaml("configs/base_config.yaml")

# 类型安全的访问
print(cfg.model.backbone.name)      # "vgg16"
print(cfg.training.learning_rate)   # 5e-5

# 创建模型
from models.rord import RoRD
model = RoRD(model_config=cfg.model)
```

**配置结构**：
- `RoRDConfig` - 主配置类
  - `ModelConfig` - 模型配置（骨干、注意力、FPN）
  - `TrainingConfig` - 训练配置
  - `MatchingConfig` - 匹配配置
  - `EvaluationConfig` - 评估配置
  - `LoggingConfig` - 日志配置
  - `PathsConfig` - 路径配置
  - `AugmentConfig` - 数据增强配置
  - `DataSourcesConfig` - 数据源配置

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

### 📈 TensorBoard 实验追踪

```bash
uv run tensorboard --logdir runs
```

TensorBoard 中将展示：

- `train.py`：损失、学习率、梯度范数等随时间变化曲线；
- `evaluate.py`：精确率 / 召回率 / F1 分数；
- `match.py`（配合 `--tb_log_matches`）：每个匹配实例的内点数量、尺度和总检测数量。

### 🚀 快速开始示例
```bash
# 1. 安装依赖
uv sync

# 2. 复制并编辑 YAML 配置
cp configs/base_config.yaml configs/exp_ic_baseline.yaml

# 3. 开始训练
uv run python train.py --config configs/exp_ic_baseline.yaml

# 4. 使用训练好的模型进行匹配
uv run python match.py --config configs/exp_ic_baseline.yaml \
                       --model_path ./output/rord_model_final.pth \
                       --layout ./test/layout.png \
                       --template ./test/template.png \
                       --output ./result.png
```

## 🧪 测试

项目包含完整的单元测试套件，覆盖核心功能：

```bash
# 运行所有测试
PYTHONPATH=. uv run pytest tests/ -v

# 运行特定测试文件
PYTHONPATH=. uv run pytest tests/test_model.py -v
PYTHONPATH=. uv run pytest tests/test_losses.py -v
PYTHONPATH=. uv run pytest tests/test_dataset.py -v
PYTHONPATH=. uv run pytest tests/test_match.py -v

# 运行测试并生成覆盖率报告
PYTHONPATH=. uv run pytest tests/ -v --cov=. --cov-report=html
```

**测试覆盖**：
- `test_model.py` - 模型前向传播、不同骨干网络、FPN、设备兼容性、新配置系统
- `test_losses.py` - 损失函数计算、数值稳定性、梯度检查
- `test_dataset.py` - 数据加载、数据增强、单应性生成
- `test_match.py` - NMS 算法、互近邻匹配、旋转角度提取、匹配评分

## 📦 数据准备

### 训练数据

* **格式**: PNG 格式的 IC 版图图像，可从 GDSII 或 OASIS 文件光栅化得到。
* **要求**: 数据集应包含多个版图图像，建议分辨率适中（例如 1024x1024）。
* **存储**: 将所有训练图像存放在一个目录中（例如 `path/to/layouts`）。

### 验证数据

* **图像**: PNG 格式的验证集图像，存储在指定目录（例如 `path/to/val/images`）。
* **模板**: 所有模板图像存储在单独的目录中（例如 `path/to/templates`）。
* **标注**: 真实标注信息以 JSON 格式提供，文件名需与对应的验证图像一致。

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

## 🔎 推理与匹配（FPN 路径与 KD-Tree NMS）

项目已支持通过 FPN 单次推理产生多尺度特征，并使用 **KD-Tree 优化的半径 NMS** 去重：

### NMS 性能优化

使用 scipy.spatial.KDTree 优化 NMS 算法，复杂度从 O(N²) 降至 O(N log N)：

| 关键点数量 | 向量化 | KD-Tree | 加速比 |
|-----------|--------|---------|--------|
| 100 | 1.48ms | 0.33ms | 4.4x |
| 1000 | 6.51ms | 1.39ms | 4.7x |
| 5000 | 19.62ms | 3.72ms | 5.3x |

### 配置示例

```yaml
model:
  fpn:
    enabled: true
    out_channels: 256
    levels: [2, 3, 4]

  backbone:
    name: "vgg16"                  # 可选：vgg16 | resnet34 | efficientnet_b0
    pretrained: false

  attention:
    enabled: false
    type: "none"                   # 可选：none | cbam | se
    places: []                      # 插入位置：backbone_high | det_head | desc_head

matching:
  use_fpn: true
  nms:
    enabled: true
    radius: 4
    score_threshold: 0.5
```

## 📊 性能基准

### 运行 A/B 基准

```bash
# 多骨干基准
PYTHONPATH=. uv run python tests/benchmark_backbones.py --device cpu --image-size 512 --runs 5

# 注意力机制基准
PYTHONPATH=. uv run python tests/benchmark_attention.py --device cpu --image-size 512 --runs 10

# 三维网格基准
PYTHONPATH=. uv run python tests/benchmark_grid.py --device cpu --image-size 512 --runs 3
```

### FPN vs 滑窗对标

```bash
PYTHONPATH=. uv run python tests/benchmark_fpn.py --device cpu --image-size 512
```

## 📈 代码质量

### 代码审查结果

项目已完成全面的代码审查，修复了 26 个问题：

| 严重程度 | 数量 | 状态 |
|---------|------|------|
| 🔴 严重 | 5 | ✅ 全部修复 |
| 🟠 中等 | 6 | ✅ 全部修复 |
| 🟡 轻微 | 5 | ✅ 全部修复 |
| 🔵 性能 | 3 | ✅ 全部修复 |
| 🟣 架构 | 3 | ✅ 全部修复 |
| 🔒 安全 | 2 | ✅ 全部修复 |

详细报告见：`docs/codereview/2026-03-17_code_review_report.md`

### 主要优化

1. **配置系统重构**：使用 dataclass 实现类型安全的配置
2. **NMS 算法优化**：使用 KD-Tree 降低复杂度至 O(N log N)
3. **数值稳定性**：使用 `torch.linalg.inv` 替代 `torch.inverse`
4. **设备兼容性**：自动检测 GPU/CPU，支持 `map_location`
5. **性能优化**：预计算旋转矩阵，合并图像转换操作

## 📄 许可协议

本项目根据 [Apache License 2.0](LICENSE.txt) 授权。

---

## 🧪 合成数据一键流程与常见问题

### 一键命令
```bash
uv run python tools/generate_synthetic_layouts.py --out_dir data/synthetic/gds --num 200 --seed 42
uv run python tools/layout2png.py --in data/synthetic/gds --out data/synthetic/png --dpi 600
uv run python tools/preview_dataset.py --dir data/synthetic/png --out preview.png --n 8 --elastic
uv run python train.py --config configs/base_config.yaml
```

或使用单脚本一键执行（含配置写回）：
```bash
uv run python tools/synth_pipeline.py --out_root data/synthetic --num 200 --dpi 600 \
  --config configs/base_config.yaml --ratio 0.3 --enable_elastic
```

### YAML 关键片段
```yaml
synthetic:
  enabled: true
  png_dir: data/synthetic/png
  ratio: 0.3

augment:
  elastic:
    enabled: true
    alpha: 40
    sigma: 6
    alpha_affine: 6
    prob: 0.3
```

### 参数建议
- DPI：600–900；图形极细时可到 1200（注意磁盘占用与 IO）。
- ratio：数据少取 0.3–0.5；中等 0.2–0.3；数据多 0.1–0.2。
- Elastic：alpha=40, sigma=6, prob=0.3 为安全起点。

### FAQ
- 找不到 `klayout`：安装系统级 KLayout 并加入 PATH；或使用回退（gdstk+SVG）。
- `cairosvg`/`gdstk` 报错：升级版本、确认写权限、检查输出目录存在。
- 训练集为空：检查 `paths.layout_dir` 与 `synthetic.png_dir` 是否存在且包含 .png；若 syn 目录为空将自动仅用真实数据。