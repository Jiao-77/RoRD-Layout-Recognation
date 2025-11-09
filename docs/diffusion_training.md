# RoRD 扩散训练流程

本文档介绍如何使用新的扩散模型训练流程，该流程不再使用程序生成的版图图片，而是使用原始数据和扩散模型生成的相似图像进行训练。

## 🔄 新的训练流程

### 原有流程问题
- 依赖程序化生成的IC版图图像
- 程序生成的图像可能缺乏真实数据的复杂性和多样性
- 数据来源比例控制不够灵活

### 新流程优势
- **数据来源**：仅使用原始真实数据 + 扩散模型生成的相似图像
- **可控性**：通过配置文件精确控制两种数据源的比例
- **质量提升**：扩散模型基于真实数据学习，生成更真实的版图图像
- **完整管线**：从训练扩散模型到生成数据再到模型训练的一站式解决方案

## 📁 项目结构

```
RoRD-Layout-Recognation/
├── tools/diffusion/
│   ├── ic_layout_diffusion.py          # 扩散模型核心实现
│   ├── generate_diffusion_data.py      # 一键生成扩散数据
│   ├── train_layout_diffusion.py       # 原有扩散训练接口（兼容）
│   └── sample_layouts.py               # 原有扩散采样接口（兼容）
├── tools/setup_diffusion_training.py   # 一键设置脚本
├── configs/
│   └── base_config.yaml               # 更新的配置文件
└── train.py                           # 更新的训练脚本
```

## 🚀 快速开始

### 方法1：一键设置（推荐）

```bash
# 一键设置整个训练流程
python tools/setup_diffusion_training.py
```

这个脚本会：
1. 检查运行环境
2. 创建必要的目录
3. 生成示例配置文件
4. 训练扩散模型
5. 生成扩散数据
6. 启动RoRD模型训练

### 方法2：分步执行

#### 1. 手动训练扩散模型

```bash
# 训练扩散模型
python tools/diffusion/ic_layout_diffusion.py train \
    --data_dir data/layouts \
    --output_dir models/diffusion \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --image_size 256 \
    --augment
```

#### 2. 生成扩散数据

```bash
# 使用训练好的模型生成图像
python tools/diffusion/ic_layout_diffusion.py generate \
    --checkpoint models/diffusion/diffusion_final.pth \
    --output_dir data/diffusion_generated \
    --num_samples 200 \
    --image_size 256
```

#### 3. 更新配置文件

编辑 `configs/base_config.yaml`：

```yaml
data_sources:
  real:
    enabled: true
    ratio: 0.7  # 70% 真实数据
  diffusion:
    enabled: true
    png_dir: "data/diffusion_generated"
    ratio: 0.3  # 30% 扩散数据
```

#### 4. 开始训练

```bash
python train.py --config configs/base_config.yaml
```

## ⚙️ 配置文件说明

### 新的数据源配置

```yaml
data_sources:
  # 真实数据配置
  real:
    enabled: true        # 是否启用真实数据
    ratio: 0.7          # 在训练数据中的比例

  # 扩散数据配置
  diffusion:
    enabled: true                    # 是否启用扩散数据
    model_dir: "models/diffusion"    # 扩散模型保存目录
    png_dir: "data/diffusion_generated"  # 生成数据保存目录
    ratio: 0.3                       # 在训练数据中的比例

    # 扩散模型训练参数
    training:
      epochs: 100
      batch_size: 8
      lr: 1e-4
      image_size: 256
      timesteps: 1000
      augment: true

    # 扩散生成参数
    generation:
      num_samples: 200
      timesteps: 1000
```

### 兼容性配置

为了向后兼容，保留了原有的 `synthetic` 配置节，但建议使用新的 `data_sources` 配置。

## 🔧 高级用法

### 自定义扩散模型训练

```bash
# 自定义训练参数
python tools/diffusion/ic_layout_diffusion.py train \
    --data_dir /path/to/your/data \
    --output_dir /path/to/save/model \
    --epochs 200 \
    --batch_size 16 \
    --lr 5e-5 \
    --timesteps 1000 \
    --image_size 512 \
    --augment
```

### 批量生成数据

```bash
# 生成大量样本
python tools/diffusion/ic_layout_diffusion.py generate \
    --checkpoint models/diffusion/diffusion_final.pth \
    --output_dir data/large_diffusion_set \
    --num_samples 1000 \
    --image_size 256
```

### 使用一键生成脚本

```bash
# 完整的扩散数据生成管线
python tools/diffusion/generate_diffusion_data.py \
    --config configs/base_config.yaml \
    --data_dir data/layouts \
    --num_samples 500 \
    --ratio 0.4 \
    --epochs 150 \
    --batch_size 12
```

## 📊 性能对比

| 指标 | 原流程（程序生成） | 新流程（扩散生成） |
|------|------------------|------------------|
| 数据真实性 | 中等 | 高 |
| 训练稳定性 | 良好 | 优秀 |
| 泛化能力 | 中等 | 良好 |
| 配置灵活性 | 低 | 高 |
| 计算开销 | 低 | 中等 |

## 🛠️ 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   --batch_size 4
   ```

2. **扩散模型训练太慢**
   ```bash
   # 减少时间步数或epochs
   --timesteps 500
   --epochs 50
   ```

3. **生成图像质量不佳**
   ```bash
   # 增加训练轮数
   --epochs 200
   # 启用数据增强
   --augment
   ```

4. **数据目录不存在**
   ```bash
   # 检查路径并创建目录
   mkdir -p data/layouts
   # 放置您的原始IC版图图像到 data/layouts/
   ```

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- PIL (Pillow)
- PyYAML

### 可选依赖

- tqdm (用于进度条显示)
- tensorboard (用于训练可视化)

## 📝 API参考

### 扩散模型训练命令

```bash
python tools/diffusion/ic_layout_diffusion.py train [OPTIONS]
```

**选项：**
- `--data_dir`: 训练数据目录
- `--output_dir`: 模型保存目录
- `--image_size`: 图像尺寸 (默认: 256)
- `--batch_size`: 批次大小 (默认: 8)
- `--epochs`: 训练轮数 (默认: 100)
- `--lr`: 学习率 (默认: 1e-4)
- `--timesteps`: 扩散时间步数 (默认: 1000)
- `--augment`: 启用数据增强

### 扩散数据生成命令

```bash
python tools/diffusion/ic_layout_diffusion.py generate [OPTIONS]
```

**选项：**
- `--checkpoint`: 模型检查点路径
- `--output_dir`: 输出目录
- `--num_samples`: 生成样本数量
- `--image_size`: 图像尺寸
- `--timesteps`: 扩散时间步数

## 🔄 迁移指南

如果您之前使用程序生成的版图数据，请按以下步骤迁移：

1. **备份现有配置**
   ```bash
   cp configs/base_config.yaml configs/base_config_backup.yaml
   ```

2. **更新配置文件**
   - 设置 `synthetic.enabled: false`
   - 配置 `data_sources.diffusion.enabled: true`
   - 调整 `data_sources.diffusion.ratio` 到期望值

3. **生成新的扩散数据**
   ```bash
   python tools/diffusion/generate_diffusion_data.py --config configs/base_config.yaml
   ```

4. **重新训练模型**
   ```bash
   python train.py --config configs/base_config.yaml
   ```

## 🤝 贡献

欢迎提交问题报告和功能请求！如果您想贡献代码，请：

1. Fork 这个项目
2. 创建您的功能分支
3. 提交您的更改
4. 推送到分支
5. 创建一个 Pull Request

## 📄 许可证

本项目遵循原始项目的许可证。