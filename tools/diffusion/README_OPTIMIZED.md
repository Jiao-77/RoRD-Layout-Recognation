# 优化的IC版图扩散模型

针对曼哈顿多边形IC版图光栅化图像生成的去噪扩散模型优化版本。

## 🎯 优化目标

专门优化以曼哈顿多边形为全部组成元素的IC版图光栅化图像生成，主要特点：

- **曼哈顿几何感知**：模型架构专门处理水平/垂直线条特征
- **边缘锐化**：保持IC版图清晰的边缘特性
- **多尺度结构**：保持从微观到宏观的结构一致性
- **几何约束**：确保生成结果符合曼哈顿几何规则
- **后处理优化**：进一步提升生成质量

## 📁 文件结构

```
tools/diffusion/
├── ic_layout_diffusion_optimized.py  # 优化的核心模型实现
├── train_optimized.py                # 训练脚本
├── generate_optimized.py             # 生成脚本
├── run_optimized_pipeline.py         # 一键运行管线
├── README_OPTIMIZED.md              # 本文档
└── original/                        # 原始实现（参考用）
    ├── ic_layout_diffusion.py
    └── ...
```

## 🚀 快速开始

### 1. 基本使用 - 一键运行

```bash
# 完整管线（训练 + 生成）
python tools/diffusion/run_optimized_pipeline.py \
    --data_dir data/ic_layouts \
    --output_dir outputs/diffusion_optimized \
    --epochs 50 \
    --num_samples 200

# 仅生成（使用已有模型）
python tools/diffusion/run_optimized_pipeline.py \
    --skip_training \
    --checkpoint outputs/diffusion_optimized/model/best_model.pth \
    --data_dir data/ic_layouts \
    --output_dir outputs/diffusion_generated \
    --num_samples 500
```

### 2. 分步使用

#### 训练模型

```bash
python tools/diffusion/train_optimized.py \
    --data_dir data/ic_layouts \
    --output_dir models/diffusion_optimized \
    --image_size 256 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --edge_condition \
    --augment \
    --manhattan_weight 0.1
```

#### 生成样本

```bash
python tools/diffusion/generate_optimized.py \
    --checkpoint models/diffusion_optimized/best_model.pth \
    --output_dir generated_layouts \
    --num_samples 200 \
    --num_steps 50 \
    --use_ddim \
    --use_post_process
```

## 🔧 关键优化特性

### 1. 曼哈顿几何感知U-Net

```python
class ManhattanAwareUNet(nn.Module):
    """曼哈顿几何感知的U-Net架构"""

    def __init__(self, use_edge_condition=False):
        # 专门的方向感知卷积
        self.horiz_conv = nn.Conv2d(in_channels, 32, (1, 7), padding=(0, 3))
        self.vert_conv = nn.Conv2d(in_channels, 32, (7, 1), padding=(3, 0))
        self.standard_conv = nn.Conv2d(in_channels, 32, 3, padding=1)

        # 特征融合
        self.fusion = nn.Conv2d(96, 64, 3, padding=1)
```

**优势**：
- 专门提取水平和垂直特征
- 保持曼哈顿几何结构
- 增强线条检测能力

### 2. 多目标损失函数

```python
# 组合损失函数
total_loss = mse_loss +
             0.3 * edge_loss +           # 边缘感知损失
             0.2 * structure_loss +      # 多尺度结构损失
             0.1 * manhattan_loss       # 曼哈顿约束损失
```

**优势**：
- 保持边缘锐利度
- 维持多尺度结构一致性
- 强制曼哈顿几何约束

### 3. 几何保持的数据增强

```python
# 只使用不破坏曼哈顿几何的增强
self.aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # 移除旋转，保持几何约束
])
```

### 4. 后处理优化

```python
def manhattan_post_process(image):
    """曼哈顿化后处理"""
    # 形态学操作强化直角特征
    # 水平和垂直增强
    # 二值化处理
    return processed_image
```

## 📊 质量评估指标

生成样本会自动评估以下指标：

1. **曼哈顿几何合规性** - 角度偏差损失（越低越好）
2. **边缘锐度** - 边缘强度平均值
3. **对比度** - 图像标准差
4. **稀疏性** - 低像素值比例（IC版图特性）

## 🎛️ 参数调优指南

### 训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `manhattan_weight` | 0.05 - 0.2 | 曼哈顿约束权重 |
| `schedule_type` | cosine | 余弦调度通常效果更好 |
| `edge_condition` | True | 使用边缘条件提高质量 |
| `batch_size` | 4 - 8 | 根据GPU内存调整 |

### 生成参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `num_steps` | 20 - 50 | DDIM采样步数 |
| `eta` | 0.0 - 0.3 | 随机性控制（0=确定性） |
| `guidance_scale` | 1.0 - 3.0 | 引导强度 |
| `post_process_threshold` | 0.4 - 0.6 | 后处理阈值 |

## 🔍 故障排除

### 1. 训练问题

**Q: 损失不下降**
- 检查数据质量和格式
- 降低学习率
- 增加批次大小
- 调整曼哈顿权重

**Q: 生成的图像模糊**
- 增加边缘损失权重
- 使用边缘条件训练
- 调整后处理阈值
- 增加训练轮数

### 2. 生成问题

**Q: 生成结果不符合曼哈顿几何**
- 增加 `manhattan_weight`
- 启用后处理
- 降低 `eta` 参数

**Q: 生成速度慢**
- 使用DDIM采样
- 减少 `num_steps`
- 增加 `batch_size`

### 3. 内存问题

**Q: GPU内存不足**
- 减少批次大小
- 减小图像尺寸
- 使用梯度累积

## 📈 性能对比

| 特性 | 原始模型 | 优化模型 |
|------|----------|----------|
| 曼哈顿几何合规性 | ❌ | ✅ |
| 边缘锐度 | 中等 | 优秀 |
| 训练稳定性 | 一般 | 优秀 |
| 生成质量 | 基础 | 优秀 |
| 后处理 | 无 | 有 |
| 质量评估 | 无 | 有 |

## 🔄 与现有管线集成

更新配置文件以使用优化的扩散数据：

```yaml
synthetic:
  enabled: true
  ratio: 0.0  # 禁用程序化合成

  diffusion:
    enabled: true
    png_dir: "outputs/diffusion_optimized/generated"
    ratio: 0.3  # 扩散数据在训练中的比例
    model_checkpoint: "outputs/diffusion_optimized/model/best_model.pth"
```

## 📚 技术原理

### 曼哈顿几何约束

IC版图具有以下几何特征：
- 所有线条都是水平或垂直的
- 角度只能是90°
- 结构具有高度的规则性

模型通过以下方式强制这些约束：
1. 方向感知卷积核
2. 角度偏差损失函数
3. 几何保持后处理

### 多尺度结构损失

确保生成结果在不同尺度下都保持结构一致性：
- 原始分辨率：细节保持
- 2x下采样：中层结构
- 4x下采样：整体布局

## 🛠️ 开发者指南

### 添加新的损失函数

```python
class CustomLoss(nn.Module):
    def forward(self, pred, target):
        # 实现自定义损失
        return loss

# 在训练器中使用
self.custom_loss = CustomLoss()
```

### 自定义后处理

```python
def custom_post_process(image):
    # 实现自定义后处理逻辑
    return processed_image
```

## 📄 许可证

本项目遵循与主项目相同的许可证。

## 🤝 贡献

欢迎提交问题报告和改进建议！

---

**注意**：这是针对特定IC版图生成任务的优化版本，对于一般的图像生成任务，请使用原始的扩散模型实现。