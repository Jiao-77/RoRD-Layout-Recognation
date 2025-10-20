# 性能基准报告 — Backbone A/B 与 FPN 对比

最后更新：2025-10-20  
设备：CPU（无 GPU）  
输入：1×3×512×512 随机张量  
重复次数：5（每组）

> 说明：本报告为初步 CPU 前向测试，主要用于比较不同骨干的相对推理耗时。实际业务场景与 GPU 上的结论可能不同，建议在目标环境再次复测。

## 结果汇总（ms）

| Backbone           | Single Mean ± Std | FPN Mean ± Std |
|--------------------|-------------------:|----------------:|
| vgg16              | 392.03 ± 4.76      | 821.91 ± 4.17   |
| resnet34           | 105.01 ± 1.57      | 131.17 ± 1.66   |
| efficientnet_b0    | 62.02 ± 2.64       | 161.71 ± 1.58   |

- 备注：本次测试在 CPU 上进行，`gpu_mem_mb` 始终为 0。

## 注意力 A/B（CPU，resnet34，512×512，runs=10，places=backbone_high+desc_head）

| Attention | Single Mean ± Std | FPN Mean ± Std |
|-----------|-------------------:|----------------:|
| none      | 97.57 ± 0.55       | 124.57 ± 0.48   |
| se        | 101.48 ± 2.13      | 123.12 ± 0.50   |
| cbam      | 119.80 ± 2.38      | 123.11 ± 0.71   |

观察：
- 单尺度路径对注意力类型更敏感，CBAM 开销相对更高，SE 较轻；
- FPN 路径耗时在本次设置下差异很小（可能因注意力仅在 `backbone_high/desc_head`，且 FPN 头部计算占比较高）。

复现实验：
```zsh
PYTHONPATH=. uv run python tests/benchmark_attention.py \
  --device cpu --image-size 512 --runs 10 \
  --backbone resnet34 --places backbone_high desc_head
```

## 三维基准（Backbone × Attention × Single/FPN）

环境：CPU，输入 1×3×512×512，重复 3 次，places=backbone_high,desc_head。

| Backbone         | Attention | Single Mean ± Std (ms) | FPN Mean ± Std (ms) |
|------------------|-----------|-----------------------:|--------------------:|
| vgg16            | none      | 351.65 ± 1.88          | 719.33 ± 3.95       |
| vgg16            | se        | 349.76 ± 2.00          | 721.41 ± 2.74       |
| vgg16            | cbam      | 354.45 ± 1.49          | 744.76 ± 29.32      |
| resnet34         | none      | 90.99 ± 0.41           | 117.22 ± 0.41       |
| resnet34         | se        | 90.78 ± 0.47           | 115.91 ± 1.31       |
| resnet34         | cbam      | 96.50 ± 3.17           | 111.09 ± 1.01       |
| efficientnet_b0  | none      | 40.45 ± 1.53           | 127.30 ± 0.09       |
| efficientnet_b0  | se        | 46.48 ± 0.26           | 142.35 ± 6.61       |
| efficientnet_b0  | cbam      | 47.11 ± 0.47           | 150.99 ± 12.47      |

复现实验：

```zsh
PYTHONPATH=. uv run python tests/benchmark_grid.py \
  --device cpu --image-size 512 --runs 3 \
  --backbones vgg16 resnet34 efficientnet_b0 \
  --attentions none se cbam \
  --places backbone_high desc_head
```

运行会同时输出控制台摘要并保存 JSON：`benchmark_grid.json`。

## 观察与解读
- vgg16 明显最慢，FPN 额外的横向/上采样代价在 CPU 上更突出（>2×）。
- resnet34 在单尺度上显著快于 vgg16，FPN 增幅较小（约 +25%）。
- efficientnet_b0 单尺度最快，但 FPN 路径的额外代价相对较高（约 +161%）。

## 建议
1. 训练/推理优先考虑 resnet34 或 efficientnet_b0 替代 vgg16，以获得更好的吞吐；若业务更多依赖多尺度鲁棒性，则进一步权衡 FPN 的开销。
2. 在 GPU 与真实数据上复测：
  - 固定输入尺寸与批次，比较三种骨干在单尺度与 FPN 的耗时与显存。
  - 对齐预处理（`utils/data_utils.get_transform`）并验证检测/匹配效果。
3. 若选择 efficientnet_b0，建议探索更适配的中间层组合（例如 features[3]/[4]/[6]），以在精度与速度上取得更好的折中。

## 复现实验
- 安装依赖并在仓库根目录执行：

```zsh
# CPU 复现
PYTHONPATH=. uv run python tests/benchmark_backbones.py --device cpu --image-size 512 --runs 5

# CUDA 复现（如可用）
PYTHONPATH=. uv run python tests/benchmark_backbones.py --device cuda --runs 20 --backbones vgg16 resnet34 efficientnet_b0
```

## 附：脚本与实现位置
- 模型与 FPN 实现：`models/rord.py`
- 骨干 A/B 基准脚本：`tests/benchmark_backbones.py`
- 相关说明：`docs/description/Backbone_FPN_Test_Change_Notes.md`

# 🚀 性能基准测试报告

**完成日期**: 2025-10-20  
**测试工具**: `tests/benchmark_fpn.py`  
**对标对象**: FPN 推理 vs 滑窗推理  

---

## 📋 目录

1. [执行摘要](#执行摘要)
2. [测试环境](#测试环境)
3. [测试方法](#测试方法)
4. [测试数据](#测试数据)
5. [性能指标](#性能指标)
6. [对标结果](#对标结果)
7. [分析与建议](#分析与建议)
8. [使用指南](#使用指南)

---

## 执行摘要

本报告对比了 **FPN（特征金字塔网络）推理路径** 与 **传统滑窗推理路径** 的性能差异。

### 🎯 预期目标

| 指标 | 目标 | 说明 |
|------|------|------|
| **推理速度** | FPN 提速 ≥ 30% | 同输入条件下，FPN 路径应快 30% 以上 |
| **内存占用** | 内存节省 ≥ 20% | GPU 显存占用应降低 20% 以上 |
| **检测精度** | 无下降 | 关键点数和匹配内点数应相当或更优 |

---

## 测试环境

### 硬件配置

```yaml
GPU: NVIDIA CUDA 计算能力 >= 7.0（可选 CPU）
内存: >= 8GB RAM
显存: >= 8GB VRAM（推荐 16GB+）
```

### 软件环境

```yaml
Python: >= 3.12
PyTorch: >= 2.7.1
CUDA: >= 12.1（如使用 GPU）
关键依赖:
  - torch
  - torchvision
  - numpy
  - psutil （用于内存监测）
```

### 配置文件

使用默认配置 `configs/base_config.yaml`：

```yaml
model:
  fpn:
    enabled: true
    out_channels: 256
    levels: [2, 3, 4]

matching:
  keypoint_threshold: 0.5
  pyramid_scales: [0.75, 1.0, 1.5]
  inference_window_size: 1024
  inference_stride: 768
  use_fpn: true
  nms:
    enabled: true
    radius: 4
    score_threshold: 0.5
```

---

## 测试方法

### 1. 测试流程

```
┌─────────────────────────────────────┐
│      加载模型与预处理配置            │
└────────────┬────────────────────────┘
             │
    ┌────────▼────────┐
    │  FPN 路径测试   │
    │  (N 次运行)     │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ 滑窗路径测试    │
    │  (N 次运行)     │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  计算对标指标   │
    │  生成报告       │
    └─────────────────┘
```

### 2. 性能指标采集

每个方法的每次运行采集以下指标：

| 指标 | 说明 | 单位 |
|------|------|------|
| **推理时间** | 从特征提取到匹配完成的总耗时 | ms |
| **关键点数** | 检测到的关键点总数 | 个 |
| **匹配数** | 通过互近邻匹配的对应点对数 | 个 |
| **GPU 内存** | 推理过程中显存峰值 | MB |

### 3. 运行方式

**基础命令**:
```bash
uv run python tests/benchmark_fpn.py \
  --layout /path/to/layout.png \
  --template /path/to/template.png \
  --num-runs 5 \
  --output benchmark_results.json
```

**完整参数**:
```bash
uv run python tests/benchmark_fpn.py \
  --config configs/base_config.yaml \
  --model_path path/to/save/model_final.pth \
  --layout /path/to/layout.png \
  --template /path/to/template.png \
  --num-runs 5 \
  --output benchmark_results.json \
  --device cuda
```

---

## 测试数据

### 数据集要求

测试数据应满足以下条件：

| 条件 | 说明 | 推荐值 |
|------|------|--------|
| **版图尺寸** | 大版图，代表实际应用场景 | ≥ 2000×2000 px |
| **模板尺寸** | 中等尺寸，能在版图中找到 | 500×500~1000×1000 px |
| **版图类型** | 实际电路版图或相似图像 | PNG/JPEG 格式 |
| **模板类型** | 版图中的某个器件或结构 | PNG/JPEG 格式 |
| **质量** | 清晰，具代表性 | 适当的对比度和细节 |

### 数据准备步骤

1. **准备版图和模板**
   ```bash
   # 将测试数据放在合适位置
   mkdir -p test_data
   cp /path/to/layout.png test_data/
   cp /path/to/template.png test_data/
   ```

2. **验证数据
   ```bash
   # 检查图像尺寸和格式
   python -c "
   from PIL import Image
   layout = Image.open('test_data/layout.png')
   template = Image.open('test_data/template.png')
   print(f'Layout size: {layout.size}')
   print(f'Template size: {template.size}')
   "
   ```

---

## 性能指标

### 1. 原始数据格式

测试脚本输出 JSON 文件，包含以下结构：

```json
{
  "timestamp": "2025-10-20 14:30:45",
  "config": "configs/base_config.yaml",
  "model_path": "path/to/model_final.pth",
  "layout_path": "test_data/layout.png",
  "layout_size": [3000, 2500],
  "template_path": "test_data/template.png",
  "template_size": [800, 600],
  "device": "cuda:0",
  "fpn": {
    "method": "FPN",
    "mean_time_ms": 245.32,
    "std_time_ms": 12.45,
    "min_time_ms": 230.21,
    "max_time_ms": 268.91,
    "all_times_ms": [...],
    "mean_keypoints": 1523.4,
    "mean_matches": 187.2,
    "gpu_memory_mb": 1024.5,
    "num_runs": 5
  },
  "sliding_window": {
    "method": "Sliding Window",
    "mean_time_ms": 352.18,
    "std_time_ms": 18.67,
    ...
  },
  "comparison": {
    "speedup_percent": 30.35,
    "memory_saving_percent": 21.14,
    "fpn_faster": true,
    "meets_speedup_target": true,
    "meets_memory_target": true
  }
}
```

### 2. 主要性能指标

**推理时间**:
- 平均耗时 (mean_time_ms)
- 标准差 (std_time_ms)
- 最小/最大耗时范围

**关键点检测**:
- 平均关键点数量
- 影响因素：keypoint_threshold，NMS 半径

**匹配性能**:
- 平均匹配对数量
- 反映特征匹配质量

**内存效率**:
- GPU 显存占用 (MB)
- CPU 内存占用可选

### 3. 对标指标

| 指标 | 计算公式 | 目标值 | 说明 |
|------|---------|--------|------|
| **推理速度提升** | (SW_time - FPN_time) / SW_time × 100% | ≥ 30% | 正值表示 FPN 更快 |
| **内存节省** | (SW_mem - FPN_mem) / SW_mem × 100% | ≥ 20% | 正值表示 FPN 更省 |
| **精度保证** | FPN_matches ≥ SW_matches × 0.95 | ✅ | 匹配数不显著下降 |

---

## 对标结果

### 测试执行

运行测试脚本，预期输出示例：

```
================================================================================
                            性能基准测试结果
================================================================================

指标                        FPN                  滑窗
----------------------------------------------------------------------
平均推理时间 (ms)          245.32               352.18
标准差 (ms)                12.45                18.67
最小时间 (ms)              230.21               328.45
最大时间 (ms)              268.91               387.22

平均关键点数               1523                 1687
平均匹配数                 187                  189

GPU 内存占用 (MB)          1024.5               1305.3

================================================================================
                              对标结果
================================================================================

推理速度提升: +30.35% ✅
  (目标: ≥30% | 达成: 是)

内存节省: +21.14% ✅
  (目标: ≥20% | 达成: 是)

🎉 FPN 相比滑窗快 30.35%

================================================================================
```

### 预期结果分析

根据设计预期：

| 情况 | 速度提升 | 内存节省 | 匹配数 | 判断 |
|------|---------|---------|--------|------|
| ✅ 最佳 | ≥30% | ≥20% | 相当/更优 | FPN 完全优于滑窗 |
| ✅ 良好 | 20-30% | 15-20% | 相当/更优 | FPN 显著优于滑窗 |
| ⚠️ 可接受 | 10-20% | 5-15% | 相当 | FPN 略优，需验证 |
| ❌ 需改进 | <10% | <5% | 下降 | 需要优化 FPN |

---

## 分析与建议

### 1. 性能原因分析

#### FPN 优势

- **多尺度特征复用**: 单次前向传播提取所有尺度，避免重复计算
- **显存效率**: 特征金字塔共享骨干网络的显存占用
- **推理时间**: 避免多次图像缩放和前向传播

#### 滑窗劣势

- **重复计算**: 多个 stride 下重复特征提取
- **显存压力**: 窗口缓存和中间特征占用
- **I/O 开销**: 图像缩放和逐窗口处理

### 2. 优化建议

**如果 FPN 性能未达预期**:

1. **检查模型配置**
   ```yaml
   # configs/base_config.yaml
   model:
     fpn:
       out_channels: 256  # 尝试降低至 128
       norm: "bn"         # 尝试 "gn" 或 "none"
   ```

2. **优化关键点提取**
   ```yaml
   matching:
     keypoint_threshold: 0.5  # 调整阈值
     nms:
       radius: 4  # 调整 NMS 半径
   ```

3. **批量处理优化**
   - 使用更大的 batch size（如果显存允许）
   - 启用 GPU 预热和同步

4. **代码优化**
   - 减少 Python 循环，使用向量化操作
   - 使用 torch.jit.script 编译关键函数

### 3. 后续测试步骤

1. **多数据集测试**
   - 测试多张不同尺寸的版图
   - 验证性能的稳定性

2. **精度验证**
   ```bash
   # 对比 FPN vs 滑窗的检测结果
   # 确保关键点和匹配内点相当或更优
   ```

3. **混合模式测试**
   - 小图像：考虑单尺度推理
   - 大图像：使用 FPN 路径

4. **实际应用验证**
   - 在真实版图上测试
   - 验证检测精度和召回率

---

## 使用指南

### 快速开始

#### 1. 准备测试数据

```bash
# 创建测试目录
mkdir -p test_data

# 放置版图和模板（需要自己准备）
# test_data/layout.png
# test_data/template.png
```

#### 2. 运行测试

```bash
# 5 次运行，输出 JSON 结果
uv run python tests/benchmark_fpn.py \
  --layout test_data/layout.png \
  --template test_data/template.png \
  --num-runs 5 \
  --output results/benchmark_fpn.json
```

#### 3. 查看结果

```bash
# JSON 格式结果
cat results/benchmark_fpn.json | python -m json.tool

# 手动解析 JSON
python -c "
import json
with open('results/benchmark_fpn.json') as f:
    data = json.load(f)
    comparison = data['comparison']
    print(f\"Speed: {comparison['speedup_percent']:.2f}%\")
    print(f\"Memory: {comparison['memory_saving_percent']:.2f}%\")
"
```

### 高级用法

#### 1. 多组测试对比

```bash
# 测试不同配置
for nms_radius in 2 4 8; do
  uv run python tests/benchmark_fpn.py \
    --layout test_data/layout.png \
    --template test_data/template.png \
    --output results/benchmark_nms_${nms_radius}.json
done
```

#### 2. CPU vs GPU 对比

```bash
# GPU 测试
uv run python tests/benchmark_fpn.py \
  --layout test_data/layout.png \
  --template test_data/template.png \
  --device cuda \
  --output results/benchmark_gpu.json

# CPU 测试
uv run python tests/benchmark_fpn.py \
  --layout test_data/layout.png \
  --template test_data/template.png \
  --device cpu \
  --output results/benchmark_cpu.json
```

#### 3. 详细日志输出

```bash
# 添加调试输出（需要修改脚本）
# 测试脚本会打印每次运行的详细信息
uv run python tests/benchmark_fpn.py \
  --layout test_data/layout.png \
  --template test_data/template.png \
  --num-runs 5 \
  --output results/benchmark.json 2>&1 | tee benchmark.log
```

### 常见问题

#### Q1: 测试失败，提示 "找不到模型"

```bash
# 检查模型路径
ls -la path/to/save/model_final.pth

# 指定模型路径
uv run python tests/benchmark_fpn.py \
  --model_path /absolute/path/to/model.pth \
  --layout test_data/layout.png \
  --template test_data/template.png
```

#### Q2: GPU 内存不足

```bash
# 使用较小的图像测试
uv run python tests/benchmark_fpn.py \
  --layout test_data/layout_small.png \
  --template test_data/template_small.png

# 或使用 CPU
uv run python tests/benchmark_fpn.py \
  --layout test_data/layout.png \
  --template test_data/template.png \
  --device cpu
```

#### Q3: 性能数据波动大

```bash
# 增加运行次数取平均
uv run python tests/benchmark_fpn.py \
  --layout test_data/layout.png \
  --template test_data/template.png \
  --num-runs 10  # 从 5 增加到 10
```

---

## 附录

### A. 脚本接口

```python
# 编程调用
from tests.benchmark_fpn import benchmark_fpn, benchmark_sliding_window
from models.rord import RoRD
from utils.data_utils import get_transform
from PIL import Image
import torch

model = RoRD().cuda()
model.load_state_dict(torch.load("path/to/model.pth"))
model.eval()

layout_img = Image.open("layout.png").convert('L')
template_img = Image.open("template.png").convert('L')
transform = get_transform()

# 获取 YAML 配置
from utils.config_loader import load_config
cfg = load_config("configs/base_config.yaml")

# 测试 FPN
fpn_result = benchmark_fpn(
    model, layout_img, template_img, transform, 
    cfg.matching, num_runs=5
)

print(f"FPN 平均时间: {fpn_result['mean_time_ms']:.2f}ms")
```

### B. 导出 TensorBoard 数据

配合导出工具 `tools/export_tb_summary.py` 导出训练日志：

```bash
# 导出 TensorBoard 标量数据
uv run python tools/export_tb_summary.py \
  --log-dir runs/train/baseline \
  --output-format csv \
  --output-file export_train_metrics.csv
```

### C. 参考资源

- [PyTorch 性能优化](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [TensorBoard 文档](https://www.tensorflow.org/tensorboard/get_started)
- [FPN 论文](https://arxiv.org/abs/1612.03144)

---

## 📝 更新日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2025-10-20 | v1.0 | 初始版本：完整的 FPN vs 滑窗性能对标文档 |

---

## ✅ 验收清单

性能基准测试已完成以下内容：

- [x] 创建 `tests/benchmark_fpn.py` 测试脚本
  - [x] FPN 性能测试函数
  - [x] 滑窗性能测试函数
  - [x] 性能对标计算
  - [x] JSON 结果输出

- [x] 创建性能基准测试报告（本文档）
  - [x] 测试方法和流程
  - [x] 性能指标说明
  - [x] 对标结果分析
  - [x] 优化建议

- [x] 支持多种配置和参数
  - [x] CLI 参数灵活配置
  - [x] 支持 CPU/GPU 切换
  - [x] 支持自定义模型路径

- [x] 完整的文档和示例
  - [x] 快速开始指南
  - [x] 高级用法示例
  - [x] 常见问题解答

---

🎉 **性能基准测试工具已就绪！**

下一步：准备测试数据，运行测试，并根据结果优化模型配置。

