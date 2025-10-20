# 测试修改说明 — RoRD 多骨干 FPN 支持与基准脚本

最后更新：2025-10-20
作者：项目自动化助手

## 概述
本次修改面向「模型架构（Backbone 与 FPN）」的工程化完善，目标是在不破坏现有接口的前提下，支持更现代的骨干网络，并提供可复现的基准测试脚本。

包含内容：
- 修复并重构 `models/rord.py` 的初始化与 FPN 逻辑，支持三种骨干：`vgg16`、`resnet34`、`efficientnet_b0`。
- 新增 A/B 基准脚本 `tests/benchmark_backbones.py`，比较不同骨干在单尺度与 FPN 前向的耗时与显存占用。
- 为 FPN 输出添加「真实下采样步幅（stride）」标注，避免坐标还原误差。

兼容性：
- 公共接口未变，`RoRD` 的前向签名保持不变（`return_pyramid` 开关控制是否走 FPN）。
- 默认配置仍为 `vgg16`，单尺度路径保持与原基线一致（处理到 relu4_3，stride≈8）。

## 代码变更
- `models/rord.py`
  - 修复：配置解析、骨干构建、FPN 模块初始化的缩进与作用域问题。
  - 新增：按骨干类型提取中间层 C2/C3/C4（VGG: relu2_2/3_3/4_3；ResNet34: layer2/3/4；Eff-B0: features[2]/[3]/[6]）。
  - 新增：FPN 输出携带每层 stride（相对输入）。
  - 注意：非 VGG 场景下不再访问 `self.features`（避免未定义错误）。
- `tests/benchmark_backbones.py`
  - 新增：单文件基准工具，可在相同输入下对比三种骨干在单尺度与 FPN 的推理耗时（ms）与显存占用（MB）。
- `configs/base_config.yaml`
  - 已存在/确认字段：
    - `model.backbone.name`: vgg16 | resnet34 | efficientnet_b0
    - `model.backbone.pretrained`: true/false
    - `model.attention`（默认关闭，可选 `cbam`/`se`）

## FPN 下采样步幅说明（按骨干）
- vgg16：P2/P3/P4 对应 stride ≈ 2 / 4 / 8
- resnet34：P2/P3/P4 对应 stride ≈ 8 / 16 / 32
- efficientnet_b0：P2/P3/P4 对应 stride ≈ 4 / 8 / 32

说明：stride 用于将特征图坐标映射回原图坐标，`match.py` 中的坐标还原与 NMS 逻辑可直接使用返回的 stride 值。

## 快速验证（Smoke Test）
以下为在 1×3×256×256 随机张量上前向的形状验证（节选）：
- vgg16 单尺度：det [1, 1, 32, 32]，desc [1, 128, 32, 32]
- vgg16 FPN：
  - P4: [1, 1, 32, 32]（stride 8）
  - P3: [1, 1, 64, 64]（stride 4）
  - P2: [1, 1, 128, 128]（stride 2）
- resnet34 FPN：
  - P4: [1, 1, 8, 8]（stride 32）
  - P3: [1, 1, 16, 16]（stride 16）
  - P2: [1, 1, 32, 32]（stride 8）
- efficientnet_b0 FPN：
  - P4: [1, 1, 8, 8]（stride 32）
  - P3: [1, 1, 32, 32]（stride 8）
  - P2: [1, 1, 64, 64]（stride 4）

以上输出与各骨干的下采样规律一致，说明中间层选择与 FPN 融合逻辑正确。

## 如何运行基准测试
- 环境准备（一次性）：已在项目 `pyproject.toml` 中声明依赖（含 `torch`、`torchvision`、`psutil`）。
- 骨干 A/B 基准：
  - CPU 示例：
    ```zsh
    uv run python tests/benchmark_backbones.py --device cpu --image-size 512 --runs 5
    ```
  - CUDA 示例：
    ```zsh
    uv run python tests/benchmark_backbones.py --device cuda --runs 20 --backbones vgg16 resnet34 efficientnet_b0
    ```
- FPN vs 滑窗对标（需版图/模板与模型权重）：
  ```zsh
  uv run python tests/benchmark_fpn.py \
    --layout /path/to/layout.png \
    --template /path/to/template.png \
    --num-runs 5 \
    --config configs/base_config.yaml \
    --model_path /path/to/weights.pth \
    --device cuda
  ```

## 影响评估与回滚
- 影响范围：
  - 推理路径：单尺度不变；FPN 路径新增多骨干支持与 stride 标注。
  - 训练/评估：头部输入通道通过 1×1 适配（内部已处理），无需额外修改。
- 回滚策略：
  - 将 `model.backbone.name` 设回 `vgg16`，或在推理时设置 `return_pyramid=False` 走单尺度路径。

## 后续建议
- EfficientNet 中间层可进一步调研（如 features[3]/[4]/[6] 组合）以兼顾精度与速度。
- 增补单元测试：对三种骨干的 P2/P3/P4 输出形状和 stride 进行断言（CPU 可运行，避免依赖数据集）。
- 将 A/B 基准结果沉淀至 `docs/Performance_Benchmark.md`，用于跟踪优化趋势。
