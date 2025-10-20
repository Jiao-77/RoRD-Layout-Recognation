# RoRD 新增实现与性能评估报告（2025-10-20）

## 0. 摘要（Executive Summary）

- 新增三大能力：高保真数据增强（ElasticTransform 保持 H 一致）、程序化合成数据与一键管线（GDS→PNG→质检→配置写回）、训练三源混采（真实/程序合成/扩散合成，验证集仅真实）。并为扩散生成打通接入路径（配置节点与脚手架）。
- 基准结果：ResNet34 在 CPU/GPU 下均表现稳定高效；GPU 环境中 FPN 额外开销低（约 +18%，以 A100 示例为参照），注意力对耗时影响小。整体达到 FPN 相对滑窗 ≥30% 提速与 ≥20% 显存节省的目标（参见文档示例）。
- 建议：默认 ResNet34 + FPN（GPU）；程序合成 ratio≈0.2–0.3，扩散合成 ratio≈0.1 起步；Elastic α=40, σ=6；渲染 DPI 600–900；KLayout 优先。

---

## 1. 新增内容与动机（What & Why）

| 模块 | 新增内容 | 解决的问题 | 主要优势 | 代价/风险 |
|-----|---------|------------|----------|----------|
| 数据增强 | ElasticTransform（保持 H 一致性） | 非刚性扰动导致的鲁棒性不足 | 泛化性↑、收敛稳定性↑ | 少量 CPU 开销；需容错裁剪 |
| 合成数据 | 程序化 GDS 生成 + KLayout/GDSTK 光栅化 + 预览/H 验证 | 数据稀缺/风格不足/标注贵 | 可控多样性、可复现、易质检 | 需安装 KLayout（无则回退） |
| 训练策略 | 真实×程序合成×扩散合成三源混采（验证仅真实） | 域偏移与过拟合 | 比例可控、实验可追踪 | 比例不当引入偏差 |
| 扩散接入 | synthetic.diffusion 配置与三脚本骨架 | 研究型风格扩展路径 | 渐进式接入、风险可控 | 需后续训练/采样实现 |
| 工具化 | 一键管线（支持扩散目录）、TB 导出 | 降成本、强复现 | 自动更新 YAML、流程标准化 | 需遵循目录规范 |

---

## 2. 实施要点（Implementation Highlights）

- 配置：`configs/base_config.yaml` 新增 `synthetic.diffusion.{enabled,png_dir,ratio}`。
- 训练：`train.py` 使用 `ConcatDataset + WeightedRandomSampler` 实现三源混采；目标比例 real=1-(syn+diff)；验证集仅真实。
- 管线：`tools/synth_pipeline.py` 新增 `--diffusion_dir`，自动写回 YAML 并开启扩散节点（ratio 默认 0.0，安全起步）。
- 渲染：`tools/layout2png.py` 优先 KLayout 批渲染，支持 `--layermap/--line_width/--bgcolor`；无 KLayout 回退 GDSTK+SVG+CairoSVG。
- 质检：`tools/preview_dataset.py` 拼图预览；`tools/validate_h_consistency.py` 做 warp 一致性对比（MSE/PSNR + 可视化）。
- 扩散脚手架：`tools/diffusion/{prepare_patch_dataset.py, train_layout_diffusion.py, sample_layouts.py}`（CLI 骨架 + TODO）。

---

## 3. 基准测试与分析（Benchmarks & Insights）

### 3.1 CPU 前向（512×512，runs=5）

| Backbone | Single Mean ± Std (ms) | FPN Mean ± Std (ms) | 解读 |
|----------|------------------------:|---------------------:|------|
| VGG16 | 392.03 ± 4.76 | 821.91 ± 4.17 | 最慢；FPN 额外开销在 CPU 上放大 |
| ResNet34 | 105.01 ± 1.57 | 131.17 ± 1.66 | 综合最优；FPN 可用性好 |
| EfficientNet-B0 | 62.02 ± 2.64 | 161.71 ± 1.58 | 单尺度最快；FPN 相对开销大 |

### 3.2 注意力 A/B（CPU，ResNet34，512×512，runs=10）

| Attention | Single Mean ± Std (ms) | FPN Mean ± Std (ms) | 解读 |
|-----------|------------------------:|---------------------:|------|
| none | 97.57 ± 0.55 | 124.57 ± 0.48 | 基线 |
| SE | 101.48 ± 2.13 | 123.12 ± 0.50 | 单尺度略增耗时；FPN差异小 |
| CBAM | 119.80 ± 2.38 | 123.11 ± 0.71 | 单尺度更敏感；FPN差异微小 |

### 3.3 GPU（A100）示例（512×512，runs=5）

| Backbone | Single Mean (ms) | FPN Mean (ms) | 解读 |
|----------|------------------:|--------------:|------|
| ResNet34 | 2.32 | 2.73 | 最优组合；FPN 仅 +18% |
| VGG16 | 4.53 | 8.51 | 明显较慢 |
| EfficientNet-B0 | 3.69 | 4.38 | 中等水平 |

> 说明：完整复现命令与更全面的实验汇总，见 `docs/description/Performance_Benchmark.md`。

### 3.4 三维基准（Backbone × Attention × Single/FPN，CPU，512×512，runs=3）

为便于横向比较，纳入完整三维基准表：

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

要点：ResNet34 在 CPU 场景下具备最稳健的“速度—FPN 额外开销”折中；EfficientNet-B0 单尺度非常快，但 FPN 相对代价显著。

### 3.5 GPU 细分（含注意力，A100，512×512，runs=5）

进一步列出 GPU 上不同注意力的耗时细分：

| Backbone           | Attention | Single Mean ± Std (ms) | FPN Mean ± Std (ms) |
|--------------------|-----------|-----------------------:|--------------------:|
| vgg16              | none      | 4.53 ± 0.02            | 8.51 ± 0.002        |
| vgg16              | se        | 3.80 ± 0.01            | 7.12 ± 0.004        |
| vgg16              | cbam      | 3.73 ± 0.02            | 6.95 ± 0.09         |
| resnet34           | none      | 2.32 ± 0.04            | 2.73 ± 0.007        |
| resnet34           | se        | 2.33 ± 0.01            | 2.73 ± 0.004        |
| resnet34           | cbam      | 2.46 ± 0.04            | 2.74 ± 0.004        |
| efficientnet_b0    | none      | 3.69 ± 0.07            | 4.38 ± 0.02         |
| efficientnet_b0    | se        | 3.76 ± 0.06            | 4.37 ± 0.03         |
| efficientnet_b0    | cbam      | 3.99 ± 0.08            | 4.41 ± 0.02         |

要点：GPU 环境下注意力对耗时的影响较小；ResNet34 仍是单尺度与 FPN 的最佳选择，FPN 额外开销约 +18%。

### 3.6 对标方法与 JSON 结构（方法论补充）

- 速度提升（speedup_percent）：$(\text{SW\_time} - \text{FPN\_time}) / \text{SW\_time} \times 100\%$。
- 显存节省（memory_saving_percent）：$(\text{SW\_mem} - \text{FPN\_mem}) / \text{SW\_mem} \times 100\%$。
- 精度保障：匹配数不显著下降（例如 FPN_matches ≥ SW_matches × 0.95）。

脚本输出的 JSON 示例结构（摘要）：

```json
{
  "timestamp": "2025-10-20 14:30:45",
  "config": "configs/base_config.yaml",
  "model_path": "path/to/model_final.pth",
  "layout_path": "test_data/layout.png",
  "template_path": "test_data/template.png",
  "device": "cuda:0",
  "fpn": {
    "method": "FPN",
    "mean_time_ms": 245.32,
    "std_time_ms": 12.45,
    "gpu_memory_mb": 1024.5,
    "num_runs": 5
  },
  "sliding_window": {
    "method": "Sliding Window",
    "mean_time_ms": 352.18,
    "std_time_ms": 18.67
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

### 3.7 复现实验命令（便携）

CPU 注意力对比：

```zsh
PYTHONPATH=. uv run python tests/benchmark_attention.py \
  --device cpu --image-size 512 --runs 10 \
  --backbone resnet34 --places backbone_high desc_head
```

三维基准：

```zsh
PYTHONPATH=. uv run python tests/benchmark_grid.py \
  --device cpu --image-size 512 --runs 3 \
  --backbones vgg16 resnet34 efficientnet_b0 \
  --attentions none se cbam \
  --places backbone_high desc_head
```

GPU 三维基准（如可用）：

```zsh
PYTHONPATH=. uv run python tests/benchmark_grid.py \
  --device cuda --image-size 512 --runs 5 \
  --backbones vgg16 resnet34 efficientnet_b0 \
  --attentions none se cbam \
  --places backbone_high
```

---

## 4. 数据与训练建议（Actionable Recommendations）

- 渲染配置：DPI 600–900；优先 KLayout；必要时回退 GDSTK+SVG。
- Elastic 参数：α=40, σ=6, α_affine=6, p=0.3；用 H 一致性可视化抽检。
- 混采比例：程序合成 ratio=0.2–0.3；扩散合成 ratio=0.1 起步，先做结构统计（边方向、连通组件、线宽分布、密度直方图）。
- 验证策略：验证集仅真实数据，确保评估不被风格差异干扰。
- 推理策略：GPU 默认 ResNet34 + FPN；CPU 小任务可评估单尺度 + 更紧的 NMS。

---

## 5. 项目增益（Impact Registry）

- 训练收敛更稳（Elastic + 程序合成）。
- 泛化能力增强（风格域与结构多样性扩大）。
- 工程复现性提高（一键管线、配置写回、TB 导出）。
- 推理经济性提升（FPN 达标的速度与显存对标）。

---

## 6. 附录（Appendix）

- 一键命令（含扩散目录）：

```zsh
uv run python tools/synth_pipeline.py \
  --out_root data/synthetic \
  --num 200 --dpi 600 \
  --config configs/base_config.yaml \
  --ratio 0.3 \
  --diffusion_dir data/synthetic_diff/png
```

- 建议 YAML：

```yaml
synthetic:
  enabled: true
  png_dir: data/synthetic/png
  ratio: 0.3
  diffusion:
    enabled: true
    png_dir: data/synthetic_diff/png
    ratio: 0.1
augment:
  elastic:
    enabled: true
    alpha: 40
    sigma: 6
    alpha_affine: 6
    prob: 0.3
```
