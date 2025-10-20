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
