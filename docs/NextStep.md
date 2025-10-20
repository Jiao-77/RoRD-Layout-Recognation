# 下一步工作计划 (NextStep)

**最后更新**: 2025-10-20  
**范围**: 仅聚焦于 `feature_work.md` 的第二部分「模型架构 (Model Architecture)」的落地执行计划  
**上下文**: 核心功能已完成，本文档将模型架构优化转化为可执行的工程计划，便于直接实施与验收。

> 参考来源：`docs/feature_work.md` 第二部分；更宏观的阶段规划见 `docs/todos/`

---

## 🔴 模型架构优化（Feature Work 第二部分）

目标：在保证现有精度的前提下，提升特征提取效率与推理速度；为后续注意力机制与多尺度策略提供更强的特征基础。

### 总体验收标准（全局）
- [ ] 训练/验证流程在新骨干和注意力方案下均可跑通，无崩溃/NaN。
- [ ] 在代表性验证集上，最终指标（IoU/mAP）不低于当前 VGG-16 基线；若下降需给出改进措施或回滚建议。
- [ ] 推理时延或显存占用至少一种维度优于基线，或达到“相当 + 结构可扩展”的工程收益。
- [ ] 关键改动均通过配置开关控制，可随时回退。

---

## 2.1 实验更现代的骨干网络（Backbone）

优先级：🟠 中  |  预计工期：~1 周  |  产出：可切换的 backbone 实现 + 对照报告

### 设计要点（小合约）
- 输入：与现有 `RoRD` 一致的图像张量 B×C×H×W。
- 输出：供检测头/描述子头使用的中高层特征张量；通道数因骨干不同而异（VGG:512、ResNet34:512、Eff-B0:1280）。
- 约束：不改变下游头部的接口形状（头部输入通道需根据骨干进行对齐适配）。
- 失败模式：通道不匹配/梯度不通/预训练权重未正确加载/收敛缓慢。

### 配置扩展（YAML）
在 `configs/base_config.yaml` 增加（或确认存在）：

```yaml
model:
	backbone:
		name: "vgg16"   # 可选：vgg16 | resnet34 | efficientnet_b0
		pretrained: true
		# 用于选择抽取的特征层（按不同骨干约定名称）
		feature_layers:
			vgg16: ["relu3_3", "relu4_3"]
			resnet34: ["layer3", "layer4"]
			efficientnet_b0: ["features_5", "features_7"]
```

### 代码改动建议
- 文件：`models/rord.py`
	1) 在 `__init__` 中根据 `cfg.model.backbone.name` 动态构建骨干：
		 - vgg16（现状保持）
		 - resnet34：从 `torchvision.models.resnet34(weights=IMAGENET1K_V1)` 构建；保存 `layer3/layer4` 输出。
		 - efficientnet_b0：从 `torchvision.models.efficientnet_b0(weights=IMAGENET1K_V1)` 构建；保存末两段 `features` 输出。
	2) 为不同骨干提供统一的“中间层特征导出”接口（注册 forward hook 或显式调用子模块）。
	3) 依据所选骨干的输出通道，调整检测头与描述子头的输入通道（如使用 1×1 conv 过渡层以解耦通道差异）。
	4) 保持现有前向签名与返回数据结构不变（训练/推理兼容）。

### 进展更新（2025-10-20）
- 已完成：在 `models/rord.py` 集成多骨干选择（`vgg16`/`resnet34`/`efficientnet_b0`），并实现统一的中间层抽取函数 `_extract_c234`（可后续重构为 `build_backbone`/`extract_features` 明确接口）。
- 已完成：FPN 通用化，基于 C2/C3/C4 构建 P2/P3/P4，按骨干返回正确的 stride。
- 已完成：单图前向 Smoke Test（三种骨干，单尺度与 FPN）均通过。
- 已完成：CPU 环境 A/B 基准（单尺度 vs FPN）见 `docs/description/Performance_Benchmark.md`。
- 待完成：GPU 环境基准（速度/显存）、基于真实数据的精度评估与收敛曲线对比。

### 落地步骤（Checklist）
- [x] 在 `models/rord.py` 增加/落地骨干构建与中间层抽取逻辑（当前通过 `_extract_c234` 实现）。
- [x] 接入 ResNet-34：返回等价中高层特征（layer2/3/4，通道≈128/256/512）。
- [x] 接入 EfficientNet-B0：返回 `features[2]/[3]/[6]`（约 24/40/192），FPN 以 1×1 横向连接对齐到 `fpn_out_channels`。
- [x] 头部适配：单尺度头使用骨干高层通道数；FPN 头统一使用 `fpn_out_channels`。
- [ ] 预训练权重：支持 `pretrained=true` 加载；补充权重加载摘要打印（哪些层未命中）。
- [x] 单图 smoke test：前向通过、无 NaN（三种骨干，单尺度与 FPN）。

### 评测与选择（A/B 实验）
- [ ] 在固定数据与超参下，比较 vgg16/resnet34/efficientnet_b0：
	- 收敛速度（loss 曲线 0-5 epoch）
	- 推理速度（ms / 2048×2048）与显存（GB）[CPU 初步结果已产出，GPU 待复测；见 `docs/description/Performance_Benchmark.md`]
	- 验证集 IoU/mAP（真实数据集待跑）
- [ ] 形成表格与可视化图，给出选择结论与原因（CPU 版初稿已在报告中给出观察）。
- [ ] 若新骨干在任一关键指标明显受损，则暂缓替换，仅保留为可切换实验选项。

### 验收标准（2.1）
- [ ] 三种骨干方案均可训练与推理（当前仅验证推理，训练与收敛待验证）；
- [ ] 最终入选骨干在 IoU/mAP 不低于 VGG 的前提下，带来显著的速度/显存优势之一；
- [x] 切换完全配置化（无需改代码）。

### 风险与回滚（2.1）
- 通道不匹配导致维度错误 → 在进入头部前统一使用 1×1 conv 适配；
- 预训练权重与自定义层名不一致 → 显式映射并记录未加载层；
- 收敛变慢 → 暂时提高训练轮数、调学习率/BN 冻结策略；不达标即回滚 `backbone.name=vgg16`。

---

## 2.2 集成注意力机制（CBAM / SE-Net）

优先级：🟠 中  |  预计工期：~7–10 天  |  产出：注意力增强的 RoRD 变体 + 对照报告

### 模块选择与嵌入位置
- 方案 A：CBAM（通道注意 + 空间注意），插入至骨干高层与两类头部之前；
- 方案 B：SE-Net（通道注意），轻量但仅通道维，插入多个阶段以增强稳定性；
- 建议：先实现 CBAM，保留 SE 作为备选开关。

### 配置扩展（YAML）
```yaml
model:
	attention:
		enabled: true
		type: "cbam"   # 可选：cbam | se | none
		places: ["backbone_high", "det_head", "desc_head"]
		# 可选超参：reduction、kernel_size 等
		reduction: 16
		spatial_kernel: 7
```

### 代码改动建议
- 文件：`models/rord.py`
	1) 实现 `CBAM` 与 `SEBlock` 模块（或从可靠实现迁移），提供简洁 forward。
	2) 在 `__init__` 中依据 `cfg.model.attention` 决定在何处插入：
		 - backbone 高层输出后（增强高层语义的判别性）；
		 - 检测头、描述子头输入前（分别强化不同任务所需特征）。
	3) 注意保持张量尺寸不变；若引入残差结构，保证与原路径等价时可退化为恒等映射。

### 落地步骤（Checklist）
- [ ] 实现 `CBAM`：通道注意（MLP/Avg+Max Pool）+ 空间注意（7×7 conv）。
- [ ] 实现 `SEBlock`：Squeeze（全局池化）+ Excitation（MLP, reduction）。
- [ ] 在 `RoRD` 中用配置化开关插拔注意力，默认关闭。
- [ ] 在进入检测/描述子头前分别测试开启/关闭注意力的影响。
- [ ] 记录注意力图（可选）：导出中间注意图用于可视化对比。

### 训练与评估
- [ ] 以入选骨干为基线，分别开启 `cbam` 与 `se` 进行对照；
- [ ] 记录：训练损失、验证 IoU/mAP、推理时延/显存；
- [ ] 观察注意力图是否集中在关键几何（边角/交点/突变）；
- [ ] 若带来过拟合迹象（验证下降），尝试减弱注意力强度或减少插入位置。

### 验收标准（2.2）
- [ ] 模型在开启注意力后稳定训练，无数值异常；
- [ ] 指标不低于无注意力基线；若提升则量化收益；
- [ ] 配置可一键关闭以回退。

### 风险与回滚（2.2）
- 注意力导致过拟合或梯度不稳 → 降低 reduction、减少插入点、启用正则；
- 推理时延上升明显 → 对注意力路径进行轻量化（如仅通道注意或更小 kernel）。

---

## 工程与度量配套

### 实验记录（建议）
- 在 TensorBoard 中新增：
	- `arch/backbone_name`、`arch/attention_type`（Text/Scalar）；
	- `train/loss_total`、`eval/iou_metric`、`eval/map`；
	- 推理指标：`infer/ms_per_image`、`infer/vram_gb`。

### 对照报告模板（最小集）
- 数据集与配置摘要（随机种子、批大小、学习率、图像尺寸）。
- 三个骨干 + 注意力开关的结果表（速度/显存/IoU/mAP）。
- 结论与落地选择（保留/关闭/待进一步实验）。

---

## 排期与里程碑（建议）
- M1（1 天）：骨干切换基础设施与通道适配层；单图 smoke 测试。
- M2（2–3 天）：ResNet34 与 EfficientNet-B0 接入与跑通；
- M3（1–2 天）：A/B 评测与结论；
- M4（3–4 天）：注意力模块接入、训练对照、报告输出。

---

## 相关参考
- 源文档：`docs/feature_work.md` 第二部分（模型架构）
- 阶段规划：`docs/todos/`
- 配置系统：`configs/base_config.yaml`

