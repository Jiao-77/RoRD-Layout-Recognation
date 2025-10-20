## 一、数据策略与增强 (Data Strategy & Augmentation)

> 目标：提升模型的鲁棒性和泛化能力，减少对大量真实数据的依赖。

- [x] 引入弹性变形 (Elastic Transformations)
	- ✔️ 价值：模拟芯片制造中可能出现的微小物理形变，使模型对非刚性变化更鲁棒。
	- 🧭 关键原则（与当前数据管线一致）：
		- 现有自监督训练数据集 `ICLayoutTrainingDataset` 会返回 (original, rotated, H)；其中 H 是两张 patch 间的单应关系，用于 loss 监督。
		- 非刚性弹性变形若只对其中一张或在生成 H 之后施加，会破坏几何约束，导致 H 失效。
		- 因此，Elastic 需在“生成 homography 配对之前”对基础 patch 施加；随后对该已变形的 patch 再执行旋转/镜像与单应计算，这样 H 仍严格成立。
	- 📝 执行计划：
		1) 依赖核对
			 - `pyproject.toml` 已包含 `albumentations>=2.0.8`，无需新增依赖；确保环境安装齐全。
		2) 集成位置与方式
			 - 在 `data/ic_dataset.py` 的 `ICLayoutTrainingDataset.__getitem__` 中，裁剪并缩放得到 `patch` 后，转换为 `np.ndarray`，对其调用 `albumentations` 管道（包含 `A.ElasticTransform`）。
			 - 将变形后的 `patch_np_uint8` 作为“基准图”，再按现有逻辑计算旋转/镜像与 `homography`，生成 `transformed_patch`，从而确保 H 有效。
		3) 代码改动清单（建议）
			 - `data/ic_dataset.py`
				 - 顶部新增：`import albumentations as A`
				 - `__init__` 新增可选参数：`use_albu: bool=False`、`albu_params: dict|None=None`
				 - 在 `__init__` 构造 `self.albu = A.Compose([...])`（当 `use_albu` 为 True 时），包含：
					 - `A.ElasticTransform(alpha=40, sigma=6, alpha_affine=6, p=0.3)`
					 - （可选）`A.RandomBrightnessContrast(p=0.5)`、`A.GaussNoise(var_limit=(5.0, 20.0), p=0.3)` 以替代当前手写的亮度/对比度与噪声逻辑（减少重复）。
				 - 在 `__getitem__`：裁剪与缩放后，若启用 `self.albu`：`patch_np_uint8 = self.albu(image=patch_np_uint8)["image"]`，随后再计算旋转/镜像与 `homography`。
				 - 注意：保持输出张量与当前 `utils.data_utils.get_transform()` 兼容（单通道→三通道→Normalize）。
			 - `configs/base_config.yaml`
				 - 新增配置段：
					 - `augment.elastic.enabled: true|false`
					 - `augment.elastic.alpha: 40`
					 - `augment.elastic.sigma: 6`
					 - `augment.elastic.alpha_affine: 6`
					 - `augment.elastic.prob: 0.3`
					 - （可选）`augment.photometric.*` 开关与参数
			 - `train.py`
				 - 从配置读取上述参数，并将 `use_albu` 与 `albu_params` 通过 `ICLayoutTrainingDataset(...)` 传入（不影响现有 `get_transform()`）。
		4) 参数与默认值建议
			 - 起始：`alpha=40, sigma=6, alpha_affine=6, p=0.3`；根据训练收敛与可视化效果微调。
			 - 若发现描述子对局部形变敏感，可逐步提高 `alpha` 或 `p`；若训练不稳定则降低。
		5) 验证与可视化
			 - 在 `tests/benchmark_grid.py` 或新增简单可视化脚本中，采样 16 个 (original, rotated) 对，叠加可视化 H 变换后的网格，确认几何一致性未破坏。
			 - 训练前 1000 个 batch：记录 `loss_det/loss_desc` 曲线，确认未出现异常发散。

- [x] 创建合成版图数据生成器
	- ✔️ 价值：解决真实版图数据获取难、数量少的问题，通过程序化生成大量多样化的训练样本。
	- 📝 执行计划：
		1) 新增脚本 `tools/generate_synthetic_layouts.py`
			 - 目标：使用 `gdstk` 程序化生成包含不同尺寸、密度与单元类型的 GDSII 文件。
			 - 主要能力：
				 - 随机生成“标准单元”模版（如若干矩形/多边形组合）、金属走线、过孔阵列；
				 - 支持多层（layer/datatype）与规则化阵列（row/col pitch）、占空比（density）控制；
				 - 形状参数与布局由随机种子控制，支持可重复性。
			 - CLI 设计（示例）：
				 - `--out-dir data/synthetic/gds`、`--num-samples 1000`、`--seed 42`
				 - 版图规格：`--width 200um --height 200um --grid 0.1um`
				 - 多样性开关：`--cell-types NAND,NOR,INV --metal-layers 3 --density 0.1-0.6`
			 - 关键实现要点：
				 - 使用 `gdstk.Library()` 与 `gdstk.Cell()` 组装基本单元；
				 - 通过 `gdstk.Reference` 和阵列生成放置；
				 - 生成完成后 `library.write_gds(path)` 落盘。
		2) 批量转换 GDSII → PNG（训练用）
			 - 现状核对：仓库中暂无 `tools/layout2png.py`；计划新增该脚本（与本项一并交付）。
			 - 推荐实现 A（首选）：使用 `klayout` 的 Python API（`pya`）以无头模式加载 GDS，指定层映射与缩放，导出为高分辨率 PNG：
				 - 脚本 `tools/layout2png.py` 提供 CLI：`--in data/synthetic/gds --out data/synthetic/png --dpi 600 --layers 1/0:gray,2/0:blue ...`
				 - 支持目录批量与单文件转换；可配置画布背景、线宽、边距。
			 - 替代实现 B：导出 SVG 再用 `cairosvg` 转 PNG（依赖已在项目中），适合无 klayout 环境的场景。
			 - 输出命名规范：与 GDS 同名，如 `chip_000123.gds → chip_000123.png`。
		3) 数据目录与元数据
			 - 目录结构建议：
				 - `data/synthetic/gds/`、`data/synthetic/png/`、`data/synthetic/meta/`
			 - 可选：为每个样本生成 `meta/*.json`，记录层数、单元类型分布、密度等，用于后续分析/分层采样。
		4) 与训练集集成
			 - `configs/base_config.yaml` 新增：
				 - `paths.synthetic_dir: data/synthetic/png`
				 - `training.use_synthetic_ratio: 0.0~1.0`（混合采样比例；例如 0.3 表示 30% 合成样本）
			 - 在 `train.py` 中：
				 - 若 `use_synthetic_ratio>0`，构建一个 `ICLayoutTrainingDataset` 指向合成 PNG 目录；
				 - 实现简单的比例采样器或 `ConcatDataset + WeightedRandomSampler` 以按比例混合真实与合成样本。
		5) 质量与稳健性检查
			 - 可视化抽样：随机展示若干 PNG，检查层次颜色、对比度、线宽是否清晰；
			 - 分布对齐：统计真实数据与合成数据的连线长度分布、拓扑度量（如节点度、环路数量），做基础分布对齐；
			 - 训练烟雾测试：仅用 100～200 个合成样本跑 1～2 个 epoch，确认训练闭环无错误、loss 正常下降。
		6) 基准验证与复盘
			 - 在 `tests/benchmark_grid.py` 与 `tests/benchmark_backbones.py` 增加一组“仅真实 / 真实+合成”的对照实验；
			 - 记录 mAP/匹配召回/描述子一致性等指标，评估增益；
			 - 产出 `docs/Performance_Benchmark.md` 的对比表格。

### 验收标准 (Acceptance Criteria)

- Elastic 变形：
	- [ ] 训练数据可视化（含 H 网格叠加）无几何错位；
	- [ ] 训练前若干 step loss 无异常尖峰，长期收敛不劣于 baseline；
	- [ ] 可通过配置无缝开/关与调参。
- 合成数据：
	- [ ] 能批量生成带多层元素的 GDS 文件并成功转为 PNG；
	- [ ] 训练脚本可按设定比例混合采样真实与合成样本；
	- [ ] 在小规模对照实验中，验证指标有稳定或可解释的变化（不劣化）。

### 风险与规避 (Risks & Mitigations)

- 非刚性变形破坏 H 的风险：仅在生成 homography 前对基准 patch 施加 Elastic，或在两图上施加相同变形但更新 H′=f∘H∘f⁻¹（当前计划采用前者，简单且稳定）。
- GDS → PNG 渲染差异：优先使用 `klayout`，保持工业级渲染一致性；无 `klayout` 时使用 SVG→PNG 备选路径。
- 合成分布与真实分布不匹配：通过密度与单元类型分布约束进行对齐，并在训练中控制混合比例渐进提升。

### 里程碑与时间估算 (Milestones & ETA)

## 二、实现状态与使用说明（2025-10-20 更新）

- Elastic 变形已按计划集成：
	- 开关与参数：见 `configs/base_config.yaml` 下的 `augment.elastic` 与 `augment.photometric`；
	- 数据集实现：`data/ic_dataset.py` 中 `ICLayoutTrainingDataset`；
	- 可视化验证：`tools/preview_dataset.py --dir <png_dir> --n 8 --elastic`。

- 合成数据生成与渲染：
	- 生成 GDS：`tools/generate_synthetic_layouts.py --out-dir data/synthetic/gds --num 100 --seed 42`；
	- 转换 PNG：`tools/layout2png.py --in data/synthetic/gds --out data/synthetic/png --dpi 600`；
	- 训练混采：在 `configs/base_config.yaml` 设置 `synthetic.enabled: true`、`synthetic.png_dir: data/synthetic/png`、`synthetic.ratio: 0.3`。

- 训练脚本：
	- `train.py` 已接入真实/合成混采（ConcatDataset + WeightedRandomSampler），验证集仅用真实数据；
	- TensorBoard 文本摘要记录数据构成（mix 开关、比例、样本量）。

注意：若未安装 KLayout，可自动回退 gdstk+SVG 路径；显示效果可能与 KLayout 存在差异。

- D1：Elastic 集成 + 可视化验证（代码改动与测试）
- D2：合成生成器初版（GDS 生成 + PNG 渲染脚本）
- D3：训练混合采样接入 + 小规模基准
- D4：参数扫与报告更新（Performance_Benchmark.md）

### 一键流水线（生成 → 渲染 → 预览 → 训练）

1) 生成 GDS（合成版图）
```bash
uv run python tools/generate_synthetic_layouts.py --out_dir data/synthetic/gds --num 200 --seed 42
```

2) 渲染 PNG（KLayout 优先，自动回退 gdstk+SVG）
```bash
uv run python tools/layout2png.py --in data/synthetic/gds --out data/synthetic/png --dpi 600
```

3) 预览训练对（核验增强/H 一致性）
```bash
uv run python tools/preview_dataset.py --dir data/synthetic/png --out preview.png --n 8 --elastic
```

4) 在 YAML 中开启混采与 Elastic（示例）
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

5) 开始训练
```bash
uv run python train.py --config configs/base_config.yaml
```

可选：使用单脚本一键执行（含配置写回）
```bash
uv run python tools/synth_pipeline.py --out_root data/synthetic --num 200 --dpi 600 \
	--config configs/base_config.yaml --ratio 0.3 --enable_elastic
```

### 参数建议与经验

- 渲染 DPI：600–900 通常足够，图形极细时可提高到 1200（注意磁盘与 IO）。
- 混采比例 synthetic.ratio：
	- 数据少（<500 张）可取 0.3–0.5；
	- 数据中等（500–2000 张）建议 0.2–0.3；
	- 数据多（>2000 张）建议 0.1–0.2 以免分布偏移。
- Elastic 强度：从 alpha=40, sigma=6 开始；若描述子对局部形变敏感，可小步上调 alpha 或 prob。

### 质量检查清单（建议在首次跑通后执行）

- 预览拼图无明显几何错位（orig/rot 对应边界对齐合理）。
- 训练日志包含混采信息（real/syn 样本量、ratio、启停状态）。
- 若开启 Elastic，训练初期 loss 无异常尖峰，长期收敛不劣于 baseline。
- 渲染 PNG 与 GDS 在关键层上形态一致（优先使用 KLayout）。

### 常见问题与排查（FAQ）

- klayout: command not found
	- 方案A：安装系统级 KLayout 并确保可执行文件在 PATH；
	- 方案B：暂用 gdstk+SVG 回退（外观可能略有差异）。
- cairosvg 报错或 SVG 不生成
	- 升级 `cairosvg` 与 `gdstk`；确保磁盘有写入权限；检查 `.svg` 是否被安全软件拦截。
- gdstk 版本缺少 write_svg
	- 尝试升级 gdstk；脚本已做 library 与 cell 双路径兼容，仍失败则优先使用 KLayout。
- 训练集为空或样本过少
	- 检查 `paths.layout_dir` 与 `synthetic.png_dir` 是否存在且包含 .png；ratio>0 但 syn 目录为空会自动回退仅真实数据。

