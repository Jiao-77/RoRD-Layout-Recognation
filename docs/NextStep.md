# 本地 TensorBoard 实验追踪方案

日期：2025-09-25

## 目标
- 在本地工作站搭建一套轻量、低成本的实验追踪与可视化管道，覆盖训练、评估和模板匹配流程。
- 结合现有 YAML 配置体系，为后续扩展（自动化调参、远程同步）保留接口。

## 环境与前置准备
1. **系统与软件**
   - 操作系统：Ubuntu 22.04 / Windows 11 / macOS 14（任选其一）。
   - Python 环境：使用项目默认的 `uv` 虚拟环境（见 `uv.lock` / `pyproject.toml`）。
2. **依赖安装**
   ```bash
   uv add tensorboard tensorboardX
   ```
3. **目录规划**
   - 在项目根目录创建 `runs/`，建议按 `runs/<experiment_name>/` 组织日志。
   - 训练与评估可分别输出到 `runs/train/`、`runs/eval/` 子目录。

## 集成步骤
### 1. 配置项扩展
- 在 `configs/base_config.yaml` 中添加：
  ```yaml
  logging:
    use_tensorboard: true
    log_dir: "runs"
    experiment_name: "baseline"
  ```
- 命令行新增 `--log-dir`、`--experiment-name` 参数，默认读取配置，可在执行时覆盖。

### 2. 训练脚本 `train.py`
1. **初始化 SummaryWriter**
   ```python
   from torch.utils.tensorboard import SummaryWriter

   log_dir = Path(args.log_dir or cfg.logging.log_dir)
   experiment = args.experiment_name or cfg.logging.experiment_name
   writer = SummaryWriter(log_dir=log_dir / "train" / experiment)
   ```
2. **记录训练指标**（每个 iteration）
   ```python
   global_step = epoch * len(train_dataloader) + i
   writer.add_scalar("loss/total", loss.item(), global_step)
   writer.add_scalar("loss/det", det_loss.item(), global_step)
   writer.add_scalar("loss/desc", desc_loss.item(), global_step)
   writer.add_scalar("optimizer/lr", scheduler.optimizer.param_groups[0]['lr'], global_step)
   ```
3. **验证阶段记录**
   - Epoch 结束后写入平均损失、F1 等指标。
   - 可视化关键点热力图、匹配示意图：`writer.add_image()`。
4. **资源清理**
   - 训练结束调用 `writer.close()`。

### 3. 评估脚本 `evaluate.py`
1. 初始化 `SummaryWriter(log_dir / "eval" / experiment)`。
2. 收集所有验证样本的预测框 (boxes)、置信度 (scores) 与真实标注 (ground truth boxes)。
3. 使用 `sklearn.metrics.average_precision_score` 或 `pycocotools` 计算每个样本的 Average Precision，并汇总为 mAP：
   ```python
   from sklearn.metrics import average_precision_score
   ap = average_precision_score(y_true, y_scores)
   writer.add_scalar("eval/AP", ap, global_step)
   ```
4. 在成功匹配（IoU ≥ 阈值）后，从 `match_template_multiscale` 返回值中获取单应矩阵 `H`。
5. 使用 `cv2.decomposeHomographyMat` 或手动分解方法，将 `H` 提取为旋转角度、平移向量和缩放因子：
   ```python
   _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, np.eye(3))
   rot_angle = compute_angle(Rs[0])
   trans_vec = Ts[0]
   scale = np.linalg.norm(Ns[0])
   ```
6. 从标注文件中读取真实几何变换参数 (rotation_gt, trans_gt, scale_gt)，计算误差：
   ```python
   err_rot = abs(rot_angle - rotation_gt)
   err_trans = np.linalg.norm(trans_vec - trans_gt)
   err_scale = abs(scale - scale_gt)
   writer.add_scalar("eval/err_rot", err_rot, img_id)
   writer.add_scalar("eval/err_trans", err_trans, img_id)
   writer.add_scalar("eval/err_scale", err_scale, img_id)
   ```
7. 使用 `writer.add_histogram` 记录误差分布，并通过 `writer.add_image` 可选地上传误差直方图：
   ```python
   writer.add_histogram("eval/err_rot_hist", err_rot_list, epoch)
   ```
8. 在 TensorBoard 的 Scalars、Histograms 和 Images 面板中分别查看指标曲线、误差分布及可视化结果。

### 4. 模板匹配调试 `match.py`
- 新增参数 `--tb-log-matches`（布尔值）。
- 启用后，将关键点分布、Homography 误差统计写入 `runs/match/<experiment>/`。

## 可视化与使用
1. **启动 TensorBoard**
   ```bash
   tensorboard --logdir runs --port 6006
   ```
   - 浏览器访问 `http://localhost:6006`。
   - 若需局域网共享可加 `--bind_all`。
2. **推荐面板布局**
   - Scalars：损失曲线、学习率、评估指标。
   - Images：关键点热力图、模板匹配结果。
   - Histograms：描述子分布、梯度范数（可选）。
   - Text：记录配置摘要、Git 提交信息。

## 版本控制与组织
- 实验命名建议采用 `YYYYMMDD_project_variant`，方便检索。
- 使用 `writer.add_text()` 保存关键配置和 CLI 参数，形成自描述日志。
- 可开发 `tools/export_tb_summary.py` 导出曲线数据供文档或汇报使用。

## 进阶扩展
1. **自动化脚本**：在 `Makefile` / `tasks.json` 中增加命令，一键启动训练 + TensorBoard。
2. **远程访问**：通过 `ssh -L` 或 `ngrok` 转发端口，注意访问权限控制。
3. **对比实验**：利用 TensorBoard `Compare Runs` 功能或统一父目录对比多次实验。
4. **CI 集成**：在持续集成流程中生成日志，作为构建工件保存。

## 验证与维护
- **功能自测**：运行 1–2 个 epoch，确认日志生成并正确展示。
- **存储监控**：定期清理或压缩旧实验，避免磁盘占满。
- **备份策略**：重要实验可打包日志或同步至远程仓库。
- **团队培训**：在 README 中补充使用说明，组织示例演示。

## 下一步
- [ ] 修改配置和脚本，接入 SummaryWriter。
- [ ] 准备示例 Notebook/文档，展示 TensorBoard 面板截图。
- [ ] 后续评估是否需要接入 W&B、MLflow 等更高级平台。

---

# 推理与匹配改造计划（FPN + NMS）

日期：2025-09-25

## 目标
- 将当前的“图像金字塔 + 多次推理”的匹配流程，升级为“单次推理 + 特征金字塔 (FPN)”以显著提速。
- 在滑动窗口提取关键点后增加去重（NMS/半径抑制），降低冗余点与后续 RANSAC 的计算量。
- 保持与现有 YAML 配置、TensorBoard 记录和命令行接口的一致性；以 uv 为包管理器管理依赖和运行。

## 设计概览
- FPN：在 `models/rord.py` 中，从骨干网络多层提取特征（例如 VGG 的 relu2_2/relu3_3/relu4_3），通过横向 1x1 卷积与自顶向下上采样构建 P2/P3/P4 金字塔特征；为每个尺度共享或独立地接上检测头与描述子头，导出同维度描述子。
- 匹配路径：`match.py` 新增 FPN 路径，单次前向获得多尺度特征，逐层与模板进行匹配与几何验证；保留旧路径（图像金字塔）作为回退，通过配置开关切换。
- 去重策略：在滑窗聚合关键点后，基于“分数优先 + 半径抑制 (radius NMS)”进行去重；半径和分数阈值配置化。

## 配置变更（YAML）
在 `configs/base_config.yaml` 中新增/扩展：

```yaml
model:
   fpn:
      enabled: true            # 开启 FPN 推理
      out_channels: 256        # 金字塔特征通道数
      levels: [2, 3, 4]        # 输出层级，对应 P2/P3/P4
      norm: "bn"              # 归一化类型：bn/gn/none

matching:
   use_fpn: true              # 使用 FPN 路径；false 则沿用图像金字塔
   nms:
      enabled: true
      radius: 4                # 半径抑制像素半径
      score_threshold: 0.5     # 关键点保留的最低分数
   # 其余已有参数保留，如 ransac_reproj_threshold/min_inliers/inference_window_size...
```

注意：所有相对路径依旧使用 `utils.config_loader.to_absolute_path` 以配置文件所在目录为基准解析。

## 实施步骤

1) 基线分支与依赖
- 新开分支保存改造：
   ```bash
   git checkout -b feature/fpn-matching
   uv sync
   ```
- 目前不引入新三方库，继续使用现有 `torch/opencv/numpy`。

2) 模型侧改造（`models/rord.py`）
- 提取多层特征：在骨干网络中暴露中间层输出（如 C2/C3/C4）。
- 构建 FPN：
   - 使用 1x1 conv 降维对齐通道；
   - 自顶向下上采样并逐级相加；
   - 3x3 conv 平滑，得到 P2/P3/P4；
   - 可选归一化（BN/GN）。
- 头部适配：复用或复制现有检测头/描述子头到每个 P 层，输出：
   - det_scores[L]：B×1×H_L×W_L
   - descriptors[L]：B×D×H_L×W_L（D 与现有描述子维度一致）
- 前向接口：
   - 训练模式：维持现有输出以兼容训练；
   - 匹配/评估模式：支持 `return_pyramid=True` 返回 {P2,P3,P4} 的 det/desc。

3) 匹配侧改造（`match.py`）
- 配置读取：根据 `matching.use_fpn` 决定走 FPN 或旧图像金字塔路径。
- FPN 路径：
   - 对 layout 与 template 各做一次前向，获得 {det, desc}×L；
   - 对每个层级 L：
      - 从 det_scores[L] 以 `score_threshold` 抽取关键点坐标与分数；
      - 半径 NMS 去重（见步骤 4）；
      - 使用 desc[L] 在对应层做特征最近邻匹配（可选比值测试）并估计单应性 H_L（RANSAC）；
   - 融合多个层级的候选：选取内点数最多或综合打分最佳的实例；
   - 将层级坐标映射回原图坐标；输出 bbox 与 H。
- 旧路径保留：若 `use_fpn=false`，继续使用当前图像金字塔多次推理策略，便于回退与对照实验。

4) 关键点去重（NMS/半径抑制）
- 输入：关键点集合 K = {(x, y, score)}。
- 算法：按 score 降序遍历，若与已保留点的欧氏距离 < radius 则丢弃，否则保留。
- 复杂度：O(N log N) 排序 + O(N·k) 检查（k 为邻域个数，可通过网格划分加速）。
- 参数：`matching.nms.radius`、`matching.nms.score_threshold`。

5) TensorBoard 记录（扩展）
- Scalars：
   - `match_fpn/level_L/keypoints_before_nms`、`keypoints_after_nms`
   - `match_fpn/level_L/inliers`、`best_instance_inliers`
   - `match_fpn/instances_found`、`runtime_ms`
- Text/Image：
   - 关键点可视化（可选），最佳实例覆盖图；
   - 记录使用的层级与最终选中尺度信息。

6) 兼容性与回退
- 通过 YAML `matching.use_fpn` 开关控制路径；
- 保持 CLI 不变，新增可选 `--fpn-off`（等同 use_fpn=false）仅作为临时调试；
- 若新路径异常可快速回退旧路径，保证生产可用性。

## 开发里程碑与工时预估
- M1（0.5 天）：配置与分支、占位接口、日志钩子。
- M2（1.5 天）：FPN 实现与前向接口；单图 smoke 测试。
- M3（1 天）：`match.py` FPN 路径、尺度回映射与候选融合。
- M4（0.5 天）：NMS 实现与参数打通；
- M5（0.5 天）：TensorBoard 指标与可视化；
- M6（0.5 天）：对照基线的性能与速度评估，整理报告。

## 质量门禁与验收标准
- 构建：`uv sync` 无错误；`python -m compileall` 通过；
- 功能：在 2–3 张样例上，FPN 路径输出的实例数量与旧路径相当或更优；
- 速度：相同输入，FPN 路径总耗时较旧路径下降 ≥ 30%；
- 稳定性：无异常崩溃；在找不到匹配时能优雅返回空结果；
- 指标：TensorBoard 中关键点数量、NMS 前后对比、内点数、总实例数与运行时均可见。

## 快速试用（命令）
```bash
# 同步环境
uv sync

# 基于 YAML 启用 FPN 匹配（推荐）
uv run python match.py \
   --config configs/base_config.yaml \
   --layout /path/to/layout.png \
   --template /path/to/template.png \
   --tb_log_matches

# 临时关闭 FPN（对照实验）
# 可通过把 configs 中 matching.use_fpn 设为 false，或后续提供 --fpn-off 开关

# 打开 TensorBoard 查看匹配指标
uv run tensorboard --logdir runs
```

## 风险与回滚
- FPN 输出与原检测/描述子头的维度/分布不一致，需在实现时对齐通道与归一化；
- 多层融合策略（如何选取最佳实例）可能影响稳定性，可先以“内点数最大”作为启发式；
- 如出现精度下降或不稳定，立即回退 `matching.use_fpn=false`，保留旧流程并开启数据记录比对差异。
