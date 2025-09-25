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
