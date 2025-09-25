# evaluate.py

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from match import match_template_multiscale
from models.rord import RoRD
from utils.config_loader import load_config, to_absolute_path
from utils.data_utils import get_transform

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
    x2, y2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']
    inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
    inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --- (已修改) 评估函数 ---
def evaluate(
    model,
    val_dataset_dir,
    val_annotations_dir,
    template_dir,
    matching_cfg,
    iou_threshold,
    summary_writer: SummaryWriter | None = None,
    global_step: int = 0,
):
    model.eval()
    all_tp, all_fp, all_fn = 0, 0, 0
    
    # 只需要一个统一的 transform 给匹配函数内部使用
    transform = get_transform()
    
    template_paths = [os.path.join(template_dir, f) for f in os.listdir(template_dir) if f.endswith('.png')]
    layout_image_names = [f for f in os.listdir(val_dataset_dir) if f.endswith('.png')]

    if summary_writer:
        summary_writer.add_text(
            "dataset/info",
            f"layouts={len(layout_image_names)}, templates={len(template_paths)}",
            global_step,
        )

    # (已修改) 循环遍历验证集中的每个版图文件
    for layout_name in layout_image_names:
        print(f"\n正在评估版图: {layout_name}")
        layout_path = os.path.join(val_dataset_dir, layout_name)
        annotation_path = os.path.join(val_annotations_dir, layout_name.replace('.png', '.json'))

        # 加载原始PIL图像，以支持滑动窗口
        layout_image = Image.open(layout_path).convert('L')

        # 加载标注信息
        if not os.path.exists(annotation_path):
            continue
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        # 按模板对真实标注进行分组
        gt_by_template = {os.path.basename(box['template']): [] for box in annotation.get('boxes', [])}
        for box in annotation.get('boxes', []):
            gt_by_template[os.path.basename(box['template'])].append(box)

        # 遍历每个模板，在当前版图上进行匹配
        for template_path in template_paths:
            template_name = os.path.basename(template_path)
            template_image = Image.open(template_path).convert('L')
            
            # (已修改) 调用新的多尺度匹配函数
            detected = match_template_multiscale(model, layout_image, template_image, transform, matching_cfg)
            
            gt_boxes = gt_by_template.get(template_name, [])
            
            # 计算 TP, FP, FN (这部分逻辑不变)
            matched_gt = [False] * len(gt_boxes)
            tp = 0
            if len(detected) > 0:
                for det_box in detected:
                    best_iou = 0
                    best_gt_idx = -1
                    for i, gt_box in enumerate(gt_boxes):
                        if matched_gt[i]: continue
                        iou = compute_iou(det_box, gt_box)
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, i
                    
                    if best_iou > iou_threshold:
                        if not matched_gt[best_gt_idx]:
                            tp += 1
                            matched_gt[best_gt_idx] = True
            
            fp = len(detected) - tp
            fn = len(gt_boxes) - tp

            all_tp += tp
            all_fp += fp
            all_fn += fn

    # 计算最终指标
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if summary_writer:
        summary_writer.add_scalar("metrics/precision", precision, global_step)
        summary_writer.add_scalar("metrics/recall", recall, global_step)
        summary_writer.add_scalar("metrics/f1", f1, global_step)
        summary_writer.add_scalar("counts/true_positive", all_tp, global_step)
        summary_writer.add_scalar("counts/false_positive", all_fp, global_step)
        summary_writer.add_scalar("counts/false_negative", all_fn, global_step)

    return {'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 RoRD 模型性能")
    parser.add_argument('--config', type=str, default="configs/base_config.yaml", help="YAML 配置文件路径")
    parser.add_argument('--model_path', type=str, default=None, help="模型权重路径，若未提供则使用配置文件中的路径")
    parser.add_argument('--val_dir', type=str, default=None, help="验证图像目录，若未提供则使用配置文件中的路径")
    parser.add_argument('--annotations_dir', type=str, default=None, help="验证标注目录，若未提供则使用配置文件中的路径")
    parser.add_argument('--templates_dir', type=str, default=None, help="模板目录，若未提供则使用配置文件中的路径")
    parser.add_argument('--log_dir', type=str, default=None, help="TensorBoard 日志根目录，覆盖配置文件设置")
    parser.add_argument('--experiment_name', type=str, default=None, help="TensorBoard 实验名称，覆盖配置文件设置")
    parser.add_argument('--disable_tensorboard', action='store_true', help="禁用 TensorBoard 记录")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent
    paths_cfg = cfg.paths
    matching_cfg = cfg.matching
    eval_cfg = cfg.evaluation
    logging_cfg = cfg.get("logging", None)

    model_path = args.model_path or str(to_absolute_path(paths_cfg.model_path, config_dir))
    val_dir = args.val_dir or str(to_absolute_path(paths_cfg.val_img_dir, config_dir))
    annotations_dir = args.annotations_dir or str(to_absolute_path(paths_cfg.val_ann_dir, config_dir))
    templates_dir = args.templates_dir or str(to_absolute_path(paths_cfg.template_dir, config_dir))
    iou_threshold = float(eval_cfg.iou_threshold)

    use_tensorboard = False
    log_dir = None
    experiment_name = None
    if logging_cfg is not None:
        use_tensorboard = bool(logging_cfg.get("use_tensorboard", False))
        log_dir = logging_cfg.get("log_dir", "runs")
        experiment_name = logging_cfg.get("experiment_name", "default")

    if args.disable_tensorboard:
        use_tensorboard = False
    if args.log_dir is not None:
        log_dir = args.log_dir
    if args.experiment_name is not None:
        experiment_name = args.experiment_name

    writer = None
    if use_tensorboard and log_dir:
        log_root = Path(log_dir).expanduser()
        exp_folder = experiment_name or "default"
        tb_path = log_root / "eval" / exp_folder
        tb_path.parent.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tb_path.as_posix())

    model = RoRD().cuda()
    model.load_state_dict(torch.load(model_path))
    
    results = evaluate(
        model,
        val_dir,
        annotations_dir,
        templates_dir,
        matching_cfg,
        iou_threshold,
        summary_writer=writer,
        global_step=0,
    )
    
    print("\n--- 评估结果 ---")
    print(f"  精确率 (Precision): {results['precision']:.4f}")
    print(f"  召回率 (Recall):    {results['recall']:.4f}")
    print(f"  F1 分数 (F1 Score):  {results['f1']:.4f}")

    if writer:
        writer.add_text("metadata/model_path", model_path)
        writer.close()