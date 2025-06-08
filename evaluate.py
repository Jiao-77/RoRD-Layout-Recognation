# evaluate.py

import torch
from PIL import Image
import json
import os
import argparse

import config
from models.rord import RoRD
from utils.data_utils import get_transform
from data.ic_dataset import ICLayoutDataset
from match import match_template_to_layout

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
    x2, y2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']
    inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
    inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate(model, val_dataset, template_dir):
    model.eval()
    all_tp, all_fp, all_fn = 0, 0, 0
    transform = get_transform()
    
    template_paths = [os.path.join(template_dir, f) for f in os.listdir(template_dir) if f.endswith('.png')]

    for layout_tensor, annotation in val_dataset:
        layout_tensor = layout_tensor.unsqueeze(0).cuda()
        gt_by_template = {box['template']: [] for box in annotation.get('boxes', [])}
        for box in annotation.get('boxes', []):
            gt_by_template[box['template']].append(box)

        for template_path in template_paths:
            template_name = os.path.basename(template_path)
            template_tensor = transform(Image.open(template_path).convert('L')).unsqueeze(0).cuda()
            
            detected = match_template_to_layout(model, layout_tensor, template_tensor)
            gt_boxes = gt_by_template.get(template_name, [])
            
            matched_gt = [False] * len(gt_boxes)
            tp = 0
            for det_box in detected:
                best_iou = 0
                best_gt_idx = -1
                for i, gt_box in enumerate(gt_boxes):
                    if matched_gt[i]: continue
                    iou = compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, i
                
                if best_iou > config.IOU_THRESHOLD:
                    tp += 1
                    matched_gt[best_gt_idx] = True
            
            all_tp += tp
            all_fp += len(detected) - tp
            all_fn += len(gt_boxes) - tp

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 RoRD 模型性能")
    parser.add_argument('--model_path', type=str, default=config.MODEL_PATH)
    parser.add_argument('--val_dir', type=str, default=config.VAL_IMG_DIR)
    parser.add_argument('--annotations_dir', type=str, default=config.VAL_ANN_DIR)
    parser.add_argument('--templates_dir', type=str, default=config.TEMPLATE_DIR)
    args = parser.parse_args()

    model = RoRD().cuda()
    model.load_state_dict(torch.load(args.model_path))
    val_dataset = ICLayoutDataset(args.val_dir, args.annotations_dir, get_transform())
    
    results = evaluate(model, val_dataset, args.templates_dir)
    print("评估结果：")
    print(f"  精确率 (Precision): {results['precision']:.4f}")
    print(f"  召回率 (Recall):    {results['recall']:.4f}")
    print(f"  F1 分数 (F1 Score):  {results['f1']:.4f}")