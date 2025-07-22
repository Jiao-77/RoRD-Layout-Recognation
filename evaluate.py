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
# (Modified) Import new matching function
from match import match_template_multiscale

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
    x2, y2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']
    inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
    inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --- (Modified) Evaluation function ---
def evaluate(model, val_dataset_dir, val_annotations_dir, template_dir):
    model.eval()
    all_tp, all_fp, all_fn = 0, 0, 0
    
    # Only need a unified transform for internal use by matching function
    transform = get_transform()
    
    template_paths = [os.path.join(template_dir, f) for f in os.listdir(template_dir) if f.endswith('.png')]
    layout_image_names = [f for f in os.listdir(val_dataset_dir) if f.endswith('.png')]

    # (Modified) Loop through each layout file in validation set
    for layout_name in layout_image_names:
        print(f"\nEvaluating layout: {layout_name}")
        layout_path = os.path.join(val_dataset_dir, layout_name)
        annotation_path = os.path.join(val_annotations_dir, layout_name.replace('.png', '.json'))

        # Load original PIL image to support sliding window
        layout_image = Image.open(layout_path).convert('L')

        # Load annotation information
        if not os.path.exists(annotation_path):
            continue
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        # Group ground truth annotations by template
        gt_by_template = {os.path.basename(box['template']): [] for box in annotation.get('boxes', [])}
        for box in annotation.get('boxes', []):
            gt_by_template[os.path.basename(box['template'])].append(box)

        # Iterate through each template and perform matching on current layout
        for template_path in template_paths:
            template_name = os.path.basename(template_path)
            template_image = Image.open(template_path).convert('L')
            
            # (Modified) Call new multi-scale matching function
            detected = match_template_multiscale(model, layout_image, template_image, transform)
            
            gt_boxes = gt_by_template.get(template_name, [])
            
            # Calculate TP, FP, FN (this logic remains unchanged)
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
                    
                    if best_iou > config.IOU_THRESHOLD:
                        if not matched_gt[best_gt_idx]:
                            tp += 1
                            matched_gt[best_gt_idx] = True
            
            fp = len(detected) - tp
            fn = len(gt_boxes) - tp

            all_tp += tp
            all_fp += fp
            all_fn += fn

    # Calculate final metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RoRD model performance")
    parser.add_argument('--model_path', type=str, default=config.MODEL_PATH)
    parser.add_argument('--val_dir', type=str, default=config.VAL_IMG_DIR)
    parser.add_argument('--annotations_dir', type=str, default=config.VAL_ANN_DIR)
    parser.add_argument('--templates_dir', type=str, default=config.TEMPLATE_DIR)
    args = parser.parse_args()

    model = RoRD().cuda()
    model.load_state_dict(torch.load(args.model_path))
    
    # (Modified) No longer need to preload dataset, directly pass paths
    results = evaluate(model, args.val_dir, args.annotations_dir, args.templates_dir)
    
    print("\n--- Evaluation Results ---")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")