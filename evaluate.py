from models.rord import RoRD
from data.ic_dataset import ICLayoutDataset
from utils.transforms import SobelTransform
from match import match_template_to_layout
import torch
from torchvision import transforms
import json
import os
from PIL import Image

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
    x2, y2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']
    
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def evaluate(model, val_dataset, templates, iou_threshold=0.5):
    model.eval()
    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0

    for layout_idx in range(len(val_dataset)):
        layout_image, annotation = val_dataset[layout_idx]
        # layout_image is [3, H, W]
        layout_tensor = layout_image.unsqueeze(0).cuda()  # [1, 3, H, W]
        
        # 假设 annotation 是 {"boxes": [{"template": "template1.png", "x": x, "y": y, "width": w, "height": h}, ...]}
        gt_boxes_by_template = {}
        for box in annotation.get('boxes', []):
            template_name = box['template']
            if template_name not in gt_boxes_by_template:
                gt_boxes_by_template[template_name] = []
            gt_boxes_by_template[template_name].append(box)

        for template_path in templates:
            template_name = os.path.basename(template_path)
            template_image = Image.open(template_path).convert('L')
            template_tensor = transform(template_image).unsqueeze(0).cuda()  # [1, 3, H, W]

            # 执行匹配
            detected_bboxes = match_template_to_layout(model, layout_tensor, template_tensor)

            # 获取当前模板的 gt_boxes
            gt_boxes = gt_boxes_by_template.get(template_name, [])

            # 初始化已分配的 gt_box 索引
            assigned_gt = set()

            for det_box in detected_bboxes:
                best_iou = 0
                best_gt_idx = -1
                for idx, gt_box in enumerate(gt_boxes):
                    if idx in assigned_gt:
                        continue
                    iou = compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                if best_iou > iou_threshold and best_gt_idx != -1:
                    all_true_positives += 1
                    assigned_gt.add(best_gt_idx)
                else:
                    all_false_positives += 1

            # 计算 FN：未分配的 gt_box
            for idx in range(len(gt_boxes)):
                if idx not in assigned_gt:
                    all_false_negatives += 1

    # 计算评估指标
    precision = all_true_positives / (all_true_positives + all_false_positives) if (all_true_positives + all_false_positives) > 0 else 0
    recall = all_true_positives / (all_true_positives + all_false_negatives) if (all_true_positives + all_false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    # 设置变换
    transform = transforms.Compose([
        SobelTransform(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # [1, H, W] -> [3, H, W]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载模型
    model = RoRD().cuda()
    model.load_state_dict(torch.load('path/to/weights.pth'))
    model.eval()

    # 定义验证数据集
    val_dataset = ICLayoutDataset(
        image_dir='path/to/val/images',
        annotation_dir='path/to/val/annotations',
        transform=transform
    )

    # 定义模板列表
    templates = ['path/to/templates/template1.png', 'path/to/templates/template2.png']  # 替换为实际模板路径

    # 评估模型
    results = evaluate(model, val_dataset, templates)
    print("评估结果：")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1 分数: {results['f1']:.4f}")