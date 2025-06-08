# match.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import argparse
import os

import config
from models.rord import RoRD
from utils.data_utils import get_transform

def extract_keypoints_and_descriptors(model, image, kp_thresh):
    with torch.no_grad():
        detection_map, desc = model(image)
        binary_map = (detection_map > kp_thresh).float()
        coords = torch.nonzero(binary_map[0, 0]).float()
        keypoints_input = coords[:, [1, 0]] * 8.0 # Stride of descriptor is 8

        descriptors = F.grid_sample(desc, coords.flip(1).view(1, -1, 1, 2) / torch.tensor([(desc.shape[3]-1)/2, (desc.shape[2]-1)/2], device=desc.device) - 1, align_corners=True).squeeze().T
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return keypoints_input, descriptors

def mutual_nearest_neighbor(descs1, descs2):
    sim = descs1 @ descs2.T
    nn12 = torch.max(sim, dim=1)
    nn21 = torch.max(sim, dim=0)
    ids1 = torch.arange(0, sim.shape[0], device=sim.device)
    mask = (ids1 == nn21.indices[nn12.indices])
    matches = torch.stack([ids1[mask], nn12.indices[mask]], dim=1)
    return matches.cpu().numpy()

def match_template_to_layout(model, layout_image, template_image):
    layout_kps, layout_descs = extract_keypoints_and_descriptors(model, layout_image, config.KEYPOINT_THRESHOLD)
    template_kps, template_descs = extract_keypoints_and_descriptors(model, template_image, config.KEYPOINT_THRESHOLD)

    active_layout_mask = torch.ones(len(layout_kps), dtype=bool, device=layout_kps.device)
    found_instances = []

    while True:
        current_indices = torch.nonzero(active_layout_mask).squeeze(1)
        if len(current_indices) < config.MIN_INLIERS:
            break

        current_layout_kps, current_layout_descs = layout_kps[current_indices], layout_descs[current_indices]
        matches = mutual_nearest_neighbor(template_descs, current_layout_descs)
        
        if len(matches) < 4: break

        src_pts = template_kps[matches[:, 0]].cpu().numpy()
        dst_pts = current_layout_kps[matches[:, 1]].cpu().numpy()

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)
        if H is None or mask.sum() < config.MIN_INLIERS:
            break

        inlier_mask = mask.ravel().astype(bool)
        
        # 区域屏蔽逻辑
        inlier_layout_kps = dst_pts[inlier_mask]
        x_min, y_min = inlier_layout_kps.min(axis=0)
        x_max, y_max = inlier_layout_kps.max(axis=0)
        
        instance = {'x': int(x_min), 'y': int(y_min), 'width': int(x_max - x_min), 'height': int(y_max - y_min), 'homography': H}
        found_instances.append(instance)

        kp_x, kp_y = layout_kps[:, 0], layout_kps[:, 1]
        region_mask = (kp_x >= x_min) & (kp_x <= x_max) & (kp_y >= y_min) & (kp_y <= y_max)
        active_layout_mask[region_mask] = False
        
        print(f"找到实例，内点数: {mask.sum()}。剩余活动关键点: {active_layout_mask.sum()}")
            
    return found_instances

def visualize_matches(layout_path, template_path, bboxes, output_path):
    layout_img = cv2.imread(layout_path)
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        cv2.rectangle(layout_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(layout_img, f"Match {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(output_path, layout_img)
    print(f"可视化结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 RoRD 进行模板匹配")
    parser.add_argument('--model_path', type=str, default=config.MODEL_PATH)
    parser.add_argument('--layout', type=str, required=True)
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    transform = get_transform()
    model = RoRD().cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    layout_tensor = transform(Image.open(args.layout).convert('L')).unsqueeze(0).cuda()
    template_tensor = transform(Image.open(args.template).convert('L')).unsqueeze(0).cuda()

    detected_bboxes = match_template_to_layout(model, layout_tensor, template_tensor)
    print("\n检测到的边界框:")
    for bbox in detected_bboxes:
        print(bbox)

    if args.output:
        visualize_matches(args.layout, args.template, detected_bboxes, args.output)