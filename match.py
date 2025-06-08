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

# --- 特征提取函数 (基本无变动) ---
def extract_keypoints_and_descriptors(model, image_tensor, kp_thresh):
    with torch.no_grad():
        detection_map, desc = model(image_tensor)
    
    device = detection_map.device
    binary_map = (detection_map > kp_thresh).squeeze(0).squeeze(0)
    coords = torch.nonzero(binary_map).float() # y, x
    
    if len(coords) == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    # 描述子采样
    coords_for_grid = coords.flip(1).view(1, -1, 1, 2) # N, 2 -> 1, N, 1, 2 (x,y)
    # 归一化到 [-1, 1]
    coords_for_grid = coords_for_grid / torch.tensor([(desc.shape[3]-1)/2, (desc.shape[2]-1)/2], device=device) - 1
    
    descriptors = F.grid_sample(desc, coords_for_grid, align_corners=True).squeeze().T
    descriptors = F.normalize(descriptors, p=2, dim=1)
    
    # 将关键点坐标从特征图尺度转换回图像尺度
    # VGG到relu4_3的下采样率为8
    keypoints = coords.flip(1) * 8.0 # x, y

    return keypoints, descriptors

# --- (新增) 滑动窗口特征提取函数 ---
def extract_features_sliding_window(model, large_image, transform):
    """
    使用滑动窗口从大图上提取所有关键点和描述子
    """
    print("使用滑动窗口提取大版图特征...")
    device = next(model.parameters()).device
    W, H = large_image.size
    window_size = config.INFERENCE_WINDOW_SIZE
    stride = config.INFERENCE_STRIDE

    all_kps = []
    all_descs = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # 确保窗口不越界
            x_end = min(x + window_size, W)
            y_end = min(y + window_size, H)
            
            # 裁剪窗口
            patch = large_image.crop((x, y, x_end, y_end))
            
            # 预处理
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            
            # 提取特征
            kps, descs = extract_keypoints_and_descriptors(model, patch_tensor, config.KEYPOINT_THRESHOLD)
            
            if len(kps) > 0:
                # 将局部坐标转换为全局坐标
                kps[:, 0] += x
                kps[:, 1] += y
                all_kps.append(kps)
                all_descs.append(descs)
    
    if not all_kps:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    print(f"大版图特征提取完毕，共找到 {sum(len(k) for k in all_kps)} 个关键点。")
    return torch.cat(all_kps, dim=0), torch.cat(all_descs, dim=0)


# --- 互近邻匹配 (无变动) ---
def mutual_nearest_neighbor(descs1, descs2):
    if len(descs1) == 0 or len(descs2) == 0:
        return torch.empty((0, 2), dtype=torch.int64)
    sim = descs1 @ descs2.T
    nn12 = torch.max(sim, dim=1)
    nn21 = torch.max(sim, dim=0)
    ids1 = torch.arange(0, sim.shape[0], device=sim.device)
    mask = (ids1 == nn21.indices[nn12.indices])
    matches = torch.stack([ids1[mask], nn12.indices[mask]], dim=1)
    return matches

# --- (已修改) 多尺度、多实例匹配主函数 ---
def match_template_multiscale(model, layout_image, template_image, transform):
    """
    在不同尺度下搜索模板，并检测多个实例
    """
    # 1. 对大版图使用滑动窗口提取全部特征
    layout_kps, layout_descs = extract_features_sliding_window(model, layout_image, transform)
    
    if len(layout_kps) < config.MIN_INLIERS:
        print("从大版图中提取的关键点过少，无法进行匹配。")
        return []

    found_instances = []
    active_layout_mask = torch.ones(len(layout_kps), dtype=bool, device=layout_kps.device)
    
    # 2. 多实例迭代检测
    while True:
        current_active_indices = torch.nonzero(active_layout_mask).squeeze(1)
        
        # 如果剩余活动关键点过少，则停止
        if len(current_active_indices) < config.MIN_INLIERS:
            break

        current_layout_kps = layout_kps[current_active_indices]
        current_layout_descs = layout_descs[current_active_indices]
        
        best_match_info = {'inliers': 0, 'H': None, 'src_pts': None, 'dst_pts': None, 'mask': None}

        # 3. 图像金字塔：遍历模板的每个尺度
        print("在新尺度下搜索模板...")
        for scale in config.PYRAMID_SCALES:
            W, H = template_image.size
            new_W, new_H = int(W * scale), int(H * scale)
            
            # 缩放模板
            scaled_template = template_image.resize((new_W, new_H), Image.LANCZOS)
            template_tensor = transform(scaled_template).unsqueeze(0).to(layout_kps.device)
            
            # 提取缩放后模板的特征
            template_kps, template_descs = extract_keypoints_and_descriptors(model, template_tensor, config.KEYPOINT_THRESHOLD)
            
            if len(template_kps) < 4: continue

            # 匹配当前尺度的模板和活动状态的版图特征
            matches = mutual_nearest_neighbor(template_descs, current_layout_descs)
            
            if len(matches) < 4: continue

            # RANSAC
            # 注意：模板关键点坐标需要还原到原始尺寸，才能计算正确的H
            src_pts = template_kps[matches[:, 0]].cpu().numpy() / scale
            dst_pts_indices = current_active_indices[matches[:, 1]]
            dst_pts = layout_kps[dst_pts_indices].cpu().numpy()

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)

            if H is not None and mask.sum() > best_match_info['inliers']:
                best_match_info = {'inliers': mask.sum(), 'H': H, 'mask': mask, 'scale': scale, 'dst_pts': dst_pts}

        # 4. 如果在所有尺度中找到了最佳匹配，则记录并屏蔽
        if best_match_info['inliers'] > config.MIN_INLIERS:
            print(f"找到一个匹配实例！内点数: {best_match_info['inliers']}, 使用的模板尺度: {best_match_info['scale']:.2f}x")
            
            inlier_mask = best_match_info['mask'].ravel().astype(bool)
            inlier_layout_kps = best_match_info['dst_pts'][inlier_mask]

            x_min, y_min = inlier_layout_kps.min(axis=0)
            x_max, y_max = inlier_layout_kps.max(axis=0)
            
            instance = {'x': int(x_min), 'y': int(y_min), 'width': int(x_max - x_min), 'height': int(y_max - y_min), 'homography': best_match_info['H']}
            found_instances.append(instance)

            # 屏蔽已匹配区域的关键点，以便检测下一个实例
            kp_x, kp_y = layout_kps[:, 0], layout_kps[:, 1]
            region_mask = (kp_x >= x_min) & (kp_x <= x_max) & (kp_y >= y_min) & (kp_y <= y_max)
            active_layout_mask[region_mask] = False
            
            print(f"剩余活动关键点: {active_layout_mask.sum()}")
        else:
            # 如果在所有尺度下都找不到好的匹配，则结束搜索
            print("在所有尺度下均未找到新的匹配实例，搜索结束。")
            break
            
    return found_instances


def visualize_matches(layout_path, bboxes, output_path):
    layout_img = cv2.imread(layout_path)
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        cv2.rectangle(layout_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(layout_img, f"Match {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(output_path, layout_img)
    print(f"可视化结果已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 RoRD 进行多尺度模板匹配")
    parser.add_argument('--model_path', type=str, default=config.MODEL_PATH)
    parser.add_argument('--layout', type=str, required=True)
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    transform = get_transform()
    model = RoRD().cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    layout_image = Image.open(args.layout).convert('L')
    template_image = Image.open(args.template).convert('L')
    
    detected_bboxes = match_template_multiscale(model, layout_image, template_image, transform)
    
    print("\n检测到的边界框:")
    for bbox in detected_bboxes:
        print(bbox)

    if args.output:
        visualize_matches(args.layout, detected_bboxes, args.output)