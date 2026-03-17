# match.py

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import DictConfig
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - fallback for environments without torch tensorboard
    from tensorboardX import SummaryWriter  # type: ignore

from models.rord import RoRD
from utils.config_loader import load_config, to_absolute_path
from utils.data_utils import get_transform

# 配置日志
logger = logging.getLogger(__name__)

# --- 新增：功能增强函数 ---
def extract_rotation_angle(H: Optional[np.ndarray]) -> int:
    """
    从单应性矩阵中提取旋转角度。

    Args:
        H: 3x3 单应性矩阵，如果为 None 则返回 0

    Returns:
        旋转角度：0°, 90°, 180°, 270° 之一
    """
    if H is None:
        return 0

    # 提取旋转分量
    cos_theta = H[0, 0] / np.sqrt(H[0, 0]**2 + H[1, 0]**2 + 1e-8)
    sin_theta = H[1, 0] / np.sqrt(H[0, 0]**2 + H[1, 0]**2 + 1e-8)

    # 计算角度（弧度转角度）
    angle = np.arctan2(sin_theta, cos_theta) * 180 / np.pi

    # 四舍五入到最近的90度倍数
    angles = [0, 90, 180, 270]
    nearest_angle = min(angles, key=lambda x: abs(x - angle))

    return nearest_angle


def calculate_match_score(
    inlier_count: int,
    total_keypoints: int,
    H: Optional[np.ndarray],
    inlier_ratio: Optional[float] = None
) -> float:
    """
    计算匹配质量评分 (0-1)。

    Args:
        inlier_count: 内点数量
        total_keypoints: 总关键点数量
        H: 单应性矩阵
        inlier_ratio: 内点比例（可选）

    Returns:
        匹配质量评分，范围 [0, 1]
    """
    if inlier_ratio is None:
        inlier_ratio = inlier_count / max(total_keypoints, 1)

    # 基于内点比例的基础分数
    base_score = inlier_ratio

    # 基于变换矩阵质量的分数（越接近单位矩阵分数越高）
    if H is not None:
        # 计算变换的"理想程度"
        det = np.linalg.det(H)
        ideal_det = 1.0
        det_score = 1.0 / (1.0 + abs(np.log(det + 1e-8)))

        # 综合评分
        final_score = base_score * 0.7 + det_score * 0.3
    else:
        final_score = base_score

    return min(max(final_score, 0.0), 1.0)


def calculate_similarity(
    matches_count: int,
    template_kps_count: int,
    layout_kps_count: int
) -> float:
    """
    计算模板和版图之间的相似度。

    Args:
        matches_count: 匹配对数量
        template_kps_count: 模板关键点数量
        layout_kps_count: 版图关键点数量

    Returns:
        相似度评分，范围 [0, 1]
    """
    # 匹配率
    template_match_rate = matches_count / max(template_kps_count, 1)

    # 覆盖率（简化计算）
    coverage_rate = min(matches_count / max(layout_kps_count, 1), 1.0)

    # 综合相似度
    similarity = (template_match_rate * 0.6 + coverage_rate * 0.4)

    return min(max(similarity, 0.0), 1.0)


def generate_difference_description(
    H: Optional[np.ndarray],
    inlier_count: int,
    total_matches: int,
    angle_diff: int = 0
) -> str:
    """
    生成差异描述。

    Args:
        H: 单应性矩阵
        inlier_count: 内点数量
        total_matches: 总匹配数
        angle_diff: 角度差异

    Returns:
        差异描述字符串
    """
    descriptions = []

    # 基于内点比例的描述
    if total_matches > 0:
        inlier_ratio = inlier_count / total_matches
        if inlier_ratio > 0.8:
            descriptions.append("高度匹配")
        elif inlier_ratio > 0.6:
            descriptions.append("良好匹配")
        elif inlier_ratio > 0.4:
            descriptions.append("中等匹配")
        else:
            descriptions.append("低质量匹配")

    # 基于旋转的描述
    if angle_diff != 0:
        descriptions.append(f"旋转{angle_diff}度")
    else:
        descriptions.append("无旋转")

    # 基于变换的描述
    if H is not None:
        # 检查缩放
        scale_x = np.sqrt(H[0,0]**2 + H[1,0]**2)
        scale_y = np.sqrt(H[0,1]**2 + H[1,1]**2)
        avg_scale = (scale_x + scale_y) / 2

        if abs(avg_scale - 1.0) > 0.1:
            if avg_scale > 1.0:
                descriptions.append(f"放大{avg_scale:.2f}倍")
            else:
                descriptions.append(f"缩小{1/avg_scale:.2f}倍")

    return ", ".join(descriptions) if descriptions else "无法评估差异"


# --- 特征提取函数 (基本无变动) ---
def extract_keypoints_and_descriptors(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    kp_thresh: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从图像张量中提取关键点和描述子。

    Args:
        model: RoRD 模型
        image_tensor: 输入图像张量，形状为 (B, C, H, W)
        kp_thresh: 关键点检测阈值

    Returns:
        元组 (keypoints, descriptors):
            - keypoints: 关键点坐标，形状为 (N, 2)
            - descriptors: 描述子，形状为 (N, D)
    """
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


# --- (新增) 简单半径 NMS 去重 ---
def radius_nms(kps: torch.Tensor, scores: torch.Tensor, radius: float) -> torch.Tensor:
    """
    半径非极大值抑制 (NMS) 去重。

    Args:
        kps: 关键点坐标张量，形状为 (N, 2)
        scores: 关键点得分张量，形状为 (N,)
        radius: 抑制半径

    Returns:
        保留的关键点索引张量
    """
    # 检查空张量情况
    if kps.numel() == 0 or scores.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=kps.device)

    # 检查长度一致性
    if kps.shape[0] != scores.shape[0]:
        raise ValueError(f"关键点和得分数量不匹配: kps={kps.shape[0]}, scores={scores.shape[0]}")

    idx = torch.argsort(scores, descending=True)
    keep = []
    taken = torch.zeros(len(kps), dtype=torch.bool, device=kps.device)
    for i in idx:
        if taken[i]:
            continue
        keep.append(i.item())
        di = kps - kps[i]
        dist2 = (di[:, 0]**2 + di[:, 1]**2)
        taken |= dist2 <= (radius * radius)
        taken[i] = True
    return torch.tensor(keep, dtype=torch.long, device=kps.device)

# --- (新增) 滑动窗口特征提取函数 ---
def extract_features_sliding_window(
    model: torch.nn.Module,
    large_image: Image.Image,
    transform: Any,
    matching_cfg: DictConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用滑动窗口从大图上提取所有关键点和描述子。

    Args:
        model: RoRD 模型
        large_image: 大版图 PIL 图像
        transform: 图像预处理变换
        matching_cfg: 匹配配置

    Returns:
        元组 (keypoints, descriptors):
            - keypoints: 所有关键点坐标，形状为 (N, 2)
            - descriptors: 所有描述子，形状为 (N, D)
    """
    logger.info("使用滑动窗口提取大版图特征...")
    device = next(model.parameters()).device
    W, H = large_image.size
    window_size = int(matching_cfg.inference_window_size)
    stride = int(matching_cfg.inference_stride)
    keypoint_threshold = float(matching_cfg.keypoint_threshold)

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
            kps, descs = extract_keypoints_and_descriptors(model, patch_tensor, keypoint_threshold)
            
            if len(kps) > 0:
                # 将局部坐标转换为全局坐标
                kps[:, 0] += x
                kps[:, 1] += y
                all_kps.append(kps)
                all_descs.append(descs)
    
    if not all_kps:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    total_kps = sum(len(k) for k in all_kps)
    logger.info(f"大版图特征提取完毕，共找到 {total_kps} 个关键点。")
    return torch.cat(all_kps, dim=0), torch.cat(all_descs, dim=0)


# --- (新增) FPN 路径的关键点与描述子抽取 ---
def extract_from_pyramid(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    kp_thresh: float,
    nms_cfg: Optional[DictConfig]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 FPN 从图像中提取多尺度关键点和描述子。

    Args:
        model: RoRD 模型
        image_tensor: 输入图像张量
        kp_thresh: 关键点检测阈值
        nms_cfg: NMS 配置（可选）

    Returns:
        元组 (keypoints, descriptors)
    """
    with torch.no_grad():
        pyramid = model(image_tensor, return_pyramid=True)
    all_kps = []
    all_desc = []
    for level_name, (det, desc, stride) in pyramid.items():
        binary = (det > kp_thresh).squeeze(0).squeeze(0)
        coords = torch.nonzero(binary).float()  # y,x
        if len(coords) == 0:
            continue
        scores = det.squeeze()[binary]
        # 采样描述子
        coords_for_grid = coords.flip(1).view(1, -1, 1, 2)
        coords_for_grid = coords_for_grid / torch.tensor([(desc.shape[3]-1)/2, (desc.shape[2]-1)/2], device=desc.device) - 1
        descs = F.grid_sample(desc, coords_for_grid, align_corners=True).squeeze().T
        descs = F.normalize(descs, p=2, dim=1)

        # 映射回原图坐标
        kps = coords.flip(1) * float(stride)

        # NMS
        if nms_cfg and nms_cfg.get('enabled', False):
            keep = radius_nms(kps, scores, float(nms_cfg.get('radius', 4)))
            if len(keep) > 0:
                kps = kps[keep]
                descs = descs[keep]
        all_kps.append(kps)
        all_desc.append(descs)
    if not all_kps:
        return torch.tensor([], device=image_tensor.device), torch.tensor([], device=image_tensor.device)
    return torch.cat(all_kps, dim=0), torch.cat(all_desc, dim=0)


# --- 互近邻匹配 (无变动) ---
def mutual_nearest_neighbor(
    descs1: torch.Tensor,
    descs2: torch.Tensor
) -> torch.Tensor:
    """
    执行互近邻匹配。

    Args:
        descs1: 第一组描述子，形状为 (N, D)
        descs2: 第二组描述子，形状为 (M, D)

    Returns:
        匹配对索引，形状为 (K, 2)，每行是 (descs1_index, descs2_index)
    """
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
def match_template_multiscale(
    model: torch.nn.Module,
    layout_image: Image.Image,
    template_image: Image.Image,
    transform: Any,
    matching_cfg: DictConfig,
    log_writer: Optional[SummaryWriter] = None,
    log_step: int = 0,
    return_detailed_info: bool = True,
) -> List[Dict[str, Any]]:
    """
    在不同尺度下搜索模板，并检测多个实例。

    Args:
        model: RoRD 模型
        layout_image: 大版图 PIL 图像
        template_image: 小版图（模板）PIL 图像
        transform: 图像预处理变换
        matching_cfg: 匹配配置
        log_writer: TensorBoard 日志记录器
        log_step: 日志步数
        return_detailed_info: 是否返回详细信息

    Returns:
        匹配结果列表，每个元素包含坐标、旋转角度、置信度等信息
    """
    # 1. 版图特征提取：根据配置选择 FPN 或滑窗
    device = next(model.parameters()).device
    if getattr(matching_cfg, 'use_fpn', False):
        layout_tensor = transform(layout_image).unsqueeze(0).to(device)
        layout_kps, layout_descs = extract_from_pyramid(model, layout_tensor, float(matching_cfg.keypoint_threshold), getattr(matching_cfg, 'nms', {}))
    else:
        layout_kps, layout_descs = extract_features_sliding_window(model, layout_image, transform, matching_cfg)
    if log_writer:
        log_writer.add_scalar("match/layout_keypoints", len(layout_kps), log_step)
    
    min_inliers = int(matching_cfg.min_inliers)
    if len(layout_kps) < min_inliers:
        logger.warning("从大版图中提取的关键点过少，无法进行匹配。")
        if log_writer:
            log_writer.add_scalar("match/instances_found", 0, log_step)
        return []

    found_instances = []
    active_layout_mask = torch.ones(len(layout_kps), dtype=bool, device=layout_kps.device)
    pyramid_scales = [float(s) for s in matching_cfg.pyramid_scales]
    keypoint_threshold = float(matching_cfg.keypoint_threshold)
    ransac_threshold = float(matching_cfg.ransac_reproj_threshold)
    
    # 2. 多实例迭代检测
    while True:
        current_active_indices = torch.nonzero(active_layout_mask).squeeze(1)
        
        # 如果剩余活动关键点过少，则停止
        if len(current_active_indices) < min_inliers:
            break

        current_layout_kps = layout_kps[current_active_indices]
        current_layout_descs = layout_descs[current_active_indices]
        
        best_match_info = {'inliers': 0, 'H': None, 'src_pts': None, 'dst_pts': None, 'mask': None}

        # 3. 图像金字塔：遍历模板的每个尺度
        logger.info("在新尺度下搜索模板...")
        for scale in pyramid_scales:
            W, H = template_image.size
            new_W, new_H = int(W * scale), int(H * scale)
            
            # 缩放模板
            scaled_template = template_image.resize((new_W, new_H), Image.LANCZOS)
            template_tensor = transform(scaled_template).unsqueeze(0).to(layout_kps.device)
            
            # 提取缩放后模板的特征：FPN 或单尺度
            if getattr(matching_cfg, 'use_fpn', False):
                template_kps, template_descs = extract_from_pyramid(model, template_tensor, keypoint_threshold, getattr(matching_cfg, 'nms', {}))
            else:
                template_kps, template_descs = extract_keypoints_and_descriptors(model, template_tensor, keypoint_threshold)
            
            if len(template_kps) < 4: continue

            # 匹配当前尺度的模板和活动状态的版图特征
            matches = mutual_nearest_neighbor(template_descs, current_layout_descs)
            
            if len(matches) < 4: continue

            # RANSAC
            # 注意：模板关键点坐标需要还原到原始尺寸，才能计算正确的H
            src_pts = template_kps[matches[:, 0]].cpu().numpy() / scale
            dst_pts_indices = current_active_indices[matches[:, 1]]
            dst_pts = layout_kps[dst_pts_indices].cpu().numpy()

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

            if H is not None and mask.sum() > best_match_info['inliers']:
                best_match_info = {
                    'inliers': mask.sum(),
                    'H': H,
                    'src_pts': src_pts,
                    'dst_pts': dst_pts,
                    'mask': mask,
                    'scale': scale
                }

        # 4. 如果在所有尺度中找到了最佳匹配，则记录并屏蔽
        if best_match_info['inliers'] > min_inliers:
            logger.info(f"找到一个匹配实例！内点数: {best_match_info['inliers']}, 使用的模板尺度: {best_match_info['scale']:.2f}x")
            if log_writer:
                instance_index = len(found_instances)
                log_writer.add_scalar("match/instance_inliers", int(best_match_info['inliers']), log_step + instance_index)
                log_writer.add_scalar("match/instance_scale", float(best_match_info['scale']), log_step + instance_index)
            
            inlier_mask = best_match_info['mask'].ravel().astype(bool)
            inlier_layout_kps = best_match_info['dst_pts'][inlier_mask]

            x_min, y_min = inlier_layout_kps.min(axis=0)
            x_max, y_max = inlier_layout_kps.max(axis=0)

            # 提取旋转角度
            rotation_angle = extract_rotation_angle(best_match_info['H'])

            # 计算匹配质量评分
            confidence = calculate_match_score(
                inlier_count=int(best_match_info['inliers']),
                total_keypoints=len(current_layout_kps),
                H=best_match_info['H']
            )

            # 计算相似度
            similarity = calculate_similarity(
                matches_count=int(best_match_info['inliers']),
                template_kps_count=len(template_kps),
                layout_kps_count=len(current_layout_kps)
            )

            # 生成差异描述
            diff_description = generate_difference_description(
                H=best_match_info['H'],
                inlier_count=int(best_match_info['inliers']),
                total_matches=len(matches),
                angle_diff=rotation_angle
            )

            # 构建详细实例信息
            if return_detailed_info:
                instance = {
                    'bbox': {
                        'x': int(x_min),
                        'y': int(y_min),
                        'width': int(x_max - x_min),
                        'height': int(y_max - y_min)
                    },
                    'rotation': rotation_angle,
                    'confidence': round(confidence, 3),
                    'similarity': round(similarity, 3),
                    'inliers': int(best_match_info['inliers']),
                    'scale': best_match_info.get('scale', 1.0),
                    'homography': best_match_info['H'].tolist() if best_match_info['H'] is not None else None,
                    'description': diff_description
                }
            else:
                # 兼容旧格式
                instance = {
                    'x': int(x_min),
                    'y': int(y_min),
                    'width': int(x_max - x_min),
                    'height': int(y_max - y_min),
                    'homography': best_match_info['H']
                }

            found_instances.append(instance)

            # 屏蔽已匹配区域的关键点，以便检测下一个实例
            kp_x, kp_y = layout_kps[:, 0], layout_kps[:, 1]
            region_mask = (kp_x >= x_min) & (kp_x <= x_max) & (kp_y >= y_min) & (kp_y <= y_max)
            active_layout_mask[region_mask] = False
            
            logger.info(f"剩余活动关键点: {active_layout_mask.sum()}")
        else:
            # 如果在所有尺度下都找不到好的匹配，则结束搜索
            logger.info("在所有尺度下均未找到新的匹配实例，搜索结束。")
            break
            
    if log_writer:
        log_writer.add_scalar("match/instances_found", len(found_instances), log_step)

    return found_instances


def visualize_matches(
    layout_path: Union[str, Path],
    matches: List[Dict[str, Any]],
    output_path: Union[str, Path]
) -> None:
    """
    可视化匹配结果，支持新的详细格式。

    Args:
        layout_path: 大版图路径
        matches: 匹配结果列表
        output_path: 输出图像路径
    """
    layout_img = cv2.imread(str(layout_path))
    if layout_img is None:
        logger.error(f"无法读取图像 {layout_path}")
        return

    for i, match in enumerate(matches):
        # 支持新旧格式
        if 'bbox' in match:
            x, y, w, h = match['bbox']['x'], match['bbox']['y'], match['bbox']['width'], match['bbox']['height']
            confidence = match.get('confidence', 0)
            rotation = match.get('rotation', 0)
            description = match.get('description', '')
        else:
            # 兼容旧格式
            x, y, w, h = match['x'], match['y'], match['width'], match['height']
            confidence = 0
            rotation = 0
            description = ''

        # 绘制边界框
        cv2.rectangle(layout_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 准备标签文本
        label_parts = [f"Match {i+1}"]
        if confidence > 0:
            label_parts.append(f"Conf: {confidence:.2f}")
        if rotation != 0:
            label_parts.append(f"Rot: {rotation}°")
        if description:
            label_parts.append(f"{description[:20]}...")  # 截断长描述

        label = " | ".join(label_parts)

        # 绘制标签背景
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(layout_img, (x, y - label_height - 10), (x + label_width, y), (0, 255, 0), -1)
        cv2.putText(layout_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imwrite(str(output_path), layout_img)
    logger.info(f"可视化结果已保存至: {output_path}")


def save_matches_json(
    matches: List[Dict[str, Any]],
    output_path: Union[str, Path]
) -> None:
    """
    将匹配结果保存为 JSON 文件。

    Args:
        matches: 匹配结果列表
        output_path: 输出 JSON 文件路径
    """
    result = {
        'found_matches': len(matches) > 0,
        'total_matches': len(matches),
        'matches': matches
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"匹配结果已保存至: {output_path}")


def print_detailed_results(matches: List[Dict[str, Any]]) -> None:
    """
    打印详细的匹配结果。

    Args:
        matches: 匹配结果列表
    """
    print("\n" + "="*60)
    print("🎯 版图匹配结果详情")
    print("="*60)

    if not matches:
        print("❌ 未找到任何匹配区域")
        return

    print(f"✅ 共找到 {len(matches)} 个匹配区域\n")

    for i, match in enumerate(matches, 1):
        print(f"📍 匹配区域 #{i}")
        print("-" * 40)

        # 支持新旧格式
        if 'bbox' in match:
            bbox = match['bbox']
            print(f"📐 位置: ({bbox['x']}, {bbox['y']})")
            print(f"📏 尺寸: {bbox['width']} × {bbox['height']} 像素")

            if 'rotation' in match:
                print(f"🔄 旋转角度: {match['rotation']}°")
            if 'confidence' in match:
                print(f"🎯 置信度: {match['confidence']:.3f}")
            if 'similarity' in match:
                print(f"📊 相似度: {match['similarity']:.3f}")
            if 'inliers' in match:
                print(f"🔗 内点数量: {match['inliers']}")
            if 'scale' in match:
                print(f"📈 匹配尺度: {match['scale']:.2f}x")
            if 'description' in match:
                print(f"📝 差异描述: {match['description']}")
        else:
            # 兼容旧格式
            print(f"📐 位置: ({match['x']}, {match['y']})")
            print(f"📏 尺寸: {match['width']} × {match['height']} 像素")

        print()  # 空行分隔


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 RoRD 进行多尺度模板匹配")
    parser.add_argument('--config', type=str, default="configs/base_config.yaml", help="YAML 配置文件路径")
    parser.add_argument('--model_path', type=str, default=None, help="模型权重路径，若未提供则使用配置文件中的路径")
    parser.add_argument('--log_dir', type=str, default=None, help="TensorBoard 日志根目录，覆盖配置文件设置")
    parser.add_argument('--experiment_name', type=str, default=None, help="TensorBoard 实验名称，覆盖配置文件设置")
    parser.add_argument('--tb_log_matches', action='store_true', help="启用模板匹配过程的 TensorBoard 记录")
    parser.add_argument('--disable_tensorboard', action='store_true', help="禁用 TensorBoard 记录")
    parser.add_argument('--fpn_off', action='store_true', help="关闭 FPN 匹配路径（等同于 matching.use_fpn=false）")
    parser.add_argument('--no_nms', action='store_true', help="关闭关键点去重（NMS）")
    parser.add_argument('--layout', type=str, required=True, help="大版图图像路径")
    parser.add_argument('--template', type=str, required=True, help="小版图（模板）图像路径")
    parser.add_argument('--output', type=str, help="可视化结果保存路径")
    parser.add_argument('--json_output', type=str, help="JSON结果保存路径")
    parser.add_argument('--simple_format', action='store_true', help="使用简单的输出格式（兼容旧版本）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent
    matching_cfg = cfg.matching
    logging_cfg = cfg.get("logging", None)
    model_path = args.model_path or str(to_absolute_path(cfg.paths.model_path, config_dir))

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

    should_log_matches = args.tb_log_matches and use_tensorboard and log_dir is not None
    writer = None
    if should_log_matches:
        log_root = Path(log_dir).expanduser()
        exp_folder = experiment_name or "default"
        tb_path = log_root / "match" / exp_folder
        tb_path.parent.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tb_path.as_posix())

    # CLI 快捷开关覆盖 YAML 配置
    try:
        if args.fpn_off:
            matching_cfg.use_fpn = False
        if args.no_nms and hasattr(matching_cfg, 'nms'):
            matching_cfg.nms.enabled = False
    except Exception:
        # 若 OmegaConf 结构不可写，忽略并在后续逻辑中以 getattr 的方式读取
        pass

    # 设备选择：支持 CUDA / CPU，优先使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU 型号: {torch.cuda.get_device_name(0)}")

    transform = get_transform()
    model = RoRD().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    layout_image = Image.open(args.layout).convert('L')
    template_image = Image.open(args.template).convert('L')
    
    # 执行匹配，根据参数选择详细或简单格式
    detected_matches = match_template_multiscale(
        model,
        layout_image,
        template_image,
        transform,
        matching_cfg,
        log_writer=writer,
        log_step=0,
        return_detailed_info=not args.simple_format,
    )

    # 打印详细结果
    print_detailed_results(detected_matches)

    # 保存JSON结果
    if args.json_output:
        save_matches_json(detected_matches, args.json_output)

    # 可视化结果
    if args.output:
        visualize_matches(args.layout, detected_matches, args.output)

    if writer:
        writer.add_scalar("match/output_instances", len(detected_matches), 0)
        writer.add_text("match/layout_path", args.layout, 0)
        writer.close()

    print("\n🎉 匹配完成！")
    if args.json_output:
        print(f"📄 详细结果已保存到: {args.json_output}")
    if args.output:
        print(f"🖼️ 可视化结果已保存到: {args.output}")