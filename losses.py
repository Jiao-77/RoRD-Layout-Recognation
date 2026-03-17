"""Loss utilities for RoRD training."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 损失函数权重常量
# ============================================================================
# 检测损失权重
DET_SMOOTH_L1_WEIGHT = 0.1  # smooth L1 损失在检测损失中的权重

# 描述子损失权重
DESC_GEOMETRIC_WEIGHT = 1.0    # 几何 triplet 损失权重
DESC_MANHATTAN_WEIGHT = 0.1    # 曼哈顿损失权重
DESC_SPARSITY_WEIGHT = 0.01    # 稀疏性损失权重
DESC_BINARY_WEIGHT = 0.05      # 二值化损失权重

# ============================================================================
# 预计算的旋转矩阵
# ============================================================================
# 用于描述子损失中的负样本生成
# 避免在循环中重复创建张量
import math
import torch

# 预计算 90°、180°、270° 的旋转矩阵
# 旋转矩阵: [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
_PRECOMPUTED_ROTATIONS_2D = {
    90: torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32),
    180: torch.tensor([[-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32),
    270: torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32),
}


def _get_rotation_matrix(angle: int, device: torch.device) -> torch.Tensor:
    """
    获取预计算的旋转矩阵。

    Args:
        angle: 旋转角度（90, 180, 270）
        device: 目标设备

    Returns:
        2x2 旋转矩阵
    """
    if angle not in _PRECOMPUTED_ROTATIONS_2D:
        raise ValueError(f"不支持的角度: {angle}，支持的角度: 90, 180, 270")
    
    rot = _PRECOMPUTED_ROTATIONS_2D[angle]
    return rot.to(device)


def _augment_homography_matrix(h_2x3: torch.Tensor) -> torch.Tensor:
    """Append the third row [0, 0, 1] to build a full 3x3 homography."""
    if h_2x3.dim() != 3 or h_2x3.size(1) != 2 or h_2x3.size(2) != 3:
        raise ValueError("Expected homography with shape (B, 2, 3)")

    batch_size = h_2x3.size(0)
    device = h_2x3.device
    bottom_row = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=h_2x3.dtype)
    bottom_row = bottom_row.view(1, 1, 3).expand(batch_size, -1, -1)
    return torch.cat([h_2x3, bottom_row], dim=1)


def warp_feature_map(feature_map: torch.Tensor, h_inv: torch.Tensor) -> torch.Tensor:
    """Warp feature map according to inverse homography."""
    return F.grid_sample(
        feature_map,
        F.affine_grid(h_inv, feature_map.size(), align_corners=False),
        align_corners=False,
    )


def compute_detection_loss(
    det_original: torch.Tensor,
    det_rotated: torch.Tensor,
    h: torch.Tensor,
    smooth_l1_weight: float = DET_SMOOTH_L1_WEIGHT,
) -> torch.Tensor:
    """
    二元交叉熵 + smooth L1 检测损失。

    Args:
        det_original: 原始图像检测图，形状为 (B, 1, H, W)
        det_rotated: 变换后图像检测图，形状为 (B, 1, H, W)
        h: 单应性矩阵，形状为 (B, 2, 3)
        smooth_l1_weight: smooth L1 损失权重，默认使用 DET_SMOOTH_L1_WEIGHT

    Returns:
        检测损失张量
    """
    h_full = _augment_homography_matrix(h)
    # 使用 torch.linalg.inv 替代 torch.inverse，更稳定
    # 添加数值稳定性保护
    try:
        h_inv = torch.linalg.inv(h_full)[:, :2, :]
    except RuntimeError:
        # 如果矩阵奇异，使用伪逆作为回退
        h_inv = torch.linalg.pinv(h_full)[:, :2, :]
    warped_det = warp_feature_map(det_rotated, h_inv)

    bce_loss = F.binary_cross_entropy(det_original, warped_det)
    smooth_l1_loss = F.smooth_l1_loss(det_original, warped_det)
    return bce_loss + smooth_l1_weight * smooth_l1_loss


def compute_description_loss(
    desc_original: torch.Tensor,
    desc_rotated: torch.Tensor,
    h: torch.Tensor,
    margin: float = 1.0,
    num_samples: int = 200,
    geometric_weight: float = DESC_GEOMETRIC_WEIGHT,
    manhattan_weight: float = DESC_MANHATTAN_WEIGHT,
    sparsity_weight: float = DESC_SPARSITY_WEIGHT,
    binary_weight: float = DESC_BINARY_WEIGHT,
) -> torch.Tensor:
    """
    Triplet-style descriptor loss with Manhattan-aware sampling.

    Args:
        desc_original: 原始图像描述子，形状为 (B, C, H, W)
        desc_rotated: 变换后图像描述子，形状为 (B, C, H, W)
        h: 单应性矩阵，形状为 (B, 2, 3)
        margin: triplet loss 的边界值
        num_samples: 曼哈顿采样点数量，默认 200。对于大特征图可适当增加
        geometric_weight: 几何 triplet 损失权重
        manhattan_weight: 曼哈顿损失权重
        sparsity_weight: 稀疏性损失权重
        binary_weight: 二值化损失权重

    Returns:
        描述子损失张量
    """
    batch_size, channels, height, width = desc_original.size()

    grid_side = int(math.sqrt(num_samples))
    h_coords = torch.linspace(-1, 1, grid_side, device=desc_original.device)
    w_coords = torch.linspace(-1, 1, grid_side, device=desc_original.device)

    manhattan_h = torch.cat([h_coords, torch.zeros_like(h_coords)])
    manhattan_w = torch.cat([torch.zeros_like(w_coords), w_coords])
    manhattan_coords = torch.stack([manhattan_h, manhattan_w], dim=1)
    manhattan_coords = manhattan_coords.unsqueeze(0).repeat(batch_size, 1, 1)

    anchor = F.grid_sample(
        desc_original,
        manhattan_coords.unsqueeze(1),
        align_corners=False,
    ).squeeze(2).transpose(1, 2)

    coords_hom = torch.cat(
        [manhattan_coords, torch.ones(batch_size, manhattan_coords.size(1), 1, device=desc_original.device)],
        dim=2,
    )

    h_full = _augment_homography_matrix(h)
    # 使用 torch.linalg.inv 替代 torch.inverse，更稳定
    try:
        h_inv = torch.linalg.inv(h_full)
    except RuntimeError:
        # 如果矩阵奇异，使用伪逆作为回退
        h_inv = torch.linalg.pinv(h_full)
    coords_transformed = (coords_hom @ h_inv.transpose(1, 2))[:, :, :2]

    positive = F.grid_sample(
        desc_rotated,
        coords_transformed.unsqueeze(1),
        align_corners=False,
    ).squeeze(2).transpose(1, 2)

    negative_list = []
    if manhattan_coords.size(1) > 0:
        # 使用预计算的旋转矩阵，避免重复创建张量
        for angle in [90, 180, 270]:
            rot = _get_rotation_matrix(angle, desc_original.device)
            rotated_coords = manhattan_coords @ rot.T
            negative_list.append(rotated_coords)

    if negative_list:
        neg_coords = torch.stack(negative_list, dim=1).reshape(batch_size, -1, 2)
        negative_candidates = F.grid_sample(
            desc_rotated,
            neg_coords.unsqueeze(1),
            align_corners=False,
        ).squeeze(2).transpose(1, 2)

        anchor_expanded = anchor.unsqueeze(2).expand(-1, -1, negative_candidates.size(1), -1)
        negative_expanded = negative_candidates.unsqueeze(1).expand(-1, anchor.size(1), -1, -1)
        manhattan_dist = torch.sum(torch.abs(anchor_expanded - negative_expanded), dim=3)

        k = max(anchor.size(1) // 2, 1)
        hard_indices = torch.topk(manhattan_dist, k=k, largest=False)[1]
        idx_expand = hard_indices.unsqueeze(-1).expand(-1, -1, -1, negative_candidates.size(2))
        negative = torch.gather(negative_candidates.unsqueeze(1).expand(-1, anchor.size(1), -1, -1), 2, idx_expand)
        negative = negative.mean(dim=2)
    else:
        negative = torch.zeros_like(anchor)

    triplet_loss = nn.TripletMarginLoss(margin=margin, p=1, reduction='mean')
    geometric_triplet = triplet_loss(anchor, positive, negative)

    manhattan_loss = 0.0
    for i in range(anchor.size(1)):
        anchor_norm = F.normalize(anchor[:, i], p=2, dim=1)
        positive_norm = F.normalize(positive[:, i], p=2, dim=1)
        cos_sim = torch.sum(anchor_norm * positive_norm, dim=1)
        manhattan_loss += torch.mean(1 - cos_sim)

    manhattan_loss = manhattan_loss / max(anchor.size(1), 1)
    sparsity_loss = torch.mean(torch.abs(anchor)) + torch.mean(torch.abs(positive))
    binary_loss = torch.mean(torch.abs(torch.sign(anchor) - torch.sign(positive)))

    return (
        geometric_weight * geometric_triplet
        + manhattan_weight * manhattan_loss
        + sparsity_weight * sparsity_loss
        + binary_weight * binary_loss
    )
