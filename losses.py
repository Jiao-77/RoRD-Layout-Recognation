"""Loss utilities for RoRD training."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
) -> torch.Tensor:
    """Binary cross-entropy + smooth L1 detection loss."""
    h_full = _augment_homography_matrix(h)
    h_inv = torch.inverse(h_full)[:, :2, :]
    warped_det = warp_feature_map(det_rotated, h_inv)

    bce_loss = F.binary_cross_entropy(det_original, warped_det)
    smooth_l1_loss = F.smooth_l1_loss(det_original, warped_det)
    return bce_loss + 0.1 * smooth_l1_loss


def compute_description_loss(
    desc_original: torch.Tensor,
    desc_rotated: torch.Tensor,
    h: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet-style descriptor loss with Manhattan-aware sampling."""
    batch_size, channels, height, width = desc_original.size()
    num_samples = 200

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
    h_inv = torch.inverse(h_full)
    coords_transformed = (coords_hom @ h_inv.transpose(1, 2))[:, :, :2]

    positive = F.grid_sample(
        desc_rotated,
        coords_transformed.unsqueeze(1),
        align_corners=False,
    ).squeeze(2).transpose(1, 2)

    negative_list = []
    if manhattan_coords.size(1) > 0:
        angles = [0, 90, 180, 270]
        for angle in angles:
            if angle == 0:
                continue
            theta = torch.tensor(angle * math.pi / 180.0, device=desc_original.device)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            rot = torch.stack(
                [
                    torch.stack([cos_t, -sin_t]),
                    torch.stack([sin_t, cos_t]),
                ]
            )
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

    return geometric_triplet + 0.1 * manhattan_loss + 0.01 * sparsity_loss + 0.05 * binary_loss
