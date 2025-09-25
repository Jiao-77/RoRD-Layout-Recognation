# models/rord.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class RoRD(nn.Module):
    def __init__(self, fpn_out_channels: int = 256, fpn_levels=(2, 3, 4)):
        """
        修复后的 RoRD 模型。
        - 实现了共享骨干网络，以提高计算效率和减少内存占用。
        - 确保检测头和描述子头使用相同尺寸的特征图。
        - 新增（可选）FPN 推理路径，提供多尺度特征用于高效匹配。
        """
        super(RoRD, self).__init__()
        
        vgg16_features = models.vgg16(pretrained=False).features

        # VGG16 特征各阶段索引（conv & relu 层序列）
        # relu2_2 索引 8，relu3_3 索引 15，relu4_3 索引 22
        self.features = vgg16_features

        # 共享骨干（向后兼容单尺度路径，使用到 relu4_3）
        self.backbone = nn.Sequential(*list(vgg16_features.children())[:23])

        # 检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 描述子头
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.InstanceNorm2d(128)
        )

        # --- FPN 组件（用于可选多尺度推理） ---
        self.fpn_out_channels = fpn_out_channels
        self.fpn_levels = tuple(sorted(set(fpn_levels)))  # e.g., (2,3,4)

        # 横向连接 1x1 将 C2(128)/C3(256)/C4(512) 对齐到相同通道数
        self.lateral_c2 = nn.Conv2d(128, fpn_out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(256, fpn_out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(512, fpn_out_channels, kernel_size=1)

        # 平滑 3x3 conv
        self.smooth_p2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)

        # 共享的 FPN 检测/描述子头（输入通道为 fpn_out_channels）
        self.det_head_fpn = nn.Sequential(
            nn.Conv2d(fpn_out_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.desc_head_fpn = nn.Sequential(
            nn.Conv2d(fpn_out_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.InstanceNorm2d(128),
        )

    def forward(self, x: torch.Tensor, return_pyramid: bool = False):
        if not return_pyramid:
            # 向后兼容的单尺度路径（relu4_3）
            features = self.backbone(x)
            detection_map = self.detection_head(features)
            descriptors = self.descriptor_head(features)
            return detection_map, descriptors

        # --- FPN 路径：提取 C2/C3/C4 ---
        c2, c3, c4 = self._extract_c234(x)
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)

        pyramid = {}
        if 4 in self.fpn_levels:
            pyramid["P4"] = (self.det_head_fpn(p4), self.desc_head_fpn(p4), 8)
        if 3 in self.fpn_levels:
            pyramid["P3"] = (self.det_head_fpn(p3), self.desc_head_fpn(p3), 4)
        if 2 in self.fpn_levels:
            pyramid["P2"] = (self.det_head_fpn(p2), self.desc_head_fpn(p2), 2)
        return pyramid

    def _extract_c234(self, x: torch.Tensor):
        """提取 VGG 中间层特征：C2(relU2_2), C3(relu3_3), C4(relu4_3)."""
        c2 = c3 = c4 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 8:   # relu2_2
                c2 = x
            elif i == 15:  # relu3_3
                c3 = x
            elif i == 22:  # relu4_3
                c4 = x
                break
        assert c2 is not None and c3 is not None and c4 is not None
        return c2, c3, c4