# models/rord.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- Optional Attention Modules (default disabled) ---
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        # Channel attention (MLP on pooled features)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        # Spatial attention
        padding = spatial_kernel // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = torch.mean(x, dim=(2, 3))
        mx, _ = torch.max(torch.max(x, dim=2).values, dim=2)
        ch = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        ch = ch.view(b, c, 1, 1)
        x = x * ch
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * attn

class RoRD(nn.Module):
    def __init__(self, fpn_out_channels: int = 256, fpn_levels=(2, 3, 4), cfg=None):
        """
        修复后的 RoRD 模型。
        - 实现了共享骨干网络，以提高计算效率和减少内存占用。
        - 确保检测头和描述子头使用相同尺寸的特征图。
        - 新增（可选）FPN 推理路径，提供多尺度特征用于高效匹配。
        """
        super(RoRD, self).__init__()

        # 解析可选配置（保持全部默认关闭）
        backbone_name = "vgg16"
        pretrained = False
        attn_enabled = False
        attn_type = "none"
        attn_places = []
        attn_reduction = 16
        attn_spatial_kernel = 7
        try:
            if cfg is not None and hasattr(cfg, 'model'):
                m = cfg.model
                if hasattr(m, 'backbone'):
                    backbone_name = str(getattr(m.backbone, 'name', backbone_name))
                    pretrained = bool(getattr(m.backbone, 'pretrained', pretrained))
                if hasattr(m, 'attention'):
                    attn_enabled = bool(getattr(m.attention, 'enabled', attn_enabled))
                    attn_type = str(getattr(m.attention, 'type', attn_type))
                    attn_places = list(getattr(m.attention, 'places', attn_places))
                    attn_reduction = int(getattr(m.attention, 'reduction', attn_reduction))
                    attn_spatial_kernel = int(getattr(m.attention, 'spatial_kernel', attn_spatial_kernel))
        except Exception:
            # 配置非标准时，保留默认
            pass

        # 构建骨干
        self.backbone_name = backbone_name
        out_channels_backbone = 512
        # 默认各层通道（VGG 对齐）
        c2_ch, c3_ch, c4_ch = 128, 256, 512
        if backbone_name == "resnet34":
            res = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.backbone = nn.Sequential(
                res.conv1, res.bn1, res.relu, res.maxpool,
                res.layer1, res.layer2, res.layer3, res.layer4,
            )
            # 记录原始模型以备进一步扩展（如中间层 hook）
            self._backbone_raw = res
            out_channels_backbone = 512
            # 选择 layer2/layer3/layer4 作为 C2/C3/C4
            c2_ch, c3_ch, c4_ch = 128, 256, 512
        elif backbone_name == "efficientnet_b0":
            eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.backbone = eff.features
            self._backbone_raw = eff
            out_channels_backbone = 1280
            # 选择 features[2]/[3]/[6] 作为 C2/C3/C4（约 24/40/192）
            c2_ch, c3_ch, c4_ch = 24, 40, 192
        else:
            vgg16_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None).features
            # VGG16 特征各阶段索引（conv & relu 层序列）
            # relu2_2 索引 8，relu3_3 索引 15，relu4_3 索引 22
            self.features = vgg16_features
            # 共享骨干（向后兼容单尺度路径，使用到 relu4_3）
            self.backbone = nn.Sequential(*list(vgg16_features.children())[:23])
            out_channels_backbone = 512
            c2_ch, c3_ch, c4_ch = 128, 256, 512

        # 非 VGG 情况下，确保属性存在（供 _extract_c234 判断）
        if backbone_name != "vgg16":
            self.features = None

        # 检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(out_channels_backbone, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 描述子头
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(out_channels_backbone, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.InstanceNorm2d(128)
        )

        # 注意力包装（默认关闭）
        def make_attn_layer(in_channels: int) -> nn.Module:
            if not attn_enabled or attn_type == "none":
                return nn.Identity()
            if attn_type == "cbam":
                return CBAM(in_channels, reduction=attn_reduction, spatial_kernel=attn_spatial_kernel)
            return SEBlock(in_channels, reduction=attn_reduction)

        self._attn_backbone_high = make_attn_layer(out_channels_backbone) if "backbone_high" in attn_places else nn.Identity()
        if "det_head" in attn_places:
            self.detection_head = nn.Sequential(make_attn_layer(out_channels_backbone), *list(self.detection_head.children()))
        if "desc_head" in attn_places:
            self.descriptor_head = nn.Sequential(make_attn_layer(out_channels_backbone), *list(self.descriptor_head.children()))

        # --- FPN 组件（用于可选多尺度推理） ---
        self.fpn_out_channels = fpn_out_channels
        self.fpn_levels = tuple(sorted(set(fpn_levels)))  # e.g., (2,3,4)

        # 横向连接 1x1：根据骨干动态对齐到相同通道数
        self.lateral_c2 = nn.Conv2d(c2_ch, fpn_out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(c3_ch, fpn_out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(c4_ch, fpn_out_channels, kernel_size=1)

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
            # 可选：骨干高层注意力
            features = self._attn_backbone_high(features)
            detection_map = self.detection_head(features)
            descriptors = self.descriptor_head(features)
            return detection_map, descriptors

        # --- FPN 路径：提取 C2/C3/C4 ---
        c2, c3, c4 = self._extract_c234(x)
        # 根据骨干设置各层对应的下采样步幅（相对输入）
        if self.backbone_name == "vgg16":
            s2, s3, s4 = 2, 4, 8
        elif self.backbone_name == "resnet34":
            s2, s3, s4 = 8, 16, 32
        elif self.backbone_name == "efficientnet_b0":
            s2, s3, s4 = 4, 8, 32
        else:
            s2 = s3 = s4 = 8  # 合理保守默认
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)

        pyramid = {}
        if 4 in self.fpn_levels:
            pyramid["P4"] = (self.det_head_fpn(p4), self.desc_head_fpn(p4), s4)
        if 3 in self.fpn_levels:
            pyramid["P3"] = (self.det_head_fpn(p3), self.desc_head_fpn(p3), s3)
        if 2 in self.fpn_levels:
            pyramid["P2"] = (self.det_head_fpn(p2), self.desc_head_fpn(p2), s2)
        return pyramid

    def _extract_c234(self, x: torch.Tensor):
        """提取中间层特征 C2/C3/C4，适配不同骨干。"""
        if self.backbone_name == "vgg16":
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

        if self.backbone_name == "resnet34":
            res = self._backbone_raw
            x = res.conv1(x)
            x = res.bn1(x)
            x = res.relu(x)
            x = res.maxpool(x)
            x = res.layer1(x)
            c2 = res.layer2(x)  # 128
            c3 = res.layer3(c2)  # 256
            c4 = res.layer4(c3)  # 512
            return c2, c3, c4

        if self.backbone_name == "efficientnet_b0":
            # 取 features[2]/[3]/[6] 作为 C2/C3/C4
            feats = self._backbone_raw.features
            c2 = c3 = c4 = None
            x = feats[0](x)  # stem
            x = feats[1](x)
            x = feats[2](x); c2 = x
            x = feats[3](x); c3 = x
            x = feats[4](x)
            x = feats[5](x)
            x = feats[6](x); c4 = x
            return c2, c3, c4

        raise RuntimeError(f"Unsupported backbone for FPN: {self.backbone_name}")