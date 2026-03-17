# models/rord.py

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入新配置系统
try:
    from utils.config import ModelConfig, BackboneConfig, AttentionConfig, FPNConfig
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    ModelConfig = None
    BackboneConfig = None
    AttentionConfig = None
    FPNConfig = None

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
    def __init__(
        self,
        model_config: Optional["ModelConfig"] = None,
        # 以下参数保留用于向后兼容，但建议使用 model_config
        fpn_out_channels: int = 256,
        fpn_levels: Tuple[int, ...] = (2, 3, 4),
        cfg=None,  # 废弃：请使用 model_config
    ):
        """
        修复后的 RoRD 模型。
        
        支持两种配置方式：
        1. 推荐方式：使用 ModelConfig 对象
           ```python
           from utils.config import ModelConfig, BackboneConfig
           config = ModelConfig(backbone=BackboneConfig(name="resnet34"))
           model = RoRD(model_config=config)
           ```
        
        2. 向后兼容方式：使用参数或旧版 cfg 对象
           ```python
           model = RoRD(fpn_out_channels=256, fpn_levels=(2, 3, 4))
           ```
        
        Args:
            model_config: 模型配置对象（推荐）
            fpn_out_channels: FPN 输出通道数（向后兼容）
            fpn_levels: FPN 层级（向后兼容）
            cfg: 旧版配置对象（已废弃）
        """
        super(RoRD, self).__init__()

        # 解析配置
        if model_config is not None:
            # 新配置系统
            backbone_name = model_config.backbone.name
            pretrained = model_config.backbone.pretrained
            attn_enabled = model_config.attention.enabled
            attn_type = model_config.attention.type
            attn_places = model_config.attention.places
            attn_reduction = model_config.attention.reduction
            attn_spatial_kernel = model_config.attention.spatial_kernel
            fpn_enabled = model_config.fpn.enabled
            fpn_out = model_config.fpn.out_channels
            fpn_lvls = model_config.fpn.levels
        elif cfg is not None:
            # 旧配置系统（向后兼容）
            logger.warning("使用废弃的 cfg 参数，建议迁移到 model_config")
            backbone_name = "vgg16"
            pretrained = False
            attn_enabled = False
            attn_type = "none"
            attn_places = []
            attn_reduction = 16
            attn_spatial_kernel = 7
            fpn_enabled = True
            fpn_out = fpn_out_channels
            fpn_lvls = fpn_levels
            
            try:
                if hasattr(cfg, 'model'):
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
                    if hasattr(m, 'fpn'):
                        fpn_enabled = bool(getattr(m.fpn, 'enabled', fpn_enabled))
                        fpn_out = int(getattr(m.fpn, 'out_channels', fpn_out))
                        levels = getattr(m.fpn, 'levels', list(fpn_lvls))
                        fpn_lvls = tuple(levels) if levels else fpn_lvls
            except (AttributeError, KeyError, TypeError) as e:
                logger.debug(f"配置解析使用默认值: {e}")
        else:
            # 使用默认值或参数
            backbone_name = "vgg16"
            pretrained = False
            attn_enabled = False
            attn_type = "none"
            attn_places = []
            attn_reduction = 16
            attn_spatial_kernel = 7
            fpn_enabled = True
            fpn_out = fpn_out_channels
            fpn_lvls = fpn_levels

        # 构建骨干
        self.backbone_name = backbone_name
        out_channels_backbone = 512
        # 默认各层通道（VGG 对齐）
        c2_ch, c3_ch, c4_ch = 128, 256, 512
        if backbone_name == "resnet34":
            # 构建骨干并按需手动加载权重，便于打印加载摘要
            if pretrained:
                res = models.resnet34(weights=None)
                self._summarize_pretrained_load(res, models.ResNet34_Weights.DEFAULT, "resnet34")
            else:
                res = models.resnet34(weights=None)
            # 修改第一层卷积以支持 1 通道灰度输入
            old_conv1 = res.conv1
            res.conv1 = nn.Conv2d(
                1, 64, kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride, padding=old_conv1.padding,
                bias=old_conv1.bias is not None,
                dilation=old_conv1.dilation, groups=old_conv1.groups
            )
            # 复制权重：将 3 通道的权重沿通道维度平均，转换为 1 通道
            with torch.no_grad():
                res.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
                if old_conv1.bias is not None:
                    res.conv1.bias.copy_(old_conv1.bias)
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
            if pretrained:
                eff = models.efficientnet_b0(weights=None)
                self._summarize_pretrained_load(eff, models.EfficientNet_B0_Weights.DEFAULT, "efficientnet_b0")
            else:
                eff = models.efficientnet_b0(weights=None)
            # 修改第一层卷积以支持 1 通道灰度输入
            old_conv1 = eff.features[0][0]
            eff.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride, padding=old_conv1.padding,
                bias=old_conv1.bias is not None,
                dilation=old_conv1.dilation, groups=old_conv1.groups
            )
            # 复制权重：将 3 通道的权重沿通道维度平均，转换为 1 通道
            with torch.no_grad():
                eff.features[0][0].weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
                if old_conv1.bias is not None:
                    eff.features[0][0].bias.copy_(old_conv1.bias)
            self.backbone = eff.features
            self._backbone_raw = eff
            out_channels_backbone = 1280
            # 选择 features[2]/[3]/[6] 作为 C2/C3/C4（约 24/40/192）
            c2_ch, c3_ch, c4_ch = 24, 40, 192
        else:
            if pretrained:
                vgg = models.vgg16(weights=None)
                self._summarize_pretrained_load(vgg, models.VGG16_Weights.DEFAULT, "vgg16")
            else:
                vgg = models.vgg16(weights=None)
            vgg16_features = vgg.features
            # VGG16 特征各阶段索引（conv & relu 层序列）
            # relu2_2 索引 8，relu3_3 索引 15，relu4_3 索引 22
            self.features = vgg16_features
            # 共享骨干（向后兼容单尺度路径，使用到 relu4_3）
            backbone_layers = list(vgg16_features.children())[:23]
            # 修改第一层卷积以支持 1 通道灰度输入
            old_conv1 = backbone_layers[0]
            backbone_layers[0] = nn.Conv2d(
                1, 64, kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride, padding=old_conv1.padding,
                bias=old_conv1.bias is not None
            )
            # 复制权重：将 3 通道的权重沿通道维度平均，转换为 1 通道
            with torch.no_grad():
                backbone_layers[0].weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
                if old_conv1.bias is not None:
                    backbone_layers[0].bias.copy_(old_conv1.bias)
            self.backbone = nn.Sequential(*backbone_layers)
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
        self.fpn_out_channels = fpn_out
        self.fpn_enabled = fpn_enabled

        # 验证 FPN 层级配置
        valid_levels = {2, 3, 4}
        fpn_levels_set = set(fpn_lvls)
        invalid_levels = fpn_levels_set - valid_levels
        if invalid_levels:
            raise ValueError(
                f"FPN 层级必须是 2, 3, 4 的子集，但收到了无效层级: {invalid_levels}。"
                f"有效层级说明: P2(高分辨率), P3(中分辨率), P4(低分辨率)"
            )

        # 排序以确保一致的输出顺序
        # 注意: FPN 内部构建始终是自顶向下 (P4->P3->P2)，与输入顺序无关
        self.fpn_levels = tuple(sorted(fpn_levels_set))  # e.g., (2,3,4)

        # 横向连接 1x1：根据骨干动态对齐到相同通道数
        self.lateral_c2 = nn.Conv2d(c2_ch, fpn_out, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(c3_ch, fpn_out, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(c4_ch, fpn_out, kernel_size=1)

        # 平滑 3x3 conv
        self.smooth_p2 = nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)

        # 共享的 FPN 检测/描述子头（输入通道为 fpn_out_channels）
        self.det_head_fpn = nn.Sequential(
            nn.Conv2d(fpn_out, 128, kernel_size=3, padding=1),
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
            # 使用 self.backbone（已修改为 1 通道）而不是 self.features
            for i, layer in enumerate(self.backbone):
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

    # --- Utils ---
    def _summarize_pretrained_load(self, torch_model: nn.Module, weights_enum, arch_name: str) -> None:
        """手动加载 torchvision 预训练权重并打印加载摘要。
        - 使用 strict=False 以兼容可能的键差异，打印 missing/unexpected keys。
        - 输出参数量统计，便于快速核对加载情况。
        """
        try:
            state_dict = weights_enum.get_state_dict(progress=False)
        except (AttributeError, NotImplementedError, RuntimeError) as e:
            # 回退：若权重枚举不支持 get_state_dict，则跳过摘要（通常已在构造器中加载）
            logger.debug(f"[Pretrained] {arch_name}: skip summary - {e}")
            return
        incompatible = torch_model.load_state_dict(state_dict, strict=False)
        total_params = sum(p.numel() for p in torch_model.parameters())
        trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
        missing = list(getattr(incompatible, 'missing_keys', []))
        unexpected = list(getattr(incompatible, 'unexpected_keys', []))
        try:
            matched = len(state_dict) - len(unexpected)
        except (TypeError, ValueError):
            matched = 0
        print(f"[Pretrained] {arch_name}: ImageNet weights loaded (strict=False)")
        print(f"  params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")
        print(f"  keys: matched≈{matched} | missing={len(missing)} | unexpected={len(unexpected)}")
        if missing and len(missing) <= 10:
            print(f"  missing: {missing}")
        if unexpected and len(unexpected) <= 10:
            print(f"  unexpected: {unexpected}")