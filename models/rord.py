# models/rord.py

import torch
import torch.nn as nn
from torchvision import models

class RoRD(nn.Module):
    def __init__(self):
        """
        修复后的 RoRD 模型。
        - 实现了共享骨干网络，以提高计算效率和减少内存占用。
        - 确保检测头和描述子头使用相同尺寸的特征图。
        """
        super(RoRD, self).__init__()
        
        vgg16_features = models.vgg16(pretrained=False).features
        
        # 共享骨干网络 - 只使用到 relu4_3，确保特征图尺寸一致
        self.backbone = vgg16_features[:23]  # 到 relu4_3

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

    def forward(self, x):
        # 共享特征提取
        features = self.backbone(x)
        
        # 检测器和描述子使用相同的特征图
        detection_map = self.detection_head(features)
        descriptors = self.descriptor_head(features)
        
        return detection_map, descriptors