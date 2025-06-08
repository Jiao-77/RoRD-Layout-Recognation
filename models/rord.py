# models/rord.py

import torch
import torch.nn as nn
from torchvision import models

class RoRD(nn.Module):
    def __init__(self):
        """
        修复后的 RoRD 模型。
        - 实现了共享骨干网络，以提高计算效率和减少内存占用。
        - 移除了冗余的 descriptor_head_vanilla。
        """
        super(RoRD, self).__init__()
        
        vgg16_features = models.vgg16(pretrained=True).features
        
        # 共享骨干网络
        self.slice1 = vgg16_features[:23]  # 到 relu4_3
        self.slice2 = vgg16_features[23:30] # 从 relu4_3 到 relu5_3

        # 检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 描述子头
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.InstanceNorm2d(128)
        )

    def forward(self, x):
        # 共享特征提取
        features_shared = self.slice1(x)
        
        # 描述子分支
        descriptors = self.descriptor_head(features_shared)
        
        # 检测器分支
        features_det = self.slice2(features_shared)
        detection_map = self.detection_head(features_det)
        
        return detection_map, descriptors