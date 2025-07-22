# models/rord.py

import torch
import torch.nn as nn
from torchvision import models

class RoRD(nn.Module):
    def __init__(self):
        """
        Repaired RoRD model.
        - Implements shared backbone network to improve computational efficiency and reduce memory usage.
        - Ensures detection head and descriptor head use feature maps of the same size.
        """
        super(RoRD, self).__init__()
        
        vgg16_features = models.vgg16(pretrained=False).features
        
        # Shared backbone network - only uses up to relu4_3 to ensure consistent feature map dimensions
        self.backbone = nn.Sequential(*list(vgg16_features.children())[:23])

        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Descriptor head
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.InstanceNorm2d(128)
        )

    def forward(self, x):
        # Shared feature extraction
        features = self.backbone(x)
        
        # Detector and descriptor use the same feature maps
        detection_map = self.detection_head(features)
        descriptors = self.descriptor_head(features)
        
        return detection_map, descriptors