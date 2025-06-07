import torch
import torch.nn as nn
from torchvision import models

class RoRD(nn.Module):
    def __init__(self):
        super(RoRD, self).__init__()
        # 检测骨干网络：VGG-16 直到 relu5_3（层 0 到 29）
        self.backbone_det = models.vgg16(pretrained=True).features[:30]
        # 描述骨干网络：VGG-16 直到 relu4_3（层 0 到 22）
        self.backbone_desc = models.vgg16(pretrained=True).features[:23]
        
        # 检测头：输出关键点概率图
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 普通描述子头（D2-Net 风格）
        self.descriptor_head_vanilla = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.InstanceNorm2d(128)
        )
        
        # RoRD 描述子头（旋转鲁棒）
        self.descriptor_head_rord = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.InstanceNorm2d(128)
        )

    def forward(self, x):
        # 检测分支
        features_det = self.backbone_det(x)
        detection = self.detection_head(features_det)
        
        # 描述分支
        features_desc = self.backbone_desc(x)
        desc_vanilla = self.descriptor_head_vanilla(features_desc)
        desc_rord = self.descriptor_head_rord(features_desc)
        
        return detection, desc_vanilla, desc_rord