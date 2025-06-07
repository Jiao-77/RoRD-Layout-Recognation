import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from models.rord import RoRD

# 数据集类：生成随机旋转的训练对
class ICLayoutTrainingDataset(Dataset):
    def __init__(self, image_dir, patch_size=256, transform=None):
        """
        初始化 IC 版图训练数据集。

        参数：
            image_dir (str): 存储 PNG 格式 IC 版图图像的目录路径。
            patch_size (int): 裁剪的 patch 大小（默认 256x256）。
            transform (callable, optional): 应用于图像的变换。
        """
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        """
        返回数据集中的图像数量。

        返回：
            int: 数据集大小。
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        获取指定索引的训练对（原始 patch、旋转 patch、Homography 矩阵）。

        参数：
            index (int): 图像索引。

        返回：
            tuple: (patch, rotated_patch, H_tensor)
                - patch: 原始 patch 张量。
                - rotated_patch: 旋转后的 patch 张量。
                - H_tensor: Homography 矩阵张量。
        """
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('L')  # 灰度图像

        # 获取图像大小
        W, H = image.size

        # 随机选择裁剪的左上角坐标
        x = np.random.randint(0, W - self.patch_size + 1)
        y = np.random.randint(0, H - self.patch_size + 1)
        patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))

        # 转换为 NumPy 数组
        patch_np = np.array(patch)

        # 随机旋转角度（0°~360°）
        theta = np.random.uniform(0, 360)
        theta_rad = np.deg2rad(theta)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        # 计算旋转中心（patch 的中心）
        cx = self.patch_size / 2.0
        cy = self.patch_size / 2.0

        # 计算旋转的齐次矩阵（Homography）
        H = np.array([
            [cos_theta, -sin_theta, cx * (1 - cos_theta) + cy * sin_theta],
            [sin_theta,  cos_theta, cy * (1 - cos_theta) - cx * sin_theta],
            [0,          0,          1]
        ], dtype=np.float32)

        # 应用旋转到 patch
        rotated_patch_np = cv2.warpPerspective(patch_np, H, (self.patch_size, self.patch_size))

        # 转换回 PIL Image
        rotated_patch = Image.fromarray(rotated_patch_np)

        # 应用变换
        if self.transform:
            patch = self.transform(patch)
            rotated_patch = self.transform(rotated_patch)

        # 转换 H 为张量
        H_tensor = torch.from_numpy(H).float()

        return patch, rotated_patch, H_tensor

# 特征图变换函数
def warp_feature_map(feature_map, H_inv):
    """
    使用逆 Homography 矩阵变换特征图。

    参数：
        feature_map (torch.Tensor): 输入特征图，形状为 [B, C, H, W]。
        H_inv (torch.Tensor): 逆 Homography 矩阵，形状为 [B, 3, 3]。

    返回：
        torch.Tensor: 变换后的特征图，形状为 [B, C, H, W]。
    """
    B, C, H, W = feature_map.size()
    # 生成网格
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=feature_map.device),
        torch.linspace(-1, 1, W, device=feature_map.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)  # [H, W, 3]
    grid = grid.unsqueeze(0).expand(B, H, W, 3)  # [B, H, W, 3]

    # 将网格转换为齐次坐标并应用 H_inv
    grid_flat = grid.view(B, -1, 3)  # [B, H*W, 3]
    grid_transformed = torch.bmm(grid_flat, H_inv.transpose(1, 2))  # [B, H*W, 3]
    grid_transformed = grid_transformed.view(B, H, W, 3)  # [B, H, W, 3]
    grid_transformed = grid_transformed[..., :2] / (grid_transformed[..., 2:3] + 1e-8)  # [B, H, W, 2]

    # 使用 grid_sample 进行变换
    warped_feature = F.grid_sample(feature_map, grid_transformed, align_corners=True)
    return warped_feature

# 检测损失函数
def compute_detection_loss(det_original, det_rotated, H):
    """
    计算检测损失（MSE），比较原始检测图与旋转检测图（逆变换后）。

    参数：
        det_original (torch.Tensor): 原始图像的检测图，形状为 [B, 1, H, W]。
        det_rotated (torch.Tensor): 旋转图像的检测图，形状为 [B, 1, H, W]。
        H (torch.Tensor): Homography 矩阵，形状为 [B, 3, 3]。

    返回：
        torch.Tensor: 检测损失。
    """
    H_inv = torch.inverse(H)  # 计算逆 Homography
    warped_det_rotated = warp_feature_map(det_rotated, H_inv)
    return F.mse_loss(det_original, warped_det_rotated)

# 描述子损失函数
def compute_description_loss(desc_original, desc_rotated, H, margin=1.0):
    """
    计算描述子损失（三元组损失），基于对应点的描述子。

    参数：
        desc_original (torch.Tensor): 原始图像的描述子图，形状为 [B, 128, H, W]。
        desc_rotated (torch.Tensor): 旋转图像的描述子图，形状为 [B, 128, H, W]。
        H (torch.Tensor): Homography 矩阵，形状为 [B, 3, 3]。
        margin (float): 三元组损失的边距。

    返回：
        torch.Tensor: 描述子损失。
    """
    B, C, H, W = desc_original.size()
    # 随机选择锚点（anchor）
    num_samples = min(100, H * W)  # 每张图像采样 100 个点
    idx = torch.randint(0, H * W, (B, num_samples), device=desc_original.device)
    idx_y = idx // W
    idx_x = idx % W
    coords = torch.stack((idx_x.float(), idx_y.float()), dim=-1)  # [B, num_samples, 2]

    # 转换为齐次坐标
    coords_hom = torch.cat((coords, torch.ones(B, num_samples, 1, device=coords.device)), dim=-1)  # [B, num_samples, 3]
    coords_transformed = torch.bmm(coords_hom, H.transpose(1, 2))  # [B, num_samples, 3]
    coords_transformed = coords_transformed[..., :2] / (coords_transformed[..., 2:3] + 1e-8)  # [B, num_samples, 2]

    # 归一化到 [-1, 1] 用于 grid_sample
    coords_transformed = coords_transformed / torch.tensor([W/2, H/2], device=coords.device) - 1

    # 提取锚点和正样本描述子
    anchor = desc_original.view(B, C, -1)[:, :, idx.view(-1)]  # [B, 128, num_samples]
    positive = F.grid_sample(desc_rotated, coords_transformed.unsqueeze(2), align_corners=True).squeeze(3)  # [B, 128, num_samples]

    # 随机选择负样本
    neg_idx = torch.randint(0, H * W, (B, num_samples), device=desc_original.device)
    negative = desc_rotated.view(B, C, -1)[:, :, neg_idx.view(-1)]  # [B, 128, num_samples]

    # 三元组损失
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    loss = triplet_loss(anchor.transpose(1, 2), positive.transpose(1, 2), negative.transpose(1, 2))
    return loss

# 定义变换
transform = transforms.Compose([
    transforms.ToTensor(),  # (1, 256, 256)
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # (3, 256, 256)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 创建数据集和 DataLoader
dataset = ICLayoutTrainingDataset('path/to/layouts', patch_size=256, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# 定义模型
model = RoRD().cuda()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        original, rotated, H = batch
        original = original.cuda()
        rotated = rotated.cuda()
        H = H.cuda()

        # 前向传播
        det_original, _, desc_rord_original = model(original)
        det_rotated, _, desc_rord_rotated = model(rotated)

        # 计算损失
        detection_loss = compute_detection_loss(det_original, det_rotated, H)
        description_loss = compute_description_loss(desc_rord_original, desc_rord_rotated, H)
        total_loss_batch = detection_loss + description_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

# 保存模型
torch.save(model.state_dict(), 'path/to/save/model.pth')