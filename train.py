# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import os
import argparse

# 导入项目模块
import config
from models.rord import RoRD
from utils.data_utils import get_transform

# --- 训练专用数据集类 ---
class ICLayoutTrainingDataset(Dataset):
    def __init__(self, image_dir, patch_size=256, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('L')

        W, H = image.size
        x = np.random.randint(0, W - self.patch_size + 1)
        y = np.random.randint(0, H - self.patch_size + 1)
        patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
        patch_np = np.array(patch)
        
        # 实现8个方向的离散几何变换
        theta_deg = np.random.choice([0, 90, 180, 270])
        is_mirrored = np.random.choice([True, False])
        cx, cy = self.patch_size / 2.0, self.patch_size / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), theta_deg, 1)

        if is_mirrored:
            T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
            Flip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
            T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
            M_mirror_3x3 = T2 @ Flip @ T1
            M_3x3 = np.vstack([M, [0, 0, 1]])
            H = (M_3x3 @ M_mirror_3x3).astype(np.float32)
        else:
            H = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        
        transformed_patch_np = cv2.warpPerspective(patch_np, H, (self.patch_size, self.patch_size))
        transformed_patch = Image.fromarray(transformed_patch_np)

        if self.transform:
            patch = self.transform(patch)
            transformed_patch = self.transform(transformed_patch)

        H_tensor = torch.from_numpy(H[:2, :]).float() # 通常损失函数需要2x3的仿射矩阵
        return patch, transformed_patch, H_tensor

# --- 特征图变换与损失函数 ---
def warp_feature_map(feature_map, H_inv):
    B, C, H, W = feature_map.size()
    grid = F.affine_grid(H_inv, feature_map.size(), align_corners=False).to(feature_map.device)
    return F.grid_sample(feature_map, grid, align_corners=False)

def compute_detection_loss(det_original, det_rotated, H):
    with torch.no_grad():
        H_inv = torch.inverse(torch.cat([H, torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).repeat(H.shape[0], 1, 1)], dim=1))[:, :2, :]
    warped_det_rotated = warp_feature_map(det_rotated, H_inv)
    return F.mse_loss(det_original, warped_det_rotated)

def compute_description_loss(desc_original, desc_rotated, H, margin=1.0):
    B, C, H_feat, W_feat = desc_original.size()
    num_samples = 100
    
    # 随机采样锚点坐标
    coords = torch.rand(B, num_samples, 2, device=desc_original.device) * 2 - 1  # [-1, 1]
    
    # 提取锚点描述子
    anchor = F.grid_sample(desc_original, coords.unsqueeze(1), align_corners=False).squeeze(2).transpose(1, 2)
    
    # 计算正样本坐标
    coords_hom = torch.cat([coords, torch.ones(B, num_samples, 1, device=coords.device)], dim=2)
    M_inv = torch.inverse(torch.cat([H, torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).repeat(H.shape[0], 1, 1)], dim=1))
    coords_transformed = (coords_hom @ M_inv.transpose(1, 2))[:, :, :2]
    
    # 提取正样本描述子
    positive = F.grid_sample(desc_rotated, coords_transformed.unsqueeze(1), align_corners=False).squeeze(2).transpose(1, 2)
    
    # 随机采样负样本
    neg_coords = torch.rand(B, num_samples, 2, device=desc_original.device) * 2 - 1
    negative = F.grid_sample(desc_rotated, neg_coords.unsqueeze(1), align_corners=False).squeeze(2).transpose(1, 2)

    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    return triplet_loss(anchor, positive, negative)

# --- 主函数与命令行接口 ---
def main(args):
    print("--- 开始训练 RoRD 模型 ---")
    print(f"训练参数: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}")
    transform = get_transform()
    dataset = ICLayoutTrainingDataset(args.data_dir, patch_size=config.PATCH_SIZE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    model = RoRD().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss_val = 0
        for i, (original, rotated, H) in enumerate(dataloader):
            original, rotated, H = original.cuda(), rotated.cuda(), H.cuda()
            det_original, desc_original = model(original)
            det_rotated, desc_rotated = model(rotated)

            loss = compute_detection_loss(det_original, det_rotated, H) + compute_description_loss(desc_original, desc_rotated, H)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_val += loss.item()

        print(f"--- Epoch {epoch+1} 完成, 平均 Loss: {total_loss_val / len(dataloader):.4f} ---")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, 'rord_model_final.pth')
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 RoRD 模型")
    parser.add_argument('--data_dir', type=str, default=config.LAYOUT_DIR)
    parser.add_argument('--save_dir', type=str, default=config.SAVE_DIR)
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    main(parser.parse_args())