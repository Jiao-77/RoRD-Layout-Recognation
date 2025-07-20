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
import logging
from datetime import datetime

# 导入项目模块
import config
from models.rord import RoRD
from utils.data_utils import get_transform

# 设置日志记录
def setup_logging(save_dir):
    """设置训练日志记录"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# --- (已修改) 训练专用数据集类 ---
class ICLayoutTrainingDataset(Dataset):
    def __init__(self, image_dir, patch_size=256, transform=None, scale_range=(1.0, 1.0)):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.patch_size = patch_size
        self.transform = transform
        self.scale_range = scale_range # 新增尺度范围参数

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('L')
        W, H = image.size

        # --- 新增：尺度抖动数据增强 ---
        # 1. 随机选择一个缩放比例
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        # 2. 根据缩放比例计算需要从原图裁剪的尺寸
        crop_size = int(self.patch_size / scale)

        # 确保裁剪尺寸不超过图像边界
        if crop_size > min(W, H):
            crop_size = min(W, H)
        
        # 3. 随机裁剪
        x = np.random.randint(0, W - crop_size + 1)
        y = np.random.randint(0, H - crop_size + 1)
        patch = image.crop((x, y, x + crop_size, y + crop_size))

        # 4. 将裁剪出的图像块缩放回标准的 patch_size
        patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
        # --- 尺度抖动结束 ---

        # --- 新增：额外的数据增强 ---
        # 亮度调整
        if np.random.random() < 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            patch = patch.point(lambda x: int(x * brightness_factor))
        
        # 对比度调整
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            patch = patch.point(lambda x: int(((x - 128) * contrast_factor) + 128))
        
        # 添加噪声
        if np.random.random() < 0.3:
            patch_np = np.array(patch, dtype=np.float32)
            noise = np.random.normal(0, 5, patch_np.shape)
            patch_np = np.clip(patch_np + noise, 0, 255)
            patch = Image.fromarray(patch_np.astype(np.uint8))
        # --- 额外数据增强结束 ---

        patch_np = np.array(patch)
        
        # 实现8个方向的离散几何变换 (这部分逻辑不变)
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

        H_tensor = torch.from_numpy(H[:2, :]).float()
        return patch, transformed_patch, H_tensor

# --- 特征图变换与损失函数 (改进版) ---
def warp_feature_map(feature_map, H_inv):
    B, C, H, W = feature_map.size()
    grid = F.affine_grid(H_inv, feature_map.size(), align_corners=False).to(feature_map.device)
    return F.grid_sample(feature_map, grid, align_corners=False)

def compute_detection_loss(det_original, det_rotated, H):
    """改进的检测损失：使用BCE损失替代MSE"""
    with torch.no_grad():
        H_inv = torch.inverse(torch.cat([H, torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).repeat(H.shape[0], 1, 1)], dim=1))[:, :2, :]
    warped_det_rotated = warp_feature_map(det_rotated, H_inv)
    
    # 使用BCE损失，更适合二分类问题
    bce_loss = F.binary_cross_entropy(det_original, warped_det_rotated)
    
    # 添加平滑L1损失作为辅助
    smooth_l1_loss = F.smooth_l1_loss(det_original, warped_det_rotated)
    
    return bce_loss + 0.1 * smooth_l1_loss

def compute_description_loss(desc_original, desc_rotated, H, margin=1.0):
    """IC版图专用几何感知描述子损失：编码曼哈顿几何特征"""
    B, C, H_feat, W_feat = desc_original.size()
    
    # 曼哈顿几何感知采样：重点采样边缘和角点区域
    num_samples = 200
    
    # 生成曼哈顿对齐的采样网格（水平和垂直优先）
    h_coords = torch.linspace(-1, 1, int(np.sqrt(num_samples)), device=desc_original.device)
    w_coords = torch.linspace(-1, 1, int(np.sqrt(num_samples)), device=desc_original.device)
    
    # 增加曼哈顿方向的采样密度
    manhattan_h = torch.cat([h_coords, torch.zeros_like(h_coords)])
    manhattan_w = torch.cat([torch.zeros_like(w_coords), w_coords])
    manhattan_coords = torch.stack([manhattan_h, manhattan_w], dim=1).unsqueeze(0).repeat(B, 1, 1)
    
    # 采样anchor点
    anchor = F.grid_sample(desc_original, manhattan_coords.unsqueeze(1), align_corners=False).squeeze(2).transpose(1, 2)
    
    # 计算对应的正样本点
    coords_hom = torch.cat([manhattan_coords, torch.ones(B, manhattan_coords.size(1), 1, device=manhattan_coords.device)], dim=2)
    M_inv = torch.inverse(torch.cat([H, torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).repeat(H.shape[0], 1, 1)], dim=1))
    coords_transformed = (coords_hom @ M_inv.transpose(1, 2))[:, :, :2]
    positive = F.grid_sample(desc_rotated, coords_transformed.unsqueeze(1), align_corners=False).squeeze(2).transpose(1, 2)
    
    # IC版图专用负样本策略：考虑重复结构
    with torch.no_grad():
        # 1. 几何感知的负样本：曼哈顿变换后的不同区域
        neg_coords = []
        for b in range(B):
            # 生成曼哈顿变换后的坐标（90度旋转等）
            angles = [0, 90, 180, 270]
            for angle in angles:
                if angle != 0:
                    theta = torch.tensor([angle * np.pi / 180])
                    rot_matrix = torch.tensor([
                        [torch.cos(theta), -torch.sin(theta), 0],
                        [torch.sin(theta), torch.cos(theta), 0]
                    ])
                    rotated_coords = manhattan_coords[b] @ rot_matrix[:2, :2].T
                    neg_coords.append(rotated_coords)
        
        neg_coords = torch.stack(neg_coords[:B*num_samples//2]).reshape(B, -1, 2)
        
        # 2. 特征空间困难负样本
        negative_candidates = F.grid_sample(desc_rotated, neg_coords, align_corners=False).squeeze(2).transpose(1, 2)
        
        # 3. 曼哈顿距离约束的困难样本选择
        anchor_expanded = anchor.unsqueeze(2).expand(-1, -1, negative_candidates.size(1), -1)
        negative_expanded = negative_candidates.unsqueeze(1).expand(-1, anchor.size(1), -1, -1)
        
        # 使用曼哈顿距离而非欧氏距离
        manhattan_dist = torch.sum(torch.abs(anchor_expanded - negative_expanded), dim=3)
        hard_indices = torch.topk(manhattan_dist, k=anchor.size(1)//2, largest=False)[1]
        negative = torch.gather(negative_candidates, 1, hard_indices)
    
    # IC版图专用的几何一致性损失
    # 1. 曼哈顿方向一致性损失
    manhattan_loss = 0
    for i in range(anchor.size(1)):
        # 计算水平和垂直方向的几何一致性
        anchor_norm = F.normalize(anchor[:, i], p=2, dim=1)
        positive_norm = F.normalize(positive[:, i], p=2, dim=1)
        
        # 鼓励描述子对曼哈顿变换不变
        cos_sim = torch.sum(anchor_norm * positive_norm, dim=1)
        manhattan_loss += torch.mean(1 - cos_sim)
    
    # 2. 稀疏性正则化（IC版图特征稀疏）
    sparsity_loss = torch.mean(torch.abs(anchor)) + torch.mean(torch.abs(positive))
    
    # 3. 二值化特征距离（处理二值化输入）
    binary_loss = torch.mean(torch.abs(torch.sign(anchor) - torch.sign(positive)))
    
    # 综合损失
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=1, reduction='mean')  # 使用L1距离
    geometric_triplet = triplet_loss(anchor, positive, negative)
    
    return geometric_triplet + 0.1 * manhattan_loss + 0.01 * sparsity_loss + 0.05 * binary_loss

# --- (已修改) 主函数与命令行接口 ---
def main(args):
    # 设置日志记录
    logger = setup_logging(args.save_dir)
    
    logger.info("--- 开始训练 RoRD 模型 ---")
    logger.info(f"训练参数: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"保存目录: {args.save_dir}")
    
    transform = get_transform()
    
    # 在数据集初始化时传入尺度抖动范围
    dataset = ICLayoutTrainingDataset(
        args.data_dir, 
        patch_size=config.PATCH_SIZE, 
        transform=transform, 
        scale_range=config.SCALE_JITTER_RANGE
    )
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = RoRD().cuda()
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 早停机制
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        total_det_loss = 0
        total_desc_loss = 0
        
        for i, (original, rotated, H) in enumerate(train_dataloader):
            original, rotated, H = original.cuda(), rotated.cuda(), H.cuda()
            
            det_original, desc_original = model(original)
            det_rotated, desc_rotated = model(rotated)
            
            det_loss = compute_detection_loss(det_original, det_rotated, H)
            desc_loss = compute_description_loss(desc_original, desc_rotated, H)
            loss = det_loss + desc_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            total_det_loss += det_loss.item()
            total_desc_loss += desc_loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {i}, Total Loss: {loss.item():.4f}, "
                          f"Det Loss: {det_loss.item():.4f}, Desc Loss: {desc_loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_det_loss = total_det_loss / len(train_dataloader)
        avg_desc_loss = total_desc_loss / len(train_dataloader)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        total_val_det_loss = 0
        total_val_desc_loss = 0
        
        with torch.no_grad():
            for original, rotated, H in val_dataloader:
                original, rotated, H = original.cuda(), rotated.cuda(), H.cuda()
                
                det_original, desc_original = model(original)
                det_rotated, desc_rotated = model(rotated)
                
                val_det_loss = compute_detection_loss(det_original, det_rotated, H)
                val_desc_loss = compute_description_loss(desc_original, desc_rotated, H)
                val_loss = val_det_loss + val_desc_loss
                
                total_val_loss += val_loss.item()
                total_val_det_loss += val_det_loss.item()
                total_val_desc_loss += val_desc_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_det_loss = total_val_det_loss / len(val_dataloader)
        avg_val_desc_loss = total_val_desc_loss / len(val_dataloader)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        logger.info(f"--- Epoch {epoch+1} 完成 ---")
        logger.info(f"训练 - Total: {avg_train_loss:.4f}, Det: {avg_det_loss:.4f}, Desc: {avg_desc_loss:.4f}")
        logger.info(f"验证 - Total: {avg_val_loss:.4f}, Det: {avg_val_det_loss:.4f}, Desc: {avg_val_desc_loss:.4f}")
        logger.info(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, 'rord_model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'learning_rate': args.lr,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs
                }
            }, save_path)
            logger.info(f"最佳模型已保存至: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停触发！{patience} 个epoch没有改善")
                break
    
    # 保存最终模型
    save_path = os.path.join(args.save_dir, 'rord_model_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_loss': avg_val_loss,
        'config': {
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
    }, save_path)
    logger.info(f"最终模型已保存至: {save_path}")
    logger.info("训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 RoRD 模型")
    parser.add_argument('--data_dir', type=str, default=config.LAYOUT_DIR)
    parser.add_argument('--save_dir', type=str, default=config.SAVE_DIR)
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    main(parser.parse_args())