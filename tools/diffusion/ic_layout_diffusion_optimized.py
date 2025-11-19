#!/usr/bin/env python3
"""
针对IC版图优化的去噪扩散模型

专门针对以曼哈顿多边形为全部组成元素的IC版图光栅化图像进行优化：
- 曼哈顿几何感知的U-Net架构
- 边缘感知损失函数
- 多尺度结构损失
- 曼哈顿约束正则化
- 几何保持的数据增强
- 后处理优化
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
import cv2

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class ICDiffusionDataset(Dataset):
    """IC版图扩散模型训练数据集 - 优化版"""

    def __init__(self, image_dir, image_size=256, augment=True, use_edge_condition=False):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.use_edge_condition = use_edge_condition

        # 获取所有PNG图像
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(list(self.image_dir.glob(ext)))

        # 基础变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # 几何保持的数据增强
        self.augment = augment
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # 移除旋转，保持曼哈顿几何
            ])

    def __len__(self):
        return len(self.image_paths)

    def _extract_edges(self, image_tensor):
        """提取边缘条件图"""
        # 使用Sobel算子提取边缘
        sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],
                              dtype=image_tensor.dtype, device=image_tensor.device)
        sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]],
                              dtype=image_tensor.dtype, device=image_tensor.device)

        edge_x = F.conv2d(image_tensor.unsqueeze(0), sobel_x, padding=1)
        edge_y = F.conv2d(image_tensor.unsqueeze(0), sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)

        return torch.clamp(edge_magnitude, 0, 1)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')

        # 基础变换
        image = self.transform(image)

        # 几何保持的数据增强
        if self.augment and np.random.random() > 0.5:
            image = self.aug_transform(image)

        if self.use_edge_condition:
            edge_condition = self._extract_edges(image)
            return image, edge_condition.squeeze(0)

        return image


class EdgeAwareLoss(nn.Module):
    """边缘感知损失函数"""

    def __init__(self):
        super().__init__()
        # 注册为缓冲区以避免重复创建
        self.register_buffer('sobel_x', torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]))
        self.register_buffer('sobel_y', torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]))

    def forward(self, pred, target):
        # 原始MSE损失
        mse_loss = F.mse_loss(pred, target)

        # 计算边缘
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1)

        # 边缘损失
        edge_loss = F.mse_loss(pred_edge_x, target_edge_x) + F.mse_loss(pred_edge_y, target_edge_y)

        return mse_loss + 0.5 * edge_loss


class MultiScaleStructureLoss(nn.Module):
    """多尺度结构损失"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # 原始分辨率损失
        loss_1x = F.mse_loss(pred, target)

        # 2x下采样损失
        pred_2x = F.avg_pool2d(pred, 2)
        target_2x = F.avg_pool2d(target, 2)
        loss_2x = F.mse_loss(pred_2x, target_2x)

        # 4x下采样损失
        pred_4x = F.avg_pool2d(pred, 4)
        target_4x = F.avg_pool2d(target, 4)
        loss_4x = F.mse_loss(pred_4x, target_4x)

        return loss_1x + 0.5 * loss_2x + 0.25 * loss_4x


def manhattan_regularization_loss(generated_image, device='cuda'):
    """曼哈顿约束正则化损失"""
    if device == 'cuda':
        device = generated_image.device

    # Sobel算子
    sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=device, dtype=generated_image.dtype)
    sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=device, dtype=generated_image.dtype)

    # 检测边缘
    edge_x = F.conv2d(generated_image, sobel_x, padding=1)
    edge_y = F.conv2d(generated_image, sobel_y, padding=1)

    # 边缘强度
    edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)

    # 计算角度偏差
    angles = torch.atan2(edge_y, edge_x)

    # 惩罚不接近0°、90°、180°或270°的角度
    angle_penalty = torch.min(
        torch.min(torch.abs(angles), torch.abs(angles - np.pi/2)),
        torch.min(torch.abs(angles - np.pi), torch.abs(angles - 3*np.pi/2))
    )

    return torch.mean(angle_penalty * edge_magnitude)


class ManhattanAwareUNet(nn.Module):
    """曼哈顿几何感知的U-Net架构"""

    def __init__(self, in_channels=1, out_channels=1, time_dim=256, use_edge_condition=False):
        super().__init__()
        self.use_edge_condition = use_edge_condition

        # 输入通道数（原始图像 + 可选边缘条件）
        input_channels = in_channels + (1 if use_edge_condition else 0)

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # 曼哈顿几何感知的初始卷积层
        self.horiz_conv = nn.Conv2d(input_channels, 32, (1, 7), padding=(0, 3))
        self.vert_conv = nn.Conv2d(input_channels, 32, (7, 1), padding=(3, 0))
        self.standard_conv = nn.Conv2d(input_channels, 32, 3, padding=1)

        # 特征融合
        self.initial_fusion = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )

        # 编码器 - 增强版
        self.encoder = nn.ModuleList([
            self._make_block(64, 128),
            self._make_block(128, 256, stride=2),
            self._make_block(256, 512, stride=2),
            self._make_block(512, 1024, stride=2),
        ])

        # 残差投影层 - 确保残差连接的通道数和空间尺寸匹配
        self.residual_projections = nn.ModuleList([
            nn.Conv2d(64, 128, 1),           # 64 -> 128, 保持尺寸
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 128 -> 256, 尺寸减半
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # 256 -> 512, 尺寸减半
            nn.Conv2d(512, 1024, 3, stride=2, padding=1), # 512 -> 1024, 尺寸减半
        ])

        # 中间层
        self.middle = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.GroupNorm(8, 1024),
            nn.SiLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.GroupNorm(8, 1024),
            nn.SiLU(),
        )

        # 解码器
        self.decoder = nn.ModuleList([
            self._make_decoder_block(1024, 512),
            self._make_decoder_block(512, 256),
            self._make_decoder_block(256, 128),
            self._make_decoder_block(128, 64),
        ])

        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )

        # 时间融合层 - 与编码器输出通道数匹配
        self.time_fusion = nn.ModuleList([
            nn.Linear(time_dim, 128),
            nn.Linear(time_dim, 256),
            nn.Linear(time_dim, 512),
            nn.Linear(time_dim, 1024),
        ])

    def _make_block(self, in_channels, out_channels, stride=1):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def _make_decoder_block(self, in_channels, out_channels):
        """创建解码器块"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x, t, edge_condition=None):
        # 如果有边缘条件，连接到输入
        if self.use_edge_condition and edge_condition is not None:
            x = torch.cat([x, edge_condition], dim=1)

        # 时间嵌入
        t_emb = self.time_mlp(t.float().unsqueeze(-1))  # [B, time_dim]

        # 曼哈顿几何感知的特征提取
        h_features = F.silu(self.horiz_conv(x))
        v_features = F.silu(self.vert_conv(x))
        s_features = F.silu(self.standard_conv(x))

        # 融合特征
        x = torch.cat([h_features, v_features, s_features], dim=1)
        x = self.initial_fusion(x)

        # 编码器路径
        skips = []
        for i, (encoder, fusion) in enumerate(zip(self.encoder, self.time_fusion)):
            # 保存残差连接并使用投影层匹配通道数
            residual = self.residual_projections[i](x)
            x = encoder(x)

            # 融合时间信息
            t_feat = fusion(t_emb).unsqueeze(-1).unsqueeze(-1)
            x = x + t_feat

            # 跳跃连接 - 添加投影后的残差
            skips.append(x + residual)

        # 中间层
        x = self.middle(x)

        # 解码器路径
        for i, (decoder, skip) in enumerate(zip(self.decoder, reversed(skips))):
            x = decoder(x)

            # 确保跳跃连接尺寸匹配 - 如果尺寸不匹配则进行裁剪或填充
            if x.shape[2:] != skip.shape[2:]:
                # 裁剪到最小尺寸
                h_min = min(x.shape[2], skip.shape[2])
                w_min = min(x.shape[3], skip.shape[3])
                if x.shape[2] > h_min:
                    x = x[:, :, :h_min, :]
                if x.shape[3] > w_min:
                    x = x[:, :, :, :w_min]
                if skip.shape[2] > h_min:
                    skip = skip[:, :, :h_min, :]
                if skip.shape[3] > w_min:
                    skip = skip[:, :, :, :w_min]

            x = x + skip  # 跳跃连接

        # 输出
        x = self.output(x)
        return x


class OptimizedNoiseScheduler:
    """优化的噪声调度器"""

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        self.num_timesteps = num_timesteps

        # 不同调度策略
        if schedule_type == 'cosine':
            # 余弦调度，通常效果更好
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        else:
            # 线性调度
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

        # 预计算 - 确保所有张量都是float32
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, t):
        """向干净图像添加噪声"""
        noise = torch.randn_like(x_0)
        # 确保调度器张量与输入张量在同一设备上
        device = x_0.device
        if self.sqrt_alphas_cumprod.device != device:
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def sample_timestep(self, batch_size, device=None):
        """采样时间步"""
        t = torch.randint(0, self.num_timesteps, (batch_size,))
        if device is not None:
            t = t.to(device)
        return t

    def step(self, model, x_t, t):
        """单步去噪"""
        # 预测噪声
        predicted_noise = model(x_t, t)

        # 确保调度器张量与输入张量在同一设备上
        device = x_t.device
        if self.alphas.device != device:
            self.alphas = self.alphas.to(device)
            self.betas = self.betas.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

        # 计算系数
        alpha_t = self.alphas[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        beta_t = self.betas[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 计算均值
        model_mean = (1.0 / sqrt_alpha_t) * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)

        if t.min() == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(beta_t) * noise


def manhattan_post_process(image, threshold=0.5):
    """曼哈顿化后处理"""
    device = image.device

    # 二值化
    binary = (image > threshold).float()

    # 形态学操作强化直角特征
    kernel_h = torch.tensor([[[[1,1,1]]]], device=device)
    kernel_v = torch.tensor([[[[1],[1],[1]]]], device=device)

    # 水平和垂直增强
    horizontal = F.conv2d(binary, kernel_h, padding=(0,1))
    vertical = F.conv2d(binary, kernel_v, padding=(1,0))

    # 合并结果
    result = torch.clamp(horizontal + vertical - binary, 0, 1)

    # 最终阈值处理
    result = (result > 0.5).float()

    return result


class OptimizedDiffusionTrainer:
    """优化的扩散模型训练器"""

    def __init__(self, model, scheduler, device='cuda', use_edge_condition=False):
        self.model = model.to(device)
        self.device = device
        self.use_edge_condition = use_edge_condition

        # 确保调度器的所有张量都在正确的设备上
        self._move_scheduler_to_device(scheduler)
        self.scheduler = scheduler

        # 组合损失函数
        self.edge_loss = EdgeAwareLoss().to(device)
        self.structure_loss = MultiScaleStructureLoss().to(device)
        self.mse_loss = nn.MSELoss()

    def _move_scheduler_to_device(self, scheduler):
        """将调度器的所有张量移动到指定设备"""
        if hasattr(scheduler, 'betas'):
            scheduler.betas = scheduler.betas.to(self.device)
        if hasattr(scheduler, 'alphas'):
            scheduler.alphas = scheduler.alphas.to(self.device)
        if hasattr(scheduler, 'alphas_cumprod'):
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
        if hasattr(scheduler, 'sqrt_alphas_cumprod'):
            scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(self.device)
        if hasattr(scheduler, 'sqrt_one_minus_alphas_cumprod'):
            scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(self.device)

    def train_step(self, optimizer, dataloader, manhattan_weight=0.1):
        """单步训练"""
        self.model.train()
        total_loss = 0
        total_edge_loss = 0
        total_structure_loss = 0
        total_manhattan_loss = 0

        for batch in dataloader:
            if self.use_edge_condition:
                images, edge_conditions = batch
                edge_conditions = edge_conditions.to(self.device)
            else:
                images = batch
                edge_conditions = None

            images = images.to(self.device)

            # 采样时间步
            t = self.scheduler.sample_timestep(images.shape[0]).to(self.device)

            # 添加噪声
            noisy_images, noise = self.scheduler.add_noise(images, t)

            # 预测噪声
            predicted_noise = self.model(noisy_images, t, edge_conditions)

            # 计算多种损失
            mse_loss = self.mse_loss(predicted_noise, noise)
            edge_loss = self.edge_loss(predicted_noise, noise)
            structure_loss = self.structure_loss(predicted_noise, noise)

            # 曼哈顿正则化损失
            with torch.no_grad():
                # 对去噪结果应用曼哈顿约束
                denoised = noisy_images - predicted_noise
                manhattan_loss = manhattan_regularization_loss(denoised, self.device)

            # 总损失
            total_step_loss = mse_loss + 0.3 * edge_loss + 0.2 * structure_loss + manhattan_weight * manhattan_loss

            # 反向传播
            optimizer.zero_grad()
            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()

            total_loss += total_step_loss.item()
            total_edge_loss += edge_loss.item()
            total_structure_loss += structure_loss.item()
            total_manhattan_loss += manhattan_loss.item()

        num_batches = len(dataloader)
        return {
            'total_loss': total_loss / num_batches,
            'mse_loss': total_loss / num_batches,  # 近似值
            'edge_loss': total_edge_loss / num_batches,
            'structure_loss': total_structure_loss / num_batches,
            'manhattan_loss': total_manhattan_loss / num_batches
        }

    def generate(self, num_samples, image_size=256, save_dir=None, use_post_process=True):
        """生成图像"""
        self.model.eval()

        with torch.no_grad():
            # 从纯噪声开始
            x = torch.randn(num_samples, 1, image_size, image_size).to(self.device)

            # 逐步去噪
            for t in reversed(range(self.scheduler.num_timesteps)):
                t_batch = torch.full((num_samples,), t, device=self.device)
                x = self.scheduler.step(self.model, x, t_batch)

                # 限制到合理范围
                x = torch.clamp(x, -2.0, 2.0)

            # 最终处理
            x = torch.clamp(x, 0.0, 1.0)

            # 后处理
            if use_post_process:
                x = manhattan_post_process(x)

        # 保存图像
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_samples):
                img_tensor = x[i].cpu()
                img_array = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
                img.save(save_dir / f"generated_{i:06d}.png")

        return x.cpu()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="优化的IC版图扩散模型训练和生成")
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练扩散模型')
    train_parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录')
    train_parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    train_parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    train_parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    train_parser.add_argument('--timesteps', type=int, default=1000, help='扩散时间步数')
    train_parser.add_argument('--num_samples', type=int, default=50, help='生成的样本数量')
    train_parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    train_parser.add_argument('--augment', action='store_true', help='启用数据增强')
    train_parser.add_argument('--edge_condition', action='store_true', help='使用边缘条件')
    train_parser.add_argument('--manhattan_weight', type=float, default=0.1, help='曼哈顿正则化权重')
    train_parser.add_argument('--schedule_type', type=str, default='cosine', choices=['linear', 'cosine'], help='噪声调度类型')

    # 生成命令
    gen_parser = subparsers.add_parser('generate', help='使用训练好的模型生成图像')
    gen_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    gen_parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    gen_parser.add_argument('--num_samples', type=int, default=200, help='生成样本数量')
    gen_parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    gen_parser.add_argument('--timesteps', type=int, default=1000, help='扩散时间步数')
    gen_parser.add_argument('--use_post_process', action='store_true', default=True, help='启用后处理')

    args = parser.parse_args()

    # TODO: 实现训练和生成函数，使用优化后的组件
    print("[TODO] 实现完整的训练和生成流程，使用优化后的模型架构和损失函数")