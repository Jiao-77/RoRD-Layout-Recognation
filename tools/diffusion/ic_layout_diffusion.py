#!/usr/bin/env python3
"""
基于原始IC版图数据训练扩散模型，生成相似图像的完整实现。

使用DDPM (Denoising Diffusion Probabilistic Models)
针对单通道灰度IC版图图像进行优化。
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

# 尝试导入tqdm，如果没有则使用简单的进度显示
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class ICDiffusionDataset(Dataset):
    """IC版图扩散模型训练数据集"""

    def __init__(self, image_dir, image_size=256, augment=True):
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # 获取所有PNG图像
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(list(self.image_dir.glob(ext)))

        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # 转换到[0,1]范围
        ])

        # 数据增强
        self.augment = augment
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(90, fill=0),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 确保是灰度图

        # 基础变换
        image = self.transform(image)

        # 数据增强
        if self.augment and np.random.random() > 0.5:
            image = self.aug_transform(image)

        return image


class UNet(nn.Module):
    """简化的U-Net架构用于扩散模型"""

    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # 编码器
        self.encoder = nn.ModuleList([
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
        ])

        # 中间层
        self.middle = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1)
        )

        # 解码器
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
        ])

        # 输出层
        self.output = nn.Conv2d(64, out_channels, 3, padding=1)

        # 时间融合层
        self.time_fusion = nn.ModuleList([
            nn.Linear(time_dim, 64),
            nn.Linear(time_dim, 128),
            nn.Linear(time_dim, 256),
            nn.Linear(time_dim, 512),
        ])

        # 归一化层
        self.norms = nn.ModuleList([
            nn.GroupNorm(8, 64),
            nn.GroupNorm(8, 128),
            nn.GroupNorm(8, 256),
            nn.GroupNorm(8, 512),
        ])

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp(t.float().unsqueeze(-1))  # [B, time_dim]

        # 编码器路径
        skips = []
        for i, (conv, norm, fusion) in enumerate(zip(self.encoder, self.norms, self.time_fusion)):
            x = conv(x)
            x = norm(x)
            # 融合时间信息
            t_feat = fusion(t_emb).unsqueeze(-1).unsqueeze(-1)
            x = x + t_feat
            x = F.silu(x)
            skips.append(x)
            if i < len(self.encoder) - 1:
                x = F.silu(x)

        # 中间层
        x = self.middle(x)
        x = F.silu(x)

        # 解码器路径
        for i, (deconv, skip) in enumerate(zip(self.decoder, reversed(skips[:-1]))):
            x = deconv(x)
            x = x + skip  # 跳跃连接
            x = F.silu(x)

        # 输出
        x = self.output(x)
        return x


class NoiseScheduler:
    """噪声调度器"""

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # beta调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # 预计算
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, t):
        """向干净图像添加噪声"""
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def sample_timestep(self, batch_size):
        """采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,))

    def step(self, model, x_t, t):
        """单步去噪"""
        # 预测噪声
        predicted_noise = model(x_t, t)

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


class DiffusionTrainer:
    """扩散模型训练器"""

    def __init__(self, model, scheduler, device='cuda'):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
        self.loss_fn = nn.MSELoss()

    def train_step(self, optimizer, dataloader):
        """单步训练"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(self.device)

            # 采样时间步
            t = self.scheduler.sample_timestep(batch.shape[0]).to(self.device)

            # 添加噪声
            noisy_batch, noise = self.scheduler.add_noise(batch, t)

            # 预测噪声
            predicted_noise = self.model(noisy_batch, t)

            # 计算损失
            loss = self.loss_fn(predicted_noise, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def generate(self, num_samples, image_size=256, save_dir=None):
        """生成图像"""
        self.model.eval()

        with torch.no_grad():
            # 从纯噪声开始
            x = torch.randn(num_samples, 1, image_size, image_size).to(self.device)

            # 逐步去噪
            for t in reversed(range(self.scheduler.num_timesteps)):
                t_batch = torch.full((num_samples,), t, device=self.device)
                x = self.scheduler.step(self.model, x, t_batch)

            # 限制到[0,1]范围
            x = torch.clamp(x, 0.0, 1.0)

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


def train_diffusion_model(args):
    """训练扩散模型的主函数"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建数据集和数据加载器
    dataset = ICDiffusionDataset(args.data_dir, args.image_size, args.augment)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    logger.info(f"数据集大小: {len(dataset)}")

    # 创建模型和调度器
    model = UNet(in_channels=1, out_channels=1)
    scheduler = NoiseScheduler(num_timesteps=args.timesteps)
    trainer = DiffusionTrainer(model, scheduler, device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    logger.info(f"开始训练 {args.epochs} 个epoch...")
    for epoch in range(args.epochs):
        loss = trainer.train_step(optimizer, dataloader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.6f}")

        # 定期保存模型
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            checkpoint_path = Path(args.output_dir) / f"diffusion_epoch_{epoch+1}.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"保存检查点: {checkpoint_path}")

    # 生成样本
    logger.info("生成示例图像...")
    trainer.generate(
        num_samples=args.num_samples,
        image_size=args.image_size,
        save_dir=os.path.join(args.output_dir, 'samples')
    )

    # 保存最终模型
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    final_path = Path(args.output_dir) / "diffusion_final.pth"
    torch.save(final_checkpoint, final_path)
    logger.info(f"训练完成，最终模型保存在: {final_path}")


def generate_with_trained_model(args):
    """使用训练好的模型生成图像"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = UNet(in_channels=1, out_channels=1)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 创建调度器和训练器
    scheduler = NoiseScheduler(num_timesteps=args.timesteps)
    trainer = DiffusionTrainer(model, scheduler, device)

    # 生成图像
    trainer.generate(
        num_samples=args.num_samples,
        image_size=args.image_size,
        save_dir=args.output_dir
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IC版图扩散模型训练和生成")
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练扩散模型')
    train_parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录')
    train_parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    train_parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    train_parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    train_parser.add_argument('--timesteps', type=int, default=1000, help='扩散时间步数')
    train_parser.add_argument('--num_samples', type=int, default=50, help='生成的样本数量')
    train_parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    train_parser.add_argument('--augment', action='store_true', help='启用数据增强')

    # 生成命令
    gen_parser = subparsers.add_parser('generate', help='使用训练好的模型生成图像')
    gen_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    gen_parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    gen_parser.add_argument('--num_samples', type=int, default=200, help='生成样本数量')
    gen_parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    gen_parser.add_argument('--timesteps', type=int, default=1000, help='扩散时间步数')

    args = parser.parse_args()

    if args.command == 'train':
        train_diffusion_model(args)
    elif args.command == 'generate':
        generate_with_trained_model(args)
    else:
        parser.print_help()