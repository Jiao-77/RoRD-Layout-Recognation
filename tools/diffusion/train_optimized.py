#!/usr/bin/env python3
"""
使用优化后的扩散模型进行训练的完整脚本
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import yaml
from torch.utils.data import DataLoader
import argparse

# 导入优化后的模块
from ic_layout_diffusion_optimized import (
    ICDiffusionDataset,
    ManhattanAwareUNet,
    OptimizedNoiseScheduler,
    OptimizedDiffusionTrainer
)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('diffusion_training.log')
        ]
    )
    return logging.getLogger(__name__)

def save_checkpoint(model, optimizer, scheduler, epoch, losses, checkpoint_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'losses': losses
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"检查点已保存: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    losses = checkpoint.get('losses', {})

    logging.info(f"检查点已加载: {checkpoint_path}, 从epoch {start_epoch}继续")
    return start_epoch, losses

def validate_model(trainer, val_dataloader, device):
    """验证模型"""
    trainer.model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            if trainer.use_edge_condition:
                images, edge_conditions = batch
                edge_conditions = edge_conditions.to(device)
            else:
                images = batch
                edge_conditions = None

            images = images.to(device)
            t = trainer.scheduler.sample_timestep(images.shape[0]).to(device)
            noisy_images, noise = trainer.scheduler.add_noise(images, t)
            predicted_noise = trainer.model(noisy_images, t, edge_conditions)

            loss = trainer.mse_loss(predicted_noise, noise)
            total_loss += loss.item()

    trainer.model.train()
    return total_loss / len(val_dataloader)

def train_optimized_diffusion(args):
    """训练优化的扩散模型"""
    logger = setup_logging()

    # 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 使用设备: {device}")

    # 设备详细信息
    if device.type == 'cuda':
        logger.info(f"📊 GPU信息: {torch.cuda.get_device_name()}")
        logger.info(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"🔥 CUDA版本: {torch.version.cuda}")
    else:
        logger.info("💻 使用CPU进行训练（较慢，建议使用GPU）")

    # 设置随机种子
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        logger.info(f"🎲 随机种子设置为: {args.seed}")
    logger.info("=" * 60)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练配置
    config = {
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'timesteps': args.timesteps,
        'schedule_type': args.schedule_type,
        'edge_condition': args.edge_condition,
        'manhattan_weight': args.manhattan_weight,
        'augment': args.augment,
        'seed': args.seed
    }

    with open(output_dir / 'training_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # 创建数据集
    logger.info(f"📂 开始加载数据集: {args.data_dir}")
    logger.info(f"🖼️  图像尺寸: {args.image_size}x{args.image_size}")
    logger.info(f"🔄 数据增强: {'启用' if args.augment else '禁用'}")
    logger.info(f"📐 边缘条件: {'启用' if args.edge_condition else '禁用'}")

    start_time = time.time()
    dataset = ICDiffusionDataset(
        image_dir=args.data_dir,
        image_size=args.image_size,
        augment=args.augment,
        use_edge_condition=args.edge_condition
    )
    load_time = time.time() - start_time

    # 检查数据集是否为空
    if len(dataset) == 0:
        logger.error(f"❌ 数据集为空！请检查数据目录: {args.data_dir}")
        raise ValueError(f"数据集为空，在目录 {args.data_dir} 中未找到图像文件")

    logger.info(f"✅ 数据集加载完成，耗时: {load_time:.2f}秒")
    logger.info(f"📊 找到 {len(dataset)} 个训练样本")

    # 数据集分割 - 修复空数据集问题
    total_size = len(dataset)
    if total_size < 10:  # 如果数据集太小，全部用于训练
        logger.warning(f"数据集较小 ({total_size} 样本)，全部用于训练")
        train_dataset = dataset
        val_dataset = None
    else:
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # 数据加载器 - 修复None验证集问题
    if device.type == 'cuda':
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=min(args.batch_size, len(train_dataset)),  # 确保批次大小不超过数据集大小
            shuffle=True,
            num_workers=min(4, max(1, len(train_dataset) // args.batch_size)),
            pin_memory=True,
            drop_last=True  # 避免最后一个不完整的批次
        )
    else:
        # CPU模式下使用较少的worker
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=min(args.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=0,  # CPU模式下避免多进程
            drop_last=True
        )

    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=2
        )
    else:
        val_dataloader = None

    # 创建模型
    logger.info("🏗️  正在创建U-Net扩散模型...")
    model_start_time = time.time()
    model = ManhattanAwareUNet(
        in_channels=1,
        out_channels=1,
        use_edge_condition=args.edge_condition
    ).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"✅ 模型创建完成，耗时: {time.time() - model_start_time:.2f}秒")
    logger.info(f"📊 模型参数: {total_params:,} 总计, {trainable_params:,} 可训练")

    # 创建调度器
    logger.info(f"⏱️  创建噪声调度器: {args.schedule_type}, {args.timesteps} 步")
    scheduler = OptimizedNoiseScheduler(
        num_timesteps=args.timesteps,
        schedule_type=args.schedule_type
    )

    # 创建训练器
    logger.info("🎯 创建优化训练器...")
    trainer = OptimizedDiffusionTrainer(
        model, scheduler, device, args.edge_condition
    )

    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # 检查点恢复
    start_epoch = 0
    losses_history = []

    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            start_epoch, losses_history = load_checkpoint(
                checkpoint_path, model, optimizer, lr_scheduler
            )
        else:
            logger.warning(f"检查点文件不存在: {checkpoint_path}")

    logger.info(f"🚀 开始训练 {args.epochs} 个epoch (从epoch {start_epoch}开始)...")
    logger.info("=" * 80)

    # 训练循环
    best_val_loss = float('inf')
    total_training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # 训练
        logger.info(f"🏃 训练 Epoch {epoch+1}/{args.epochs}")
        step_start_time = time.time()
        train_losses = trainer.train_step(
            optimizer, train_dataloader, args.manhattan_weight
        )
        step_time = time.time() - step_start_time

        # 验证 - 修复None验证集问题
        if val_dataloader is not None:
            logger.info(f"🔍 验证 Epoch {epoch+1}/{args.epochs}")
            val_loss = validate_model(trainer, val_dataloader, device)
        else:
            # 如果没有验证集，使用训练损失作为验证损失
            val_loss = train_losses['total_loss']
            logger.info("⚠️  未使用验证集 - 使用训练损失作为参考")

        # 学习率调度
        lr_scheduler.step()

        # 记录损失和进度
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time

        # 详细日志输出
        logger.info(f"✅ Epoch {epoch+1} 完成 (耗时: {epoch_time:.1f}s, 训练: {step_time:.1f}s)")
        logger.info(f"📉 训练损失 - Total: {train_losses['total_loss']:.6f} | "
                   f"MSE: {train_losses['mse_loss']:.6f} | "
                   f"Edge: {train_losses['edge_loss']:.6f} | "
                   f"Manhattan: {train_losses['manhattan_loss']:.6f}")
        if val_dataloader is not None:
            logger.info(f"🎯 验证损失: {val_loss:.6f}")
        logger.info(f"📈 学习率: {current_lr:.8f}")

        # 内存使用情况（仅GPU）
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"💾 GPU内存: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")

        losses_history.append({
            'epoch': epoch,
            'train_loss': train_losses['total_loss'],
            'val_loss': val_loss,
            'edge_loss': train_losses['edge_loss'],
            'structure_loss': train_losses['structure_loss'],
            'manhattan_loss': train_losses['manhattan_loss'],
            'lr': current_lr
        })

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "best_model.pth"
            logger.info(f"🏆 新的最佳模型! 验证损失: {val_loss:.6f}")
            save_checkpoint(
                model, optimizer, lr_scheduler, epoch, losses_history, best_model_path
            )

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            logger.info(f"💾 保存检查点: {checkpoint_path.name}")
            save_checkpoint(
                model, optimizer, lr_scheduler, epoch, losses_history, checkpoint_path
            )

        # 生成样本
        if (epoch + 1) % args.sample_interval == 0:
            sample_dir = output_dir / f"samples_epoch_{epoch+1}"
            logger.info(f"🎨 生成 {args.num_samples} 个样本到 {sample_dir}")
            sample_start_time = time.time()
            trainer.generate(
                num_samples=args.num_samples,
                image_size=args.image_size,
                save_dir=sample_dir,
                use_post_process=True
            )
            sample_time = time.time() - sample_start_time
            logger.info(f"✅ 样本生成完成，耗时: {sample_time:.1f}秒")

        logger.info("-" * 80)  # 分隔线

    # 保存最终模型
    total_training_time = time.time() - total_training_start
    final_model_path = output_dir / "final_model.pth"
    logger.info("🎉 训练完成! 保存最终模型...")
    save_checkpoint(
        model, optimizer, lr_scheduler, args.epochs-1, losses_history, final_model_path
    )
    logger.info(f"💾 最终模型已保存: {final_model_path}")
    logger.info(f"⏱️  总训练时间: {total_training_time/3600:.2f} 小时")
    logger.info(f"⚡ 平均每epoch: {total_training_time/args.epochs:.1f} 秒")

    # 保存损失历史
    with open(output_dir / 'loss_history.yaml', 'w') as f:
        yaml.dump(losses_history, f, default_flow_style=False)

    # 最终生成
    logger.info("🎨 生成最终样本...")
    final_sample_dir = output_dir / "final_samples"
    trainer.generate(
        num_samples=args.num_samples * 2,  # 生成更多样本
        image_size=args.image_size,
        save_dir=final_sample_dir,
        use_post_process=True
    )

    # 训练总结
    logger.info("=" * 80)
    logger.info("🎉 训练总结:")
    logger.info(f"   ✅ 训练完成！")
    logger.info(f"   📊 最终验证损失: {best_val_loss:.6f}")
    logger.info(f"   📈 总训练epochs: {args.epochs}")
    logger.info(f"   🎯 训练配置: 学习率={args.lr}, 批次大小={args.batch_size}")
    logger.info(f"   📁 输出目录: {output_dir}")
    logger.info(f"   🏆 最佳模型: {output_dir / 'best_model.pth'}")
    logger.info(f"   💾 最终模型: {final_model_path}")
    logger.info(f"   🎨 最终样本: {final_sample_dir}")
    logger.info(f"   📋 损失历史: {output_dir / 'loss_history.yaml'}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="训练优化的IC版图扩散模型")

    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')

    # 模型参数
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--timesteps', type=int, default=1000, help='扩散时间步数')
    parser.add_argument('--schedule_type', type=str, default='cosine',
                       choices=['linear', 'cosine'], help='噪声调度类型')
    parser.add_argument('--edge_condition', action='store_true', help='使用边缘条件')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--manhattan_weight', type=float, default=0.1, help='曼哈顿正则化权重')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 训练控制
    parser.add_argument('--augment', action='store_true', help='启用数据增强')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--sample_interval', type=int, default=20, help='生成样本间隔')
    parser.add_argument('--num_samples', type=int, default=16, help='每次生成的样本数量')

    args = parser.parse_args()

    # 检查数据目录
    if not Path(args.data_dir).exists():
        print(f"错误: 数据目录不存在: {args.data_dir}")
        sys.exit(1)

    # 开始训练
    train_optimized_diffusion(args)


if __name__ == "__main__":
    main()