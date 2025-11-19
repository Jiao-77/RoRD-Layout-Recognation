#!/usr/bin/env python3
"""
使用优化后的扩散模型生成IC版图图像
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
import argparse
import yaml
from PIL import Image
import numpy as np

# 导入优化后的模块
from ic_layout_diffusion_optimized import (
    ManhattanAwareUNet,
    OptimizedNoiseScheduler,
    OptimizedDiffusionTrainer,
    manhattan_post_process,
    manhattan_regularization_loss
)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('diffusion_generation.log')
        ]
    )
    return logging.getLogger(__name__)

def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    logger = logging.getLogger(__name__)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从检查点中获取配置信息（如果有）
    config = checkpoint.get('config', {})

    # 创建模型
    model = ManhattanAwareUNet(
        in_channels=1,
        out_channels=1,
        use_edge_condition=config.get('edge_condition', False)
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"模型已加载: {checkpoint_path}")
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    return model, config

def ddim_sample(model, scheduler, num_samples, image_size, device, num_steps=50, eta=0.0):
    """DDIM采样，比标准DDPM更快"""
    model.eval()

    # 从纯噪声开始
    x = torch.randn(num_samples, 1, image_size, image_size).to(device)

    # 选择时间步
    timesteps = torch.linspace(scheduler.num_timesteps - 1, 0, num_steps).long().to(device)

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_batch = torch.full((num_samples,), t, device=device)

            # 预测噪声
            predicted_noise = model(x, t_batch)

            # 计算原始图像的估计 - 确保调度器张量在正确设备上
            alpha_t = scheduler.alphas[t].to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_cumprod_t = scheduler.alphas_cumprod[t].to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            beta_t = scheduler.betas[t].to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[t].to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # 计算x_0的估计
            x_0_pred = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)

            # 计算前一时间步的方向
            if i < len(timesteps) - 1:
                alpha_t_prev = scheduler.alphas[timesteps[i+1]].to(device)
                alpha_cumprod_t_prev = scheduler.alphas_cumprod[timesteps[i+1]].to(device)
                sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                sqrt_one_minus_alpha_cumprod_t_prev = torch.sqrt(1 - alpha_cumprod_t_prev).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                # 计算方差
                variance = eta * torch.sqrt(beta_t).squeeze().squeeze().squeeze()

                # 计算前一时间步的x
                x = sqrt_alpha_cumprod_t_prev * x_0_pred + torch.sqrt(1 - alpha_cumprod_t_prev - variance**2) * predicted_noise

                if eta > 0:
                    noise = torch.randn_like(x)
                    x += variance * noise
            else:
                x = x_0_pred

            # 限制范围
            x = torch.clamp(x, -2.0, 2.0)

    return torch.clamp(x, 0.0, 1.0)

def generate_with_guidance(model, scheduler, num_samples, image_size, device,
                          guidance_scale=1.0, num_steps=50, use_ddim=True):
    """带引导的采样（可扩展为classifier-free guidance）"""

    if use_ddim:
        # 使用DDIM采样
        samples = ddim_sample(model, scheduler, num_samples, image_size, device, num_steps)
    else:
        # 使用标准DDPM采样
        trainer = OptimizedDiffusionTrainer(model, scheduler, device)
        samples = trainer.generate(num_samples, image_size, save_dir=None, use_post_process=False)

    return samples

def evaluate_generation_quality(samples, device):
    """评估生成质量"""
    logger = logging.getLogger(__name__)

    quality_metrics = {}

    # 1. 曼哈顿几何合规性
    manhattan_loss = manhattan_regularization_loss(samples, device)
    quality_metrics['manhattan_compliance'] = float(manhattan_loss.item())

    # 2. 边缘锐度
    sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=device, dtype=samples.dtype)
    sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=device, dtype=samples.dtype)

    edge_x = F.conv2d(samples, sobel_x, padding=1)
    edge_y = F.conv2d(samples, sobel_y, padding=1)
    edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)

    quality_metrics['edge_sharpness'] = float(torch.mean(edge_magnitude).item())

    # 3. 对比度
    quality_metrics['contrast'] = float(torch.std(samples).item())

    # 4. 稀疏性（IC版图通常是稀疏的）
    quality_metrics['sparsity'] = float((samples < 0.1).float().mean().item())

    logger.info("生成质量评估:")
    for metric, value in quality_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return quality_metrics

def generate_optimized_samples(args):
    """生成优化样本的主函数"""
    logger = setup_logging()

    # 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model, config = load_model(args.checkpoint, device)

    # 创建调度器
    scheduler = OptimizedNoiseScheduler(
        num_timesteps=config.get('timesteps', args.timesteps),
        schedule_type=config.get('schedule_type', args.schedule_type)
    )

    # 确保调度器的所有张量都在正确的设备上
    if hasattr(scheduler, 'betas'):
        scheduler.betas = scheduler.betas.to(device)
    if hasattr(scheduler, 'alphas'):
        scheduler.alphas = scheduler.alphas.to(device)
    if hasattr(scheduler, 'alphas_cumprod'):
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    if hasattr(scheduler, 'sqrt_alphas_cumprod'):
        scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(device)
    if hasattr(scheduler, 'sqrt_one_minus_alphas_cumprod'):
        scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)

    # 生成参数
    generation_config = {
        'num_samples': args.num_samples,
        'image_size': args.image_size,
        'guidance_scale': args.guidance_scale,
        'num_steps': args.num_steps,
        'use_ddim': args.use_ddim,
        'eta': args.eta,
        'seed': args.seed
    }

    # 保存生成配置
    with open(output_dir / 'generation_config.yaml', 'w') as f:
        yaml.dump(generation_config, f, default_flow_style=False)

    logger.info(f"开始生成 {args.num_samples} 个样本...")
    logger.info(f"采样步数: {args.num_steps}, DDIM: {args.use_ddim}, ETA: {args.eta}")

    # 分批生成以避免内存不足
    all_samples = []
    batch_size = min(args.batch_size, args.num_samples)
    num_batches = (args.num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, args.num_samples)
        current_batch_size = end_idx - start_idx

        logger.info(f"生成批次 {batch_idx + 1}/{num_batches} ({current_batch_size} 个样本)")

        # 生成样本
        with torch.no_grad():
            samples = generate_with_guidance(
                model, scheduler, current_batch_size, args.image_size, device,
                args.guidance_scale, args.num_steps, args.use_ddim
            )

            # 后处理
            if args.use_post_process:
                samples = manhattan_post_process(samples, threshold=args.post_process_threshold)

            all_samples.append(samples)

            # 立即保存当前批次
            batch_dir = output_dir / f"batch_{batch_idx + 1}"
            batch_dir.mkdir(exist_ok=True)

            for i in range(current_batch_size):
                img_tensor = samples[i].cpu()
                img_array = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
                img.save(batch_dir / f"sample_{start_idx + i:06d}.png")

    # 合并所有样本
    all_samples = torch.cat(all_samples, dim=0)

    # 评估生成质量
    quality_metrics = evaluate_generation_quality(all_samples, device)

    # 保存质量评估结果
    with open(output_dir / 'quality_metrics.yaml', 'w') as f:
        yaml.dump(quality_metrics, f, default_flow_style=False)

    # 创建质量报告
    report_content = f"""
IC版图扩散模型生成报告
======================

生成配置:
- 模型检查点: {args.checkpoint}
- 样本数量: {args.num_samples}
- 图像尺寸: {args.image_size}x{args.image_size}
- 采样步数: {args.num_steps}
- DDIM采样: {args.use_ddim}
- 后处理: {args.use_post_process}

质量指标:
- 曼哈顿几何合规性: {quality_metrics['manhattan_compliance']:.4f} (越低越好)
- 边缘锐度: {quality_metrics['edge_sharpness']:.4f}
- 对比度: {quality_metrics['contrast']:.4f}
- 稀疏性: {quality_metrics['sparsity']:.4f}

输出目录: {output_dir}
"""

    with open(output_dir / 'generation_report.txt', 'w') as f:
        f.write(report_content)

    logger.info("生成完成!")
    logger.info(f"样本保存目录: {output_dir}")
    logger.info(f"质量报告: {output_dir / 'generation_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description="使用优化的扩散模型生成IC版图")

    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')

    # 生成参数
    parser.add_argument('--num_samples', type=int, default=200, help='生成样本数量')
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')

    # 采样参数
    parser.add_argument('--num_steps', type=int, default=50, help='采样步数')
    parser.add_argument('--use_ddim', action='store_true', default=True, help='使用DDIM采样')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='引导尺度')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta参数 (0=确定性, 1=随机)')

    # 后处理参数
    parser.add_argument('--use_post_process', action='store_true', default=True, help='启用后处理')
    parser.add_argument('--post_process_threshold', type=float, default=0.5, help='后处理阈值')

    # 模型配置（用于覆盖检查点中的配置）
    parser.add_argument('--timesteps', type=int, default=1000, help='扩散时间步数')
    parser.add_argument('--schedule_type', type=str, default='cosine',
                       choices=['linear', 'cosine'], help='噪声调度类型')

    args = parser.parse_args()

    # 检查检查点文件
    if not Path(args.checkpoint).exists():
        print(f"错误: 检查点文件不存在: {args.checkpoint}")
        sys.exit(1)

    # 开始生成
    generate_optimized_samples(args)


if __name__ == "__main__":
    main()