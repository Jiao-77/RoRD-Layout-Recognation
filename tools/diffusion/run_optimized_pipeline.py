#!/usr/bin/env python3
"""
一键运行优化的IC版图扩散模型训练和生成管线
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
import logging
import shutil

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('optimized_pipeline.log')
        ]
    )
    return logging.getLogger(__name__)

def run_command(cmd, description, logger):
    """运行命令并处理错误"""
    logger.info(f"执行: {description}")
    logger.info(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"{description} - 成功")
        if result.stdout:
            logger.debug(f"输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} - 失败")
        logger.error(f"错误码: {e.returncode}")
        logger.error(f"错误输出: {e.stderr}")
        return False

def validate_data_directory(data_dir, logger):
    """验证数据目录"""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_path}")
        return False

    # 检查图像文件
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_path.glob(f"*{ext}"))
        image_files.extend(data_path.glob(f"*{ext.upper()}"))

    if len(image_files) == 0:
        logger.error(f"数据目录中没有找到图像文件: {data_path}")
        return False

    logger.info(f"数据验证通过 - 找到 {len(image_files)} 个图像文件")
    return True

def create_sample_images(output_dir, logger, num_samples=5):
    """创建示例图像"""
    logger.info("创建示例图像...")

    # 创建简单的曼哈顿几何图案
    from PIL import Image, ImageDraw
    import numpy as np

    sample_dir = Path(output_dir) / "reference_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        # 创建空白图像
        img = Image.new('L', (256, 256), 255)  # 白色背景
        draw = ImageDraw.Draw(img)

        # 绘制曼哈顿几何图案
        np.random.seed(i)

        # 外框
        draw.rectangle([20, 20, 236, 236], outline=0, width=2)

        # 随机内部矩形
        for _ in range(np.random.randint(3, 8)):
            x1 = np.random.randint(40, 180)
            y1 = np.random.randint(40, 180)
            x2 = x1 + np.random.randint(20, 60)
            y2 = y1 + np.random.randint(20, 60)
            if x2 < 220 and y2 < 220:  # 确保不超出边界
                draw.rectangle([x1, y1, x2, y2], outline=0, width=1)

        # 保存图像
        img.save(sample_dir / f"sample_{i:03d}.png")

    logger.info(f"示例图像已保存到: {sample_dir}")

def run_optimized_pipeline(args):
    """运行优化管线"""
    logger = setup_logging()

    logger.info("=== 开始优化的IC版图扩散模型管线 ===")

    # 验证输入
    if not validate_data_directory(args.data_dir, logger):
        return False

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果需要，创建示例数据
    if args.create_sample_data:
        create_sample_images(args.data_dir, logger)

    # 训练阶段
    if not args.skip_training:
        logger.info("\n=== 第一阶段: 训练优化模型 ===")

        train_cmd = [
            sys.executable, "train_optimized.py",
            "--data_dir", args.data_dir,
            "--output_dir", str(output_dir / "model"),
            "--image_size", str(args.image_size),
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--timesteps", str(args.timesteps),
            "--schedule_type", args.schedule_type,
            "--manhattan_weight", str(args.manhattan_weight),
            "--seed", str(args.seed),
            "--save_interval", str(args.save_interval),
            "--sample_interval", str(args.sample_interval),
            "--num_samples", str(args.train_samples)
        ]

        if args.edge_condition:
            train_cmd.append("--edge_condition")

        if args.augment:
            train_cmd.append("--augment")

        if args.resume:
            train_cmd.extend(["--resume", args.resume])

        success = run_command(train_cmd, "训练优化模型", logger)
        if not success:
            logger.error("训练阶段失败")
            return False

        # 查找最佳模型
        model_checkpoint = output_dir / "model" / "best_model.pth"
        if not model_checkpoint.exists():
            # 如果没有最佳模型，使用最终模型
            model_checkpoint = output_dir / "model" / "final_model.pth"

        if not model_checkpoint.exists():
            logger.error("找不到训练好的模型")
            return False

    else:
        logger.info("\n=== 跳过训练阶段 ===")
        model_checkpoint = args.checkpoint
        if not model_checkpoint:
            logger.error("跳过训练时需要提供 --checkpoint 参数")
            return False

        if not Path(model_checkpoint).exists():
            logger.error(f"指定的检查点不存在: {model_checkpoint}")
            return False

    # 生成阶段
    logger.info("\n=== 第二阶段: 生成样本 ===")

    generate_cmd = [
        sys.executable, "generate_optimized.py",
        "--checkpoint", str(model_checkpoint),
        "--output_dir", str(output_dir / "generated"),
        "--num_samples", str(args.num_samples),
        "--image_size", str(args.image_size),
        "--batch_size", str(args.gen_batch_size),
        "--num_steps", str(args.num_steps),
        "--seed", str(args.seed),
        "--timesteps", str(args.timesteps),
        "--schedule_type", args.schedule_type
    ]

    if args.use_ddim:
        generate_cmd.append("--use_ddim")

    if args.use_post_process:
        generate_cmd.append("--use_post_process")

    success = run_command(generate_cmd, "生成样本", logger)
    if not success:
        logger.error("生成阶段失败")
        return False

    # 更新配置文件（如果提供了）
    if args.update_config and Path(args.update_config).exists():
        logger.info("\n=== 第三阶段: 更新配置文件 ===")

        config_path = Path(args.update_config)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 更新扩散配置
        if 'synthetic' not in config:
            config['synthetic'] = {}

        config['synthetic']['enabled'] = True
        config['synthetic']['ratio'] = 0.0  # 禁用程序化合成

        if 'diffusion' not in config['synthetic']:
            config['synthetic']['diffusion'] = {}

        config['synthetic']['diffusion']['enabled'] = True
        config['synthetic']['diffusion']['png_dir'] = str(output_dir / "generated")
        config['synthetic']['diffusion']['ratio'] = args.diffusion_ratio
        config['synthetic']['diffusion']['model_checkpoint'] = str(model_checkpoint)

        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"配置文件已更新: {config_path}")
        logger.info(f"扩散数据比例: {args.diffusion_ratio}")

    # 创建管线报告
    create_pipeline_report(output_dir, model_checkpoint, args, logger)

    logger.info("\n=== 优化管线完成 ===")
    logger.info(f"模型: {model_checkpoint}")
    logger.info(f"生成数据: {output_dir / 'generated'}")
    logger.info(f"管线报告: {output_dir / 'pipeline_report.txt'}")

    return True

def create_pipeline_report(output_dir, model_checkpoint, args, logger):
    """创建管线报告"""
    report_content = f"""
IC版图扩散模型优化管线报告
============================

管线配置:
- 数据目录: {args.data_dir}
- 输出目录: {args.output_dir}
- 图像尺寸: {args.image_size}x{args.image_size}
- 训练轮数: {args.epochs}
- 批次大小: {args.batch_size}
- 学习率: {args.lr}
- 时间步数: {args.timesteps}
- 调度类型: {args.schedule_type}
- 曼哈顿权重: {args.manhattan_weight}
- 随机种子: {args.seed}

模型配置:
- 边缘条件: {args.edge_condition}
- 数据增强: {args.augment}
- 最终模型: {model_checkpoint}

生成配置:
- 生成样本数: {args.num_samples}
- 生成批次大小: {args.gen_batch_size}
- 采样步数: {args.num_steps}
- DDIM采样: {args.use_ddim}
- 后处理: {args.use_post_process}

优化特性:
- 曼哈顿几何感知的U-Net架构
- 边缘感知损失函数
- 多尺度结构损失
- 曼哈顿约束正则化
- 几何保持的数据增强
- 后处理优化

输出目录结构:
- model/: 训练好的模型和检查点
- generated/: 生成的IC版图样本
- pipeline_report.txt: 本报告

质量评估:
生成完成后，请查看 generated/quality_metrics.yaml 和 generation_report.txt 获取详细的质量评估。

使用说明:
1. 训练数据应包含高质量的IC版图图像
2. 建议使用边缘条件来提高生成质量
3. 生成的样本可以使用后处理进一步优化
4. 可根据质量评估结果调整训练参数
"""

    report_path = output_dir / 'pipeline_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"管线报告已保存: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="一键运行优化的IC版图扩散模型管线")

    # 基本参数
    parser.add_argument("--data_dir", type=str, required=True, help="训练数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    # 训练参数
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--timesteps", type=int, default=1000, help="扩散时间步数")
    parser.add_argument("--schedule_type", type=str, default='cosine', choices=['linear', 'cosine'], help="噪声调度类型")
    parser.add_argument("--manhattan_weight", type=float, default=0.1, help="曼哈顿正则化权重")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument("--sample_interval", type=int, default=20, help="样本生成间隔")
    parser.add_argument("--train_samples", type=int, default=16, help="训练时生成的样本数量")

    # 训练控制
    parser.add_argument("--skip_training", action='store_true', help="跳过训练，使用现有模型")
    parser.add_argument("--checkpoint", type=str, help="现有模型检查点路径（skip_training时使用）")
    parser.add_argument("--resume", type=str, help="恢复训练的检查点路径")
    parser.add_argument("--edge_condition", action='store_true', help="使用边缘条件")
    parser.add_argument("--augment", action='store_true', help="启用数据增强")

    # 生成参数
    parser.add_argument("--num_samples", type=int, default=200, help="生成样本数量")
    parser.add_argument("--gen_batch_size", type=int, default=8, help="生成批次大小")
    parser.add_argument("--num_steps", type=int, default=50, help="采样步数")
    parser.add_argument("--use_ddim", action='store_true', default=True, help="使用DDIM采样")
    parser.add_argument("--use_post_process", action='store_true', default=True, help="启用后处理")

    # 配置更新
    parser.add_argument("--update_config", type=str, help="要更新的配置文件路径")
    parser.add_argument("--diffusion_ratio", type=float, default=0.3, help="扩散数据在训练中的比例")

    # 开发选项
    parser.add_argument("--create_sample_data", action='store_true', help="创建示例训练数据")

    args = parser.parse_args()

    # 验证参数
    if args.skip_training and not args.checkpoint:
        print("错误: 跳过训练时必须提供 --checkpoint 参数")
        sys.exit(1)

    # 运行管线
    success = run_optimized_pipeline(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()