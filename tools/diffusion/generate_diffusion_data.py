#!/usr/bin/env python3
"""
一键生成扩散数据的脚本：
1. 基于原始数据训练扩散模型
2. 使用训练好的模型生成相似图像
3. 更新配置文件
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
import logging


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def train_diffusion_model(data_dir, model_dir, logger, **train_kwargs):
    """训练扩散模型"""
    logger.info("开始训练扩散模型...")

    # 构建训练命令
    cmd = [
        sys.executable, "tools/diffusion/ic_layout_diffusion.py", "train",
        "--data_dir", data_dir,
        "--output_dir", model_dir,
        "--image_size", str(train_kwargs.get("image_size", 256)),
        "--batch_size", str(train_kwargs.get("batch_size", 8)),
        "--epochs", str(train_kwargs.get("epochs", 100)),
        "--lr", str(train_kwargs.get("lr", 1e-4)),
        "--timesteps", str(train_kwargs.get("timesteps", 1000)),
        "--num_samples", str(train_kwargs.get("num_samples", 50)),
        "--save_interval", str(train_kwargs.get("save_interval", 10))
    ]

    if train_kwargs.get("augment", False):
        cmd.append("--augment")

    # 执行训练
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"扩散模型训练失败: {result.stderr}")
        return False

    logger.info("扩散模型训练完成")
    return True


def generate_samples(model_dir, output_dir, num_samples, logger, **gen_kwargs):
    """生成样本"""
    logger.info(f"开始生成 {num_samples} 个样本...")

    # 查找最终模型
    model_path = Path(model_dir) / "diffusion_final.pth"
    if not model_path.exists():
        # 如果没有最终模型，查找最新的检查点
        checkpoints = list(Path(model_dir).glob("diffusion_epoch_*.pth"))
        if checkpoints:
            model_path = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        else:
            logger.error(f"在 {model_dir} 中找不到模型检查点")
            return False

    logger.info(f"使用模型: {model_path}")

    # 构建生成命令
    cmd = [
        sys.executable, "tools/diffusion/ic_layout_diffusion.py", "generate",
        "--checkpoint", str(model_path),
        "--output_dir", output_dir,
        "--num_samples", str(num_samples),
        "--image_size", str(gen_kwargs.get("image_size", 256)),
        "--timesteps", str(gen_kwargs.get("timesteps", 1000))
    ]

    # 执行生成
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"样本生成失败: {result.stderr}")
        return False

    logger.info("样本生成完成")
    return True


def update_config(config_path, output_dir, ratio, logger):
    """更新配置文件"""
    logger.info(f"更新配置文件: {config_path}")

    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 确保必要的结构存在
    if 'synthetic' not in config:
        config['synthetic'] = {}

    # 更新扩散配置
    config['synthetic']['enabled'] = True
    config['synthetic']['ratio'] = 0.0  # 禁用程序化合成

    if 'diffusion' not in config['synthetic']:
        config['synthetic']['diffusion'] = {}

    config['synthetic']['diffusion']['enabled'] = True
    config['synthetic']['diffusion']['png_dir'] = output_dir
    config['synthetic']['diffusion']['ratio'] = ratio

    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"配置文件更新完成，扩散数据比例: {ratio}")


def validate_generated_data(output_dir, logger):
    """验证生成的数据"""
    logger.info("验证生成的数据...")

    output_path = Path(output_dir)
    if not output_path.exists():
        logger.error(f"输出目录不存在: {output_dir}")
        return False

    # 统计生成的图像
    png_files = list(output_path.glob("*.png"))
    if not png_files:
        logger.error("没有找到生成的PNG图像")
        return False

    logger.info(f"验证通过，生成了 {len(png_files)} 个图像")
    return True


def main():
    parser = argparse.ArgumentParser(description="一键生成扩散数据管线")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--data_dir", type=str, help="原始数据目录（覆盖配置文件）")
    parser.add_argument("--model_dir", type=str, default="models/diffusion", help="扩散模型保存目录")
    parser.add_argument("--output_dir", type=str, default="data/diffusion_generated", help="生成数据保存目录")
    parser.add_argument("--num_samples", type=int, default=200, help="生成的样本数量")
    parser.add_argument("--ratio", type=float, default=0.3, help="扩散数据在训练中的比例")
    parser.add_argument("--skip_training", action="store_true", help="跳过训练，直接生成")
    parser.add_argument("--model_checkpoint", type=str, help="指定模型检查点路径（skip_training时使用）")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--augment", action="store_true", help="启用数据增强")

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging()

    # 读取配置文件获取数据目录
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 确定数据目录
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # 从配置文件获取数据目录
        config_dir = config_path.parent
        layout_dir = config.get('paths', {}).get('layout_dir', 'data/layouts')
        data_dir = str(config_dir / layout_dir)

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_path}")
        return False

    logger.info(f"使用数据目录: {data_path}")
    logger.info(f"模型保存目录: {args.model_dir}")
    logger.info(f"生成数据目录: {args.output_dir}")
    logger.info(f"生成样本数量: {args.num_samples}")
    logger.info(f"训练比例: {args.ratio}")

    # 1. 训练扩散模型（如果需要）
    if not args.skip_training:
        success = train_diffusion_model(
            data_dir=data_dir,
            model_dir=args.model_dir,
            logger=logger,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_samples=args.num_samples,
            augment=args.augment
        )
        if not success:
            logger.error("扩散模型训练失败")
            return False
    else:
        logger.info("跳过训练步骤")

    # 2. 生成样本
    success = generate_samples(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        logger=logger,
        image_size=args.image_size
    )
    if not success:
        logger.error("样本生成失败")
        return False

    # 3. 验证生成的数据
    if not validate_generated_data(args.output_dir, logger):
        logger.error("数据验证失败")
        return False

    # 4. 更新配置文件
    update_config(
        config_path=args.config,
        output_dir=args.output_dir,
        ratio=args.ratio,
        logger=logger
    )

    logger.info("=== 扩散数据生成管线完成 ===")
    logger.info(f"生成数据位置: {args.output_dir}")
    logger.info(f"配置文件已更新: {args.config}")
    logger.info(f"扩散数据比例: {args.ratio}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)