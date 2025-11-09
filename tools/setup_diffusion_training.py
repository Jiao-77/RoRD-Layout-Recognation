#!/usr/bin/env python3
"""
一键设置扩散训练流程的脚本

此脚本帮助用户：
1. 检查环境
2. 生成扩散数据
3. 配置训练参数
4. 启动训练
"""

import sys
import argparse
import yaml
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


def check_environment(logger):
    """检查运行环境"""
    logger.info("检查运行环境...")

    # 检查Python包
    required_packages = ['torch', 'torchvision', 'numpy', 'PIL', 'yaml']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} 未安装")

    if missing_packages:
        logger.error(f"缺少必需的包: {missing_packages}")
        logger.info("请安装缺少的包：pip install " + " ".join(missing_packages))
        return False

    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA 可用，设备数量: {torch.cuda.device_count()}")
        else:
            logger.warning("✗ CUDA 不可用，将使用CPU训练（速度较慢）")
    except Exception as e:
        logger.warning(f"无法检查CUDA状态: {e}")

    logger.info("环境检查完成")
    return True


def create_sample_config(config_path, logger):
    """创建示例配置文件"""
    logger.info("创建示例配置文件...")

    config = {
        'training': {
            'learning_rate': 5e-5,
            'batch_size': 8,
            'num_epochs': 50,
            'patch_size': 256,
            'scale_jitter_range': [0.8, 1.2]
        },
        'model': {
            'fpn': {
                'enabled': True,
                'out_channels': 256,
                'levels': [2, 3, 4],
                'norm': 'bn'
            },
            'backbone': {
                'name': 'vgg16',
                'pretrained': False
            },
            'attention': {
                'enabled': False,
                'type': 'none',
                'places': []
            }
        },
        'paths': {
            'layout_dir': 'data/layouts',  # 原始数据目录
            'save_dir': 'models/rord',
            'val_img_dir': 'data/val/images',
            'val_ann_dir': 'data/val/annotations',
            'template_dir': 'data/templates',
            'model_path': 'models/rord/rord_model_best.pth'
        },
        'data_sources': {
            'real': {
                'enabled': True,
                'ratio': 0.7  # 70% 真实数据
            },
            'diffusion': {
                'enabled': True,
                'model_dir': 'models/diffusion',
                'png_dir': 'data/diffusion_generated',
                'ratio': 0.3,  # 30% 扩散数据
                'training': {
                    'epochs': 100,
                    'batch_size': 8,
                    'lr': 1e-4,
                    'image_size': 256,
                    'timesteps': 1000,
                    'augment': True
                },
                'generation': {
                    'num_samples': 200,
                    'timesteps': 1000
                }
            }
        },
        'logging': {
            'use_tensorboard': True,
            'log_dir': 'runs',
            'experiment_name': 'diffusion_training'
        }
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"示例配置文件已创建: {config_path}")
    return True


def setup_directories(logger):
    """创建必要的目录"""
    logger.info("创建目录结构...")

    directories = [
        'data/layouts',
        'data/diffusion_generated',
        'models/diffusion',
        'models/rord',
        'runs',
        'logs'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ {directory}")

    logger.info("目录结构创建完成")
    return True


def run_diffusion_pipeline(config_path, logger):
    """运行扩散数据生成流程"""
    logger.info("运行扩散数据生成流程...")

    cmd = [
        sys.executable, "tools/diffusion/generate_diffusion_data.py",
        "--config", config_path,
        "--data_dir", "data/layouts",
        "--model_dir", "models/diffusion",
        "--output_dir", "data/diffusion_generated",
        "--num_samples", "200",
        "--ratio", "0.3"
    ]

    logger.info(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"扩散数据生成失败: {result.stderr}")
        return False

    logger.info("扩散数据生成完成")
    return True


def start_training(config_path, logger):
    """启动训练"""
    logger.info("启动模型训练...")

    cmd = [
        sys.executable, "train.py",
        "--config", config_path
    ]

    logger.info(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)  # 实时显示输出

    if result.returncode != 0:
        logger.error("训练失败")
        return False

    logger.info("训练完成")
    return True


def main():
    parser = argparse.ArgumentParser(description="一键设置扩散训练流程")
    parser.add_argument("--config", type=str, default="configs/diffusion_config.yaml", help="配置文件路径")
    parser.add_argument("--skip_env_check", action="store_true", help="跳过环境检查")
    parser.add_argument("--skip_diffusion", action="store_true", help="跳过扩散数据生成")
    parser.add_argument("--skip_training", action="store_true", help="跳过模型训练")
    parser.add_argument("--only_check", action="store_true", help="仅检查环境")

    args = parser.parse_args()

    logger = setup_logging()

    logger.info("=== RoRD 扩散训练流程设置 ===")

    # 1. 环境检查
    if not args.skip_env_check:
        if not check_environment(logger):
            logger.error("环境检查失败")
            return False

    if args.only_check:
        logger.info("环境检查完成")
        return True

    # 2. 创建目录结构
    if not setup_directories(logger):
        logger.error("目录创建失败")
        return False

    # 3. 创建示例配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        if not create_sample_config(args.config, logger):
            logger.error("配置文件创建失败")
            return False
    else:
        logger.info(f"使用现有配置文件: {config_path}")

    # 4. 运行扩散数据生成流程
    if not args.skip_diffusion:
        if not run_diffusion_pipeline(args.config, logger):
            logger.error("扩散数据生成失败")
            return False
    else:
        logger.info("跳过扩散数据生成")

    # 5. 启动训练
    if not args.skip_training:
        if not start_training(args.config, logger):
            logger.error("训练失败")
            return False
    else:
        logger.info("跳过模型训练")

    logger.info("=== 扩散训练流程设置完成 ===")
    logger.info("您可以查看以下文件和目录：")
    logger.info(f"配置文件: {args.config}")
    logger.info("扩散模型: models/diffusion/")
    logger.info("生成数据: data/diffusion_generated/")
    logger.info("训练模型: models/rord/")
    logger.info("训练日志: runs/")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)