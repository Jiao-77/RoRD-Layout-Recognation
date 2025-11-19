#!/usr/bin/env python3
"""
优化扩散模型使用示例
演示如何使用优化后的IC版图扩散模型
"""

import os
import sys
import torch
import subprocess
from pathlib import Path

def example_basic_training():
    """基本训练示例"""
    print("=== 基本训练示例 ===")

    # 创建示例数据目录
    data_dir = "example_data/ic_layouts"
    output_dir = "example_outputs/basic_training"

    # 训练命令
    cmd = [
        sys.executable, "run_optimized_pipeline.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--epochs", 20,  # 示例用较少轮数
        "--batch_size", 2,
        "--image_size", 256,
        "--num_samples", 10,
        "--create_sample_data",  # 创建示例数据
        "--edge_condition",
        "--augment"
    ]

    print(f"运行命令: {' '.join(map(str, cmd))}")
    print("注意：这将创建示例数据并开始训练")

    # 实际运行时取消注释
    # subprocess.run(cmd)

def example_advanced_training():
    """高级训练示例"""
    print("\n=== 高级训练示例 ===")

    cmd = [
        sys.executable, "train_optimized.py",
        "--data_dir", "data/high_quality_ic_layouts",
        "--output_dir", "models/advanced_diffusion",
        "--image_size", 512,  # 更高分辨率
        "--batch_size", 8,
        "--epochs", 200,
        "--lr", 5e-5,  # 更低学习率
        "--manhattan_weight", 0.15,  # 更强的几何约束
        "--edge_condition",
        "--augment",
        "--schedule_type", "cosine",
        "--save_interval", 5,
        "--sample_interval", 10
    ]

    print(f"高级训练命令: {' '.join(map(str, cmd))}")

def example_generation_only():
    """仅生成示例"""
    print("\n=== 仅生成示例 ===")

    cmd = [
        sys.executable, "generate_optimized.py",
        "--checkpoint", "models/diffusion_optimized/best_model.pth",
        "--output_dir", "generated_samples/high_quality",
        "--num_samples", 100,
        "--num_steps", 30,  # 更快采样
        "--use_ddim",
        "--batch_size", 16,
        "--use_post_process",
        "--post_process_threshold", 0.45
    ]

    print(f"生成命令: {' '.join(map(str, cmd))}")

def example_custom_parameters():
    """自定义参数示例"""
    print("\n=== 自定义参数示例 ===")

    # 针对特定需求的参数调整
    scenarios = {
        "高质量生成": {
            "description": "追求最高质量的生成结果",
            "params": {
                "--num_steps": 100,
                "--guidance_scale": 2.0,
                "--eta": 0.0,  # 完全确定性
                "--use_post_process": True,
                "--post_process_threshold": 0.5
            }
        },
        "快速生成": {
            "description": "快速生成大量样本",
            "params": {
                "--num_steps": 20,
                "--batch_size": 32,
                "--eta": 0.3,  # 增加随机性
                "--use_ddim": True
            }
        },
        "几何约束严格": {
            "description": "严格要求曼哈顿几何",
            "params": {
                "--manhattan_weight": 0.3,  # 更强约束
                "--use_post_process": True,
                "--post_process_threshold": 0.4
            }
        }
    }

    for scenario_name, config in scenarios.items():
        print(f"\n{scenario_name}: {config['description']}")
        for param, value in config['params'].items():
            print(f"  {param}: {value}")

def example_integration():
    """集成到现有管线示例"""
    print("\n=== 集成示例 ===")

    # 更新配置文件
    config_update = {
        "config_file": "configs/train_config.yaml",
        "updates": {
            "synthetic.enabled": True,
            "synthetic.ratio": 0.0,
            "synthetic.diffusion.enabled": True,
            "synthetic.diffusion.png_dir": "outputs/diffusion_optimized/generated",
            "synthetic.diffusion.ratio": 0.4,
            "synthetic.diffusion.model_checkpoint": "outputs/diffusion_optimized/model/best_model.pth"
        }
    }

    print("配置文件更新示例:")
    print(f"配置文件: {config_update['config_file']}")
    for key, value in config_update['updates'].items():
        print(f"  {key}: {value}")

    integration_cmd = [
        sys.executable, "run_optimized_pipeline.py",
        "--data_dir", "data/training_layouts",
        "--output_dir", "outputs/integration",
        "--update_config", config_update["config_file"],
        "--diffusion_ratio", 0.4,
        "--epochs", 100,
        "--num_samples", 500
    ]

    print(f"\n集成命令: {' '.join(map(str, integration_cmd))}")

def show_tips():
    """显示使用建议"""
    print("\n=== 使用建议 ===")

    tips = [
        "🎯 数据质量是关键：使用高质量、多样化的IC版图数据进行训练",
        "⚖️ 平衡约束：曼哈顿权重不宜过高（0.05-0.2），避免过度约束影响生成多样性",
        "🔄 迭代优化：根据生成结果调整损失函数权重和后处理参数",
        "📊 质量监控：定期检查生成样本的质量指标",
        "💾 定期保存：设置合理的保存间隔，避免训练中断导致损失",
        "🚀 性能优化：使用DDIM采样可以显著提高生成速度",
        "🔧 参数调优：根据具体任务需求调整模型参数"
    ]

    for tip in tips:
        print(tip)

def main():
    """主函数"""
    print("IC版图扩散模型优化版本 - 使用示例")
    print("=" * 50)

    # 检查是否在正确的目录
    if not Path("ic_layout_diffusion_optimized.py").exists():
        print("错误：请在 tools/diffusion/ 目录下运行此脚本")
        sys.exit(1)

    # 显示示例
    example_basic_training()
    example_advanced_training()
    example_generation_only()
    example_custom_parameters()
    example_integration()
    show_tips()

    print("\n" + "=" * 50)
    print("运行示例：")
    print("1. 基本使用：python run_optimized_pipeline.py --data_dir data/ic_layouts --output_dir outputs")
    print("2. 查看完整参数：python train_optimized.py --help")
    print("3. 查看生成参数：python generate_optimized.py --help")
    print("4. 阅读详细文档：README_OPTIMIZED.md")

if __name__ == "__main__":
    main()