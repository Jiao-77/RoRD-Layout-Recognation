#!/usr/bin/env python3
"""
测试修复后的IC版图扩散模型
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 导入修复后的模型
from ic_layout_diffusion_optimized import (
    ManhattanAwareUNet,
    OptimizedNoiseScheduler,
    OptimizedDiffusionTrainer,
    ICDiffusionDataset
)

def test_model_architecture():
    """测试模型架构修复"""
    print("测试模型架构...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 测试不同的配置
    test_configs = [
        {'use_edge_condition': False, 'batch_size': 2},
        {'use_edge_condition': True, 'batch_size': 2},
    ]

    for config in test_configs:
        print(f"\n测试配置: {config}")

        try:
            # 创建模型
            model = ManhattanAwareUNet(
                in_channels=1,
                out_channels=1,
                time_dim=256,
                use_edge_condition=config['use_edge_condition']
            ).to(device)

            # 创建测试数据
            batch_size = config['batch_size']
            image_size = 64  # 使用较小的图像尺寸进行快速测试

            x = torch.randn(batch_size, 1, image_size, image_size).to(device)
            t = torch.randint(0, 1000, (batch_size,)).to(device)

            if config['use_edge_condition']:
                edge_condition = torch.randn(batch_size, 1, image_size, image_size).to(device)
                output = model(x, t, edge_condition)
            else:
                output = model(x, t)

            print(f"✓ 输入形状: {x.shape}")
            print(f"✓ 输出形状: {output.shape}")
            print(f"✓ 时间步形状: {t.shape}")

            # 检查输出形状
            expected_shape = (batch_size, 1, image_size, image_size)
            if output.shape == expected_shape:
                print(f"✓ 输出形状正确: {output.shape}")
            else:
                print(f"✗ 输出形状错误: 期望 {expected_shape}, 得到 {output.shape}")

        except Exception as e:
            print(f"✗ 模型测试失败: {e}")
            return False

    print("\n✓ 所有模型架构测试通过!")
    return True

def test_scheduler():
    """测试噪声调度器修复"""
    print("\n测试噪声调度器...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        scheduler = OptimizedNoiseScheduler(num_timesteps=1000, schedule_type='cosine')

        # 测试噪声添加
        x_0 = torch.randn(4, 1, 32, 32).to(device)
        t = torch.randint(0, 1000, (4,)).to(device)

        x_t, noise = scheduler.add_noise(x_0, t)

        print(f"✓ 原始图像形状: {x_0.shape}")
        print(f"✓ 噪声图像形状: {x_t.shape}")
        print(f"✓ 噪声形状: {noise.shape}")

        # 测试去噪步骤
        model = ManhattanAwareUNet().to(device)
        x_denoised = scheduler.step(model, x_t, t)

        print(f"✓ 去噪图像形状: {x_denoised.shape}")

        # 检查形状是否保持一致
        if x_denoised.shape == x_t.shape:
            print("✓ 去噪步骤形状正确")
        else:
            print(f"✗ 去噪步骤形状错误: 期望 {x_t.shape}, 得到 {x_denoised.shape}")
            return False

    except Exception as e:
        print(f"✗ 调度器测试失败: {e}")
        return False

    print("✓ 调度器测试通过!")
    return True

def test_trainer():
    """测试训练器修复"""
    print("\n测试训练器...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 创建模型和调度器
        model = ManhattanAwareUNet().to(device)
        scheduler = OptimizedNoiseScheduler(num_timesteps=100)
        trainer = OptimizedDiffusionTrainer(model, scheduler, device, use_edge_condition=False)

        # 创建虚拟数据
        batch_size = 2
        images = torch.randn(batch_size, 1, 32, 32).to(device)

        # 测试单个训练步骤
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # 模拟数据加载器
        class MockDataloader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                yield self.data

        mock_dataloader = MockDataloader(images)

        # 执行训练步骤
        losses = trainer.train_step(optimizer, mock_dataloader, manhattan_weight=0.1)

        print(f"✓ 训练步骤完成，损失: {losses}")

        # 测试生成
        samples = trainer.generate(
            num_samples=2,
            image_size=32,
            save_dir=None,
            use_post_process=False
        )

        print(f"✓ 生成样本形状: {samples.shape}")

    except Exception as e:
        print(f"✗ 训练器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✓ 训练器测试通过!")
    return True

def test_data_loading():
    """测试数据加载修复"""
    print("\n测试数据加载...")

    # 创建临时测试目录（如果不存在）
    test_dir = Path("test_images")

    try:
        if not test_dir.exists():
            print("创建测试图像目录...")
            test_dir.mkdir(exist_ok=True)

            # 创建一些简单的测试图像
            from PIL import Image
            import numpy as np

            for i in range(3):
                # 创建随机灰度图像
                img_array = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
                img = Image.fromarray(img_array, mode='L')
                img.save(test_dir / f"test_{i}.png")

        # 测试数据集创建
        dataset = ICDiffusionDataset(
            image_dir=str(test_dir),
            image_size=32,
            augment=False,
            use_edge_condition=False
        )

        print(f"✓ 数据集大小: {len(dataset)}")

        if len(dataset) == 0:
            print("✗ 数据集为空")
            return False

        # 测试数据加载
        sample = dataset[0]
        print(f"✓ 样本形状: {sample.shape}")

        # 测试边缘条件
        dataset_edge = ICDiffusionDataset(
            image_dir=str(test_dir),
            image_size=32,
            augment=False,
            use_edge_condition=True
        )

        image, edge = dataset_edge[0]
        print(f"✓ 图像形状: {image.shape}, 边缘形状: {edge.shape}")

    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✓ 数据加载测试通过!")
    return True

def main():
    """运行所有测试"""
    print("开始测试修复后的IC版图扩散模型...")
    print("=" * 50)

    all_tests_passed = True

    # 运行各项测试
    tests = [
        ("模型架构", test_model_architecture),
        ("噪声调度器", test_scheduler),
        ("训练器", test_trainer),
        ("数据加载", test_data_loading),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if not test_func():
            all_tests_passed = False

    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 所有测试通过！模型修复成功。")
    else:
        print("❌ 部分测试失败，需要进一步修复。")

    return all_tests_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)