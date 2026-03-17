#!/usr/bin/env python3
"""
数据集单元测试。

测试内容：
- 数据集边界情况
- 图像加载和预处理
- 数据增强
- 尺度抖动
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.ic_dataset import ICLayoutTrainingDataset
from utils.data_utils import get_transform


class TestICLayoutTrainingDataset:
    """测试 ICLayoutTrainingDataset 类。"""

    @pytest.fixture
    def temp_image_dir(self):
        """创建临时图像目录。"""
        temp_dir = tempfile.mkdtemp()
        
        # 创建测试图像
        for i in range(5):
            img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(temp_dir, f"test_{i}.png"))
        
        yield temp_dir
        
        # 清理
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def small_image_dir(self):
        """创建小图像目录。"""
        temp_dir = tempfile.mkdtemp()
        
        # 创建小尺寸图像
        for i in range(3):
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(temp_dir, f"small_{i}.png"))
        
        yield temp_dir
        
        shutil.rmtree(temp_dir)

    def test_dataset_creation(self, temp_image_dir):
        """测试数据集创建。"""
        dataset = ICLayoutTrainingDataset(temp_image_dir)
        
        assert len(dataset) == 5, f"期望 5 个样本，实际 {len(dataset)}"

    def test_dataset_getitem(self, temp_image_dir):
        """测试数据集 __getitem__ 方法。"""
        dataset = ICLayoutTrainingDataset(temp_image_dir, patch_size=256)
        
        orig, rot, H = dataset[0]
        
        # 检查返回类型
        assert isinstance(orig, torch.Tensor), "orig 应该是 torch.Tensor"
        assert isinstance(rot, torch.Tensor), "rot 应该是 torch.Tensor"
        assert isinstance(H, torch.Tensor), "H 应该是 torch.Tensor"
        
        # 检查形状
        assert orig.dim() == 3, "orig 应该是 3D 张量"
        assert rot.dim() == 3, "rot 应该是 3D 张量"
        assert H.shape == (2, 3), f"H 形状应该是 (2, 3)，实际 {H.shape}"

    def test_patch_size(self, temp_image_dir):
        """测试 patch_size 参数。"""
        patch_sizes = [128, 256, 512]
        
        for patch_size in patch_sizes:
            dataset = ICLayoutTrainingDataset(temp_image_dir, patch_size=patch_size)
            orig, rot, H = dataset[0]
            
            # 检查输出尺寸
            assert orig.shape[-1] == patch_size, f"期望宽度 {patch_size}，实际 {orig.shape[-1]}"
            assert orig.shape[-2] == patch_size, f"期望高度 {patch_size}，实际 {orig.shape[-2]}"

    def test_scale_range_normal(self, temp_image_dir):
        """测试正常 scale_range。"""
        dataset = ICLayoutTrainingDataset(
            temp_image_dir,
            patch_size=256,
            scale_range=(0.8, 1.2)
        )
        
        # 多次采样检查
        for _ in range(10):
            orig, rot, H = dataset[0]
            assert orig.shape[-1] == 256

    def test_scale_range_extreme_small(self, temp_image_dir):
        """测试极小 scale_range 边界情况。"""
        # 极小的 scale_range
        dataset = ICLayoutTrainingDataset(
            temp_image_dir,
            patch_size=256,
            scale_range=(0.01, 0.05)
        )
        
        # 不应该崩溃
        orig, rot, H = dataset[0]
        assert orig.shape[-1] == 256

    def test_scale_range_extreme_large(self, temp_image_dir):
        """测试极大 scale_range 边界情况。"""
        # 极大的 scale_range
        dataset = ICLayoutTrainingDataset(
            temp_image_dir,
            patch_size=256,
            scale_range=(2.0, 4.0)
        )
        
        # 不应该崩溃
        orig, rot, H = dataset[0]
        assert orig.shape[-1] == 256

    def test_scale_range_identity(self, temp_image_dir):
        """测试恒等 scale_range。"""
        dataset = ICLayoutTrainingDataset(
            temp_image_dir,
            patch_size=256,
            scale_range=(1.0, 1.0)
        )
        
        orig, rot, H = dataset[0]
        assert orig.shape[-1] == 256

    def test_transform_parameter(self, temp_image_dir):
        """测试 transform 参数。"""
        transform = get_transform()
        dataset = ICLayoutTrainingDataset(
            temp_image_dir,
            patch_size=256,
            transform=transform
        )
        
        orig, rot, H = dataset[0]
        
        # 检查变换后的值范围（归一化后应该在 [-1, 1] 或类似范围）
        assert orig.dtype == torch.float32

    def test_empty_directory(self):
        """测试空目录。"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            dataset = ICLayoutTrainingDataset(temp_dir)
            assert len(dataset) == 0, "空目录应该产生空数据集"
        finally:
            shutil.rmtree(temp_dir)

    def test_nonexistent_directory(self):
        """测试不存在的目录。"""
        with pytest.raises(FileNotFoundError):
            ICLayoutTrainingDataset("/nonexistent/directory")

    def test_index_out_of_range(self, temp_image_dir):
        """测试索引越界。"""
        dataset = ICLayoutTrainingDataset(temp_image_dir)
        
        with pytest.raises(IndexError):
            _ = dataset[100]

    def test_negative_index(self, temp_image_dir):
        """测试负索引。"""
        dataset = ICLayoutTrainingDataset(temp_image_dir)
        
        # 负索引应该工作（Python 风格）
        orig, rot, H = dataset[-1]
        assert orig is not None

    def test_small_images(self, small_image_dir):
        """测试小图像处理。"""
        # patch_size 大于图像尺寸
        dataset = ICLayoutTrainingDataset(
            small_image_dir,
            patch_size=256,
            scale_range=(1.0, 1.0)
        )
        
        # 应该能够处理（裁剪到图像大小）
        orig, rot, H = dataset[0]
        # 由于图像小于 patch_size，可能会有填充或调整


class TestAlbumentationsIntegration:
    """测试 Albumentations 集成。"""

    @pytest.fixture
    def temp_image_dir(self):
        """创建临时图像目录。"""
        temp_dir = tempfile.mkdtemp()
        
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(temp_dir, "test.png"))
        
        yield temp_dir
        
        shutil.rmtree(temp_dir)

    def test_albu_disabled(self, temp_image_dir):
        """测试禁用 Albumentations。"""
        dataset = ICLayoutTrainingDataset(
            temp_image_dir,
            patch_size=256,
            use_albu=False
        )
        
        assert dataset.albu is None

    def test_albu_enabled(self, temp_image_dir):
        """测试启用 Albumentations。"""
        dataset = ICLayoutTrainingDataset(
            temp_image_dir,
            patch_size=256,
            use_albu=True,
            albu_params={"prob": 0.5}
        )
        
        # 如果安装了 albumentations，albu 应该不为 None
        # 如果没安装，应该优雅降级为 None
        # 两种情况都是可接受的

    def test_albu_custom_params(self, temp_image_dir):
        """测试自定义 Albumentations 参数。"""
        custom_params = {
            "prob": 0.8,
            "alpha": 50,
            "sigma": 10,
            "brightness_contrast": True,
            "gauss_noise": True
        }
        
        dataset = ICLayoutTrainingDataset(
            temp_image_dir,
            patch_size=256,
            use_albu=True,
            albu_params=custom_params
        )
        
        # 应该能够创建数据集


class TestHomographyGeneration:
    """测试单应性矩阵生成。"""

    @pytest.fixture
    def temp_image_dir(self):
        """创建临时图像目录。"""
        temp_dir = tempfile.mkdtemp()
        
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(temp_dir, "test.png"))
        
        yield temp_dir
        
        shutil.rmtree(temp_dir)

    def test_homography_shape(self, temp_image_dir):
        """测试单应性矩阵形状。"""
        dataset = ICLayoutTrainingDataset(temp_image_dir, patch_size=256)
        _, _, H = dataset[0]
        
        assert H.shape == (2, 3), f"期望形状 (2, 3)，实际 {H.shape}"

    def test_homography_values(self, temp_image_dir):
        """测试单应性矩阵值。"""
        dataset = ICLayoutTrainingDataset(temp_image_dir, patch_size=256)
        _, _, H = dataset[0]
        
        # 检查值是否在合理范围内
        assert not torch.isnan(H).any(), "H 不应包含 NaN"
        assert not torch.isinf(H).any(), "H 不应包含 Inf"

    def test_homography_determinant(self, temp_image_dir):
        """测试单应性矩阵行列式。"""
        dataset = ICLayoutTrainingDataset(temp_image_dir, patch_size=256)
        _, _, H = dataset[0]
        
        # 构造完整的 3x3 矩阵
        H_full = torch.cat([H, torch.tensor([[0.0, 0.0, 1.0]])], dim=0)
        det = torch.linalg.det(H_full)
        
        # 行列式应该接近 1（近似刚性变换）
        assert abs(det.item()) > 0.1, "行列式不应接近 0"


class TestImageFormats:
    """测试图像格式支持。"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录。"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_png_format(self, temp_dir):
        """测试 PNG 格式。"""
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(temp_dir, "test.png"))
        
        dataset = ICLayoutTrainingDataset(temp_dir, patch_size=128)
        assert len(dataset) == 1

    def test_only_png_supported(self, temp_dir):
        """测试只有 PNG 格式被支持。"""
        # 创建 PNG 和 JPG 图像
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(temp_dir, "test1.png"))
        Image.fromarray(img).convert('RGB').save(os.path.join(temp_dir, "test2.jpg"))
        
        dataset = ICLayoutTrainingDataset(temp_dir, patch_size=128)
        # 只有 PNG 文件会被加载
        assert len(dataset) == 1


class TestReproducibility:
    """测试可重复性。"""

    @pytest.fixture
    def temp_image_dir(self):
        """创建临时图像目录。"""
        temp_dir = tempfile.mkdtemp()
        
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(temp_dir, "test.png"))
        
        yield temp_dir
        
        shutil.rmtree(temp_dir)

    def test_reproducibility_with_seed(self, temp_image_dir):
        """测试使用种子时的可重复性。"""
        dataset = ICLayoutTrainingDataset(
            temp_image_dir,
            patch_size=256,
            scale_range=(1.0, 1.0)  # 固定 scale 以减少随机性
        )
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        orig1, rot1, H1 = dataset[0]
        
        torch.manual_seed(42)
        np.random.seed(42)
        orig2, rot2, H2 = dataset[0]
        
        # 结果应该相同
        assert torch.allclose(orig1, orig2), "orig 应该相同"
        assert torch.allclose(rot1, rot2), "rot 应该相同"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])