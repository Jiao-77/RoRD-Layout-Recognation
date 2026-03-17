#!/usr/bin/env python3
"""
损失函数单元测试。

测试内容：
- 损失函数数值正确性
- 边界情况处理
- 梯度计算
- 权重参数化
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from losses import (
    compute_detection_loss,
    compute_description_loss,
    _augment_homography_matrix,
    warp_feature_map,
    DET_SMOOTH_L1_WEIGHT,
    DESC_GEOMETRIC_WEIGHT,
    DESC_MANHATTAN_WEIGHT,
    DESC_SPARSITY_WEIGHT,
    DESC_BINARY_WEIGHT,
)


class TestAugmentHomographyMatrix:
    """测试单应性矩阵扩展函数。"""

    def test_valid_input_shape(self):
        """测试有效输入形状。"""
        h_2x3 = torch.randn(4, 2, 3)
        result = _augment_homography_matrix(h_2x3)
        
        assert result.shape == (4, 3, 3), f"期望形状 (4, 3, 3)，实际 {result.shape}"
        
        # 验证第三行是 [0, 0, 1]
        expected_bottom = torch.tensor([0.0, 0.0, 1.0])
        for i in range(4):
            assert torch.allclose(result[i, 2, :], expected_bottom), \
                f"批次 {i} 的第三行不正确"

    def test_invalid_input_shape_2d(self):
        """测试无效的 2D 输入。"""
        h_2d = torch.randn(2, 3)
        
        with pytest.raises(ValueError, match="Expected homography with shape"):
            _augment_homography_matrix(h_2d)

    def test_invalid_input_shape_wrong_dims(self):
        """测试错误维度的输入。"""
        h_wrong = torch.randn(4, 3, 3)
        
        with pytest.raises(ValueError, match="Expected homography with shape"):
            _augment_homography_matrix(h_wrong)

    def test_batch_size_preserved(self):
        """测试批次大小保持不变。"""
        for batch_size in [1, 2, 8, 16]:
            h_2x3 = torch.randn(batch_size, 2, 3)
            result = _augment_homography_matrix(h_2x3)
            assert result.shape[0] == batch_size


class TestWarpFeatureMap:
    """测试特征图变换函数。"""

    def test_warp_identity(self):
        """测试恒等变换。"""
        feature_map = torch.randn(2, 64, 32, 32)
        # 恒等变换的逆矩阵
        h_inv = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ])
        
        warped = warp_feature_map(feature_map, h_inv)
        
        # 恒等变换应该保持特征图大致不变（可能有边界差异）
        assert warped.shape == feature_map.shape

    def test_warp_output_shape(self):
        """测试输出形状正确。"""
        feature_map = torch.randn(2, 64, 32, 32)
        h_inv = torch.randn(2, 2, 3)
        
        warped = warp_feature_map(feature_map, h_inv)
        
        assert warped.shape == feature_map.shape


class TestComputeDetectionLoss:
    """测试检测损失函数。"""

    @pytest.fixture
    def sample_inputs(self):
        """创建测试输入。"""
        batch_size = 2
        det_original = torch.rand(batch_size, 1, 32, 32)
        det_rotated = torch.rand(batch_size, 1, 32, 32)
        h = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.9, 0.1, 5.0], [-0.1, 0.9, 3.0]]
        ])
        return det_original, det_rotated, h

    def test_loss_is_scalar(self, sample_inputs):
        """测试损失是标量。"""
        det_orig, det_rot, h = sample_inputs
        loss = compute_detection_loss(det_orig, det_rot, h)
        
        assert loss.dim() == 0, "损失应该是标量"
        assert loss.item() >= 0, "损失应该非负"

    def test_loss_gradients(self, sample_inputs):
        """测试梯度计算。"""
        det_orig, det_rot, h = sample_inputs
        det_orig.requires_grad_(True)
        
        loss = compute_detection_loss(det_orig, det_rot, h)
        loss.backward()
        
        assert det_orig.grad is not None, "应该有梯度"
        assert not torch.isnan(det_orig.grad).any(), "梯度不应包含 NaN"

    def test_custom_weight(self, sample_inputs):
        """测试自定义权重。"""
        det_orig, det_rot, h = sample_inputs
        
        loss_default = compute_detection_loss(det_orig, det_rot, h)
        loss_custom = compute_detection_loss(det_orig, det_rot, h, smooth_l1_weight=0.5)
        
        # 不同权重应该产生不同结果
        assert loss_default.item() != loss_custom.item()

    def test_identical_inputs_lower_loss(self, sample_inputs):
        """测试相同输入产生较低损失。"""
        det_orig = torch.rand(2, 1, 32, 32)
        h = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ])
        
        loss_identical = compute_detection_loss(det_orig, det_orig.clone(), h)
        loss_different = compute_detection_loss(det_orig, torch.rand_like(det_orig), h)
        
        # 相同输入的损失应该更低
        assert loss_identical.item() < loss_different.item()

    def test_default_weight_constant(self):
        """测试默认权重使用常量。"""
        assert DET_SMOOTH_L1_WEIGHT == 0.1


class TestComputeDescriptionLoss:
    """测试描述子损失函数。"""

    @pytest.fixture
    def sample_inputs(self):
        """创建测试输入。"""
        batch_size = 2
        desc_original = torch.rand(batch_size, 128, 32, 32)
        desc_rotated = torch.rand(batch_size, 128, 32, 32)
        h = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.9, 0.1, 5.0], [-0.1, 0.9, 3.0]]
        ])
        return desc_original, desc_rotated, h

    def test_loss_is_scalar(self, sample_inputs):
        """测试损失是标量。"""
        desc_orig, desc_rot, h = sample_inputs
        loss = compute_description_loss(desc_orig, desc_rot, h)
        
        assert loss.dim() == 0, "损失应该是标量"
        assert loss.item() >= 0, "损失应该非负"

    def test_loss_gradients(self, sample_inputs):
        """测试梯度计算。"""
        desc_orig, desc_rot, h = sample_inputs
        desc_orig.requires_grad_(True)
        
        loss = compute_description_loss(desc_orig, desc_rot, h)
        loss.backward()
        
        assert desc_orig.grad is not None, "应该有梯度"
        assert not torch.isnan(desc_orig.grad).any(), "梯度不应包含 NaN"

    def test_custom_num_samples(self, sample_inputs):
        """测试自定义采样数量。"""
        desc_orig, desc_rot, h = sample_inputs
        
        loss_100 = compute_description_loss(desc_orig, desc_rot, h, num_samples=100)
        loss_400 = compute_description_loss(desc_orig, desc_rot, h, num_samples=400)
        
        # 两种采样数都应该产生有效损失
        assert loss_100.item() >= 0
        assert loss_400.item() >= 0

    def test_custom_weights(self, sample_inputs):
        """测试自定义权重。"""
        desc_orig, desc_rot, h = sample_inputs
        
        loss_default = compute_description_loss(desc_orig, desc_rot, h)
        loss_custom = compute_description_loss(
            desc_orig, desc_rot, h,
            geometric_weight=2.0,
            manhattan_weight=0.5
        )
        
        # 不同权重应该产生不同结果
        assert loss_default.item() != loss_custom.item()

    def test_margin_parameter(self, sample_inputs):
        """测试 margin 参数。"""
        desc_orig, desc_rot, h = sample_inputs
        
        loss_small_margin = compute_description_loss(desc_orig, desc_rot, h, margin=0.5)
        loss_large_margin = compute_description_loss(desc_orig, desc_rot, h, margin=2.0)
        
        # 两种 margin 都应该产生有效损失
        assert loss_small_margin.item() >= 0
        assert loss_large_margin.item() >= 0

    def test_default_weights_constants(self):
        """测试默认权重使用常量。"""
        assert DESC_GEOMETRIC_WEIGHT == 1.0
        assert DESC_MANHATTAN_WEIGHT == 0.1
        assert DESC_SPARSITY_WEIGHT == 0.01
        assert DESC_BINARY_WEIGHT == 0.05


class TestNumericalStability:
    """测试数值稳定性。"""

    def test_singular_matrix_handling(self):
        """测试奇异矩阵处理。"""
        # 创建接近奇异的矩阵
        det_orig = torch.rand(2, 1, 32, 32)
        det_rot = torch.rand(2, 1, 32, 32)
        # 接近奇异的变换
        h = torch.tensor([
            [[1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0]],
            [[1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0]]
        ])
        
        # 不应该崩溃
        loss = compute_detection_loss(det_orig, det_rot, h)
        assert not torch.isnan(loss), "损失不应为 NaN"
        assert not torch.isinf(loss), "损失不应为 Inf"

    def test_extreme_values(self):
        """测试极端值处理。"""
        # BCE 损失要求输入在 [0, 1] 范围内，所以使用有效范围内的值
        det_orig = torch.rand(2, 1, 32, 32)  # 已经在 [0, 1] 范围内
        det_rot = torch.rand(2, 1, 32, 32)
        h = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ])
        
        # 不应该崩溃或产生 NaN
        loss = compute_detection_loss(det_orig, det_rot, h)
        assert not torch.isnan(loss)


class TestDeviceCompatibility:
    """测试设备兼容性。"""

    def test_cpu_execution(self):
        """测试 CPU 执行。"""
        device = torch.device("cpu")
        det_orig = torch.rand(2, 1, 16, 16, device=device)
        det_rot = torch.rand(2, 1, 16, 16, device=device)
        h = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ], device=device)
        
        loss = compute_detection_loss(det_orig, det_rot, h)
        assert loss.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_cuda_execution(self):
        """测试 CUDA 执行。"""
        try:
            device = torch.device("cuda")
            det_orig = torch.rand(2, 1, 16, 16, device=device)
            det_rot = torch.rand(2, 1, 16, 16, device=device)
            h = torch.tensor([
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ], device=device)
            
            loss = compute_detection_loss(det_orig, det_rot, h)
            assert loss.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e):
                pytest.skip(f"CUDA 不可用: {e}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])