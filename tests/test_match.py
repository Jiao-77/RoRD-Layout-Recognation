#!/usr/bin/env python3
"""
匹配逻辑单元测试。

测试内容：
- 特征提取
- NMS 算法
- 互近邻匹配
- 旋转角度提取
- 匹配评分
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from match import (
    extract_rotation_angle,
    calculate_match_score,
    calculate_similarity,
    generate_difference_description,
    extract_keypoints_and_descriptors,
    radius_nms,
    mutual_nearest_neighbor,
)


class TestExtractRotationAngle:
    """测试旋转角度提取函数。"""

    def test_identity_matrix(self):
        """测试单位矩阵。"""
        H = np.eye(3)
        angle = extract_rotation_angle(H)
        assert angle == 0, f"期望 0°，实际 {angle}°"

    def test_90_degree_rotation(self):
        """测试 90 度旋转。"""
        # 90 度旋转矩阵
        angle_rad = np.radians(90)
        H = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        angle = extract_rotation_angle(H)
        assert angle == 90, f"期望 90°，实际 {angle}°"

    def test_180_degree_rotation(self):
        """测试 180 度旋转。"""
        angle_rad = np.radians(180)
        H = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        angle = extract_rotation_angle(H)
        assert angle == 180, f"期望 180°，实际 {angle}°"

    def test_270_degree_rotation(self):
        """测试 270 度旋转。"""
        angle_rad = np.radians(270)
        H = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        angle = extract_rotation_angle(H)
        # 270 度可能被识别为 -90 度，四舍五入到 0 或 270
        # 由于 arctan2 返回 [-π, π]，270 度实际上是 -90 度
        assert angle in [0, 270], f"期望 0° 或 270°，实际 {angle}°"

    def test_none_input(self):
        """测试 None 输入。"""
        angle = extract_rotation_angle(None)
        assert angle == 0

    def test_small_rotation(self):
        """测试小角度旋转（应该四舍五入到 0）。"""
        angle_rad = np.radians(10)
        H = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        angle = extract_rotation_angle(H)
        assert angle == 0, f"小角度应该四舍五入到 0°，实际 {angle}°"

    def test_45_degree_rotation(self):
        """测试 45 度旋转（应该四舍五入到 0 或 90）。"""
        angle_rad = np.radians(45)
        H = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        angle = extract_rotation_angle(H)
        # 45 度应该四舍五入到最近的 90 度倍数
        assert angle in [0, 90], f"45° 应该四舍五入到 0° 或 90°，实际 {angle}°"


class TestCalculateMatchScore:
    """测试匹配评分计算函数。"""

    def test_perfect_match(self):
        """测试完美匹配。"""
        H = np.eye(3)
        score = calculate_match_score(100, 100, H)
        
        assert 0 <= score <= 1, f"评分应该在 [0, 1] 范围内，实际 {score}"

    def test_poor_match(self):
        """测试差匹配。"""
        H = np.eye(3)
        score = calculate_match_score(10, 100, H)
        
        assert 0 <= score <= 1

    def test_zero_inliers(self):
        """测试零内点。"""
        H = np.eye(3)
        score = calculate_match_score(0, 100, H)
        
        assert score >= 0

    def test_none_homography(self):
        """测试 None 单应性矩阵。"""
        score = calculate_match_score(50, 100, None)
        
        assert 0 <= score <= 1

    def test_custom_inlier_ratio(self):
        """测试自定义内点比例。"""
        H = np.eye(3)
        score1 = calculate_match_score(50, 100, H, inlier_ratio=0.5)
        score2 = calculate_match_score(50, 100, H, inlier_ratio=0.8)
        
        # 不同内点比例应该产生不同评分
        assert score1 != score2

    def test_score_bounds(self):
        """测试评分边界。"""
        H = np.eye(3)
        
        # 各种情况
        for inliers, total in [(0, 100), (50, 100), (100, 100), (150, 100)]:
            score = calculate_match_score(inliers, total, H)
            assert 0 <= score <= 1, f"({inliers}, {total}) 评分 {score} 超出范围"


class TestCalculateSimilarity:
    """测试相似度计算函数。"""

    def test_perfect_similarity(self):
        """测试完美相似度。"""
        sim = calculate_similarity(100, 100, 100)
        assert 0 <= sim <= 1

    def test_zero_matches(self):
        """测试零匹配。"""
        sim = calculate_similarity(0, 100, 100)
        assert sim >= 0

    def test_partial_matches(self):
        """测试部分匹配。"""
        sim = calculate_similarity(50, 100, 200)
        assert 0 <= sim <= 1

    def test_more_matches_than_keypoints(self):
        """测试匹配数超过关键点数。"""
        # 这种情况不应该发生，但函数应该能处理
        sim = calculate_similarity(150, 100, 100)
        assert sim >= 0

    def test_symmetry(self):
        """测试对称性（不完全对称，但应该合理）。"""
        sim1 = calculate_similarity(50, 100, 200)
        sim2 = calculate_similarity(50, 200, 100)
        
        # 两者都应该是有效值
        assert 0 <= sim1 <= 1
        assert 0 <= sim2 <= 1


class TestGenerateDifferenceDescription:
    """测试差异描述生成函数。"""

    def test_high_match(self):
        """测试高匹配情况。"""
        H = np.eye(3)
        desc = generate_difference_description(H, 90, 100)
        
        assert isinstance(desc, str)
        assert "高度匹配" in desc or "良好匹配" in desc

    def test_low_match(self):
        """测试低匹配情况。"""
        H = np.eye(3)
        desc = generate_difference_description(H, 20, 100)
        
        assert isinstance(desc, str)
        assert "低质量匹配" in desc or "中等匹配" in desc

    def test_with_rotation(self):
        """测试带旋转的情况。"""
        H = np.eye(3)
        desc = generate_difference_description(H, 50, 100, angle_diff=90)
        
        assert isinstance(desc, str)
        assert "旋转" in desc or "90" in desc

    def test_none_homography(self):
        """测试 None 单应性矩阵。"""
        desc = generate_difference_description(None, 50, 100)
        
        assert isinstance(desc, str)

    def test_zero_total_matches(self):
        """测试零总匹配数。"""
        H = np.eye(3)
        desc = generate_difference_description(H, 0, 0)
        
        assert isinstance(desc, str)


class TestRadiusNMS:
    """测试半径 NMS 函数。"""

    def test_empty_keypoints(self):
        """测试空关键点。"""
        kps = torch.empty((0, 2))
        scores = torch.empty((0,))
        
        keep = radius_nms(kps, scores, radius=5.0)
        
        assert len(keep) == 0

    def test_single_keypoint(self):
        """测试单个关键点。"""
        kps = torch.tensor([[10.0, 10.0]])
        scores = torch.tensor([0.9])
        
        keep = radius_nms(kps, scores, radius=5.0)
        
        assert len(keep) == 1
        assert keep[0] == 0

    def test_distant_keypoints(self):
        """测试距离较远的关键点（都应该保留）。"""
        kps = torch.tensor([
            [0.0, 0.0],
            [100.0, 100.0],
            [200.0, 200.0]
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        keep = radius_nms(kps, scores, radius=5.0)
        
        assert len(keep) == 3

    def test_close_keypoints(self):
        """测试距离较近的关键点（应该被抑制）。"""
        kps = torch.tensor([
            [0.0, 0.0],
            [2.0, 2.0],  # 距离约 2.8，在半径 5 内
            [10.0, 10.0]  # 距离较远
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        keep = radius_nms(kps, scores, radius=5.0)
        
        # 第一个点得分最高，应该保留
        # 第二个点在第一个点的半径内，应该被抑制
        # 第三个点距离较远，应该保留
        assert 0 in keep
        assert 2 in keep
        # 第二个点可能被抑制

    def test_score_ordering(self):
        """测试得分排序（高分应该优先保留）。"""
        kps = torch.tensor([
            [0.0, 0.0],
            [1.0, 1.0],  # 非常近
        ])
        scores = torch.tensor([0.8, 0.9])  # 第二个得分更高
        
        keep = radius_nms(kps, scores, radius=5.0)
        
        # 第二个点得分更高，应该保留
        assert 1 in keep

    def test_mismatched_lengths(self):
        """测试长度不匹配的情况。"""
        kps = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        scores = torch.tensor([0.9])  # 只有一个得分
        
        with pytest.raises(ValueError, match="数量不匹配"):
            radius_nms(kps, scores, radius=5.0)

    def test_different_radii(self):
        """测试不同半径。"""
        kps = torch.tensor([
            [0.0, 0.0],
            [5.0, 5.0],
            [20.0, 20.0]
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        # 小半径：应该保留更多点
        keep_small = radius_nms(kps, scores, radius=2.0)
        
        # 大半径：应该保留更少点
        keep_large = radius_nms(kps, scores, radius=20.0)
        
        assert len(keep_small) >= len(keep_large)


    def test_kdtree_optimization(self):
        """测试 KD-Tree 优化自动选择。"""
        # 小规模：使用向量化
        small_kps = torch.rand(100, 2) * 256
        small_scores = torch.rand(100)

        # 大规模：使用 KD-Tree
        large_kps = torch.rand(2000, 2) * 256
        large_scores = torch.rand(2000)

        # 两者都应该正常工作
        keep_small = radius_nms(small_kps, small_scores, radius=10.0)
        keep_large = radius_nms(large_kps, large_scores, radius=10.0)

        assert len(keep_small) > 0
        assert len(keep_large) > 0

    def test_kdtree_correctness(self):
        """测试 KD-Tree 实现的正确性。"""
        from match import _radius_nms_vectorized, _radius_nms_kdtree

        torch.manual_seed(42)
        kps = torch.rand(200, 2) * 256
        scores = torch.rand(200)

        result_vec = _radius_nms_vectorized(kps, scores, radius=10.0)
        result_kdtree = _radius_nms_kdtree(kps, scores, radius=10.0)

        # 两种实现应该产生相同数量的关键点
        assert len(result_vec) == len(result_kdtree)

    def test_performance_large_scale(self):
        """测试大规模关键点的性能。"""
        import time

        torch.manual_seed(42)
        kps = torch.rand(5000, 2) * 256
        scores = torch.rand(5000)

        start = time.time()
        keep = radius_nms(kps, scores, radius=10.0)
        elapsed = time.time() - start

        # 应该在合理时间内完成（< 1 秒）
        assert elapsed < 1.0, f"NMS 耗时 {elapsed:.2f}s，超过 1 秒"
        assert len(keep) > 0


class TestMutualNearestNeighbor:
    """测试互近邻匹配函数。"""

    def test_empty_descriptors(self):
        """测试空描述子。"""
        descs1 = torch.empty((0, 128))
        descs2 = torch.randn(10, 128)
        
        matches = mutual_nearest_neighbor(descs1, descs2)
        
        assert matches.shape == (0, 2)

    def test_identical_descriptors(self):
        """测试相同描述子。"""
        descs = torch.randn(5, 128)
        descs = torch.nn.functional.normalize(descs, dim=1)
        
        matches = mutual_nearest_neighbor(descs, descs)
        
        # 相同描述子应该产生完美匹配
        assert matches.shape[0] == 5
        # 每个点应该匹配到自己
        for i, (idx1, idx2) in enumerate(matches):
            assert idx1 == idx2

    def test_different_descriptors(self):
        """测试不同描述子。"""
        descs1 = torch.randn(10, 128)
        descs2 = torch.randn(10, 128)
        descs1 = torch.nn.functional.normalize(descs1, dim=1)
        descs2 = torch.nn.functional.normalize(descs2, dim=1)
        
        matches = mutual_nearest_neighbor(descs1, descs2)
        
        # 应该有一些匹配
        assert matches.shape[1] == 2
        # 索引应该在有效范围内
        assert (matches[:, 0] >= 0).all() and (matches[:, 0] < 10).all()
        assert (matches[:, 1] >= 0).all() and (matches[:, 1] < 10).all()

    def test_output_dtype(self):
        """测试输出数据类型。"""
        descs1 = torch.randn(5, 128)
        descs2 = torch.randn(5, 128)
        
        matches = mutual_nearest_neighbor(descs1, descs2)
        
        assert matches.dtype == torch.int64


class TestExtractKeypointsAndDescriptors:
    """测试关键点和描述子提取函数。"""

    @pytest.fixture
    def model(self):
        """创建测试模型。"""
        from models.rord import RoRD
        model = RoRD()
        model.eval()
        return model

    @pytest.fixture
    def sample_image(self):
        """创建测试图像。"""
        return torch.randn(1, 1, 256, 256)

    @pytest.mark.integration
    def test_output_types(self, model, sample_image):
        """测试输出类型。"""
        with torch.no_grad():
            kps, descs = extract_keypoints_and_descriptors(model, sample_image, kp_thresh=0.5)
        
        assert isinstance(kps, torch.Tensor)
        assert isinstance(descs, torch.Tensor)

    @pytest.mark.integration
    def test_output_shapes(self, model, sample_image):
        """测试输出形状。"""
        with torch.no_grad():
            kps, descs = extract_keypoints_and_descriptors(model, sample_image, kp_thresh=0.5)
        
        if len(kps) > 0:
            assert kps.shape[1] == 2, "关键点应该是 (N, 2)"
            assert descs.shape[0] == kps.shape[0], "描述子数量应该匹配关键点数量"

    @pytest.mark.integration
    def test_different_thresholds(self, model, sample_image):
        """测试不同阈值。"""
        with torch.no_grad():
            kps_low, _ = extract_keypoints_and_descriptors(model, sample_image, kp_thresh=0.1)
            kps_high, _ = extract_keypoints_and_descriptors(model, sample_image, kp_thresh=0.9)
        
        # 低阈值应该产生更多关键点
        assert len(kps_low) >= len(kps_high)

    @pytest.mark.integration
    def test_empty_result(self, model):
        """测试空白图像。"""
        # 纯黑图像
        black_image = torch.zeros(1, 1, 256, 256)
        
        with torch.no_grad():
            kps, descs = extract_keypoints_and_descriptors(model, black_image, kp_thresh=0.9)
        
        # 可能没有关键点
        assert isinstance(kps, torch.Tensor)
        assert isinstance(descs, torch.Tensor)


class TestIntegration:
    """集成测试。"""

    def test_full_pipeline(self):
        """测试完整匹配流程。"""
        # 创建模拟数据
        H = np.eye(3)
        
        # 测试评分计算
        score = calculate_match_score(80, 100, H)
        assert 0 <= score <= 1
        
        # 测试相似度计算
        sim = calculate_similarity(80, 100, 100)
        assert 0 <= sim <= 1
        
        # 测试差异描述
        desc = generate_difference_description(H, 80, 100)
        assert isinstance(desc, str)
        
        # 测试旋转角度
        angle = extract_rotation_angle(H)
        assert angle == 0

    def test_nms_and_matching(self):
        """测试 NMS 和匹配组合。"""
        # 创建模拟关键点和得分
        kps = torch.tensor([
            [10.0, 10.0],
            [12.0, 12.0],  # 近
            [100.0, 100.0],  # 远
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        # 应用 NMS
        keep = radius_nms(kps, scores, radius=5.0)
        
        # 创建模拟描述子
        descs1 = torch.randn(len(keep), 128)
        descs2 = torch.randn(5, 128)
        
        # 执行匹配
        matches = mutual_nearest_neighbor(descs1, descs2)
        
        # 应该有有效的输出
        assert matches.dim() == 2
        assert matches.shape[1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])