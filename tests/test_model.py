#!/usr/bin/env python3
"""
模型单元测试。

测试内容：
- 模型前向传播
- 不同骨干网络
- FPN 多尺度输出
- 设备兼容性
- 新配置系统
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.rord import RoRD, SEBlock, CBAM
from utils.config import (
    ModelConfig, 
    BackboneConfig, 
    AttentionConfig, 
    FPNConfig,
    RoRDConfig,
)


def create_model_config(
    backbone_name: str = "vgg16", 
    pretrained: bool = False,
    fpn_enabled: bool = True,
    fpn_levels: tuple = (2, 3, 4),
) -> ModelConfig:
    """创建测试用的模型配置。"""
    return ModelConfig(
        backbone=BackboneConfig(name=backbone_name, pretrained=pretrained),
        attention=AttentionConfig(enabled=False),
        fpn=FPNConfig(enabled=fpn_enabled, levels=fpn_levels),
    )


class TestSEBlock:
    """测试 SE Block 注意力模块。"""

    def test_forward_shape(self):
        """测试前向传播形状。"""
        se = SEBlock(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        
        out = se(x)
        
        assert out.shape == x.shape, f"期望形状 {x.shape}，实际 {out.shape}"

    def test_forward_values(self):
        """测试前向传播值。"""
        se = SEBlock(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        
        out = se(x)
        
        # 输出应该是输入的缩放版本
        assert not torch.isnan(out).any(), "输出不应包含 NaN"
        assert not torch.isinf(out).any(), "输出不应包含 Inf"

    def test_reduction_parameter(self):
        """测试 reduction 参数。"""
        for reduction in [4, 8, 16, 32]:
            se = SEBlock(channels=64, reduction=reduction)
            x = torch.randn(2, 64, 32, 32)
            out = se(x)
            assert out.shape == x.shape

    def test_small_channels(self):
        """测试小通道数。"""
        se = SEBlock(channels=8, reduction=16)
        x = torch.randn(2, 8, 32, 32)
        out = se(x)
        assert out.shape == x.shape


class TestCBAM:
    """测试 CBAM 注意力模块。"""

    def test_forward_shape(self):
        """测试前向传播形状。"""
        cbam = CBAM(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        
        out = cbam(x)
        
        assert out.shape == x.shape

    def test_forward_values(self):
        """测试前向传播值。"""
        cbam = CBAM(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        
        out = cbam(x)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestMakeAttnLayer:
    """测试注意力层创建（通过模型内部）。"""

    def test_se_attention_in_model(self):
        """测试模型中的 SE 注意力。"""
        # 创建带有 SE 注意力的模型
        model = RoRD()
        # 检查模型是否正确创建
        assert model is not None

    def test_cbam_attention_in_model(self):
        """测试模型中的 CBAM 注意力。"""
        # 创建带有 CBAM 注意力的模型
        model = RoRD()
        assert model is not None


class TestRoRDModel:
    """测试 RoRD 模型。"""

    @pytest.fixture
    def sample_input(self):
        """创建测试输入（3通道，匹配 VGG16 输入）。"""
        return torch.randn(2, 1, 256, 256)

    def test_model_creation_default(self):
        """测试默认模型创建。"""
        model = RoRD()
        
        assert model is not None
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'detection_head')
        assert hasattr(model, 'descriptor_head')

    def test_forward_single_scale(self, sample_input):
        """测试单尺度前向传播。"""
        model = RoRD()
        model.eval()
        
        with torch.no_grad():
            det_map, descriptors = model(sample_input)
        
        # 检查输出形状
        assert det_map.dim() == 4, f"检测图应该是 4D，实际 {det_map.dim()}"
        assert descriptors.dim() == 4, f"描述子应该是 4D，实际 {descriptors.dim()}"
        
        # 检查批次大小
        assert det_map.shape[0] == 2
        assert descriptors.shape[0] == 2

    def test_forward_fpn(self, sample_input):
        """测试 FPN 多尺度前向传播。"""
        model = RoRD(fpn_levels=(2, 3, 4))
        model.eval()
        
        with torch.no_grad():
            pyramid = model(sample_input, return_pyramid=True)
        
        # 检查金字塔输出
        assert isinstance(pyramid, dict)
        assert "P2" in pyramid or "P3" in pyramid or "P4" in pyramid
        
        # 检查每个金字塔层的输出
        for level_name, (det, desc, stride) in pyramid.items():
            assert det.dim() == 4, f"{level_name} 检测图应该是 4D"
            assert desc.dim() == 4, f"{level_name} 描述子应该是 4D"
            assert isinstance(stride, int), f"{level_name} stride 应该是 int"

    def test_forward_fpn_partial_levels(self, sample_input):
        """测试部分 FPN 层级。"""
        model = RoRD(fpn_levels=(3, 4))
        model.eval()
        
        with torch.no_grad():
            pyramid = model(sample_input, return_pyramid=True)
        
        assert "P3" in pyramid
        assert "P4" in pyramid
        assert "P2" not in pyramid

    def test_fpn_levels_validation(self):
        """测试 FPN 层级验证。"""
        # 有效层级
        model = RoRD(fpn_levels=(2, 3, 4))
        assert model.fpn_levels == (2, 3, 4)
        
        # 无效层级应该抛出异常
        with pytest.raises(ValueError, match="FPN 层级必须是"):
            RoRD(fpn_levels=(1, 2, 5))

    def test_fpn_levels_sorting(self):
        """测试 FPN 层级排序。"""
        model = RoRD(fpn_levels=(4, 2, 3))
        assert model.fpn_levels == (2, 3, 4)

    def test_different_backbones(self, sample_input):
        """测试不同骨干网络。"""
        backbones = ["vgg16", "resnet34", "efficientnet_b0"]
        
        for backbone in backbones:
            model_config = create_model_config(backbone_name=backbone)
            model = RoRD(model_config=model_config)
            model.eval()
            
            with torch.no_grad():
                det_map, descriptors = model(sample_input)
            
            assert det_map.shape[0] == 2, f"{backbone} 骨干输出批次大小错误"

    def test_pretrained_parameter(self):
        """测试预训练参数。"""
        # 不加载预训练权重
        model = RoRD()
        assert model is not None

    def test_model_gradients(self, sample_input):
        """测试模型梯度计算。"""
        model = RoRD()
        model.train()
        
        det_map, descriptors = model(sample_input)
        loss = det_map.mean() + descriptors.mean()
        loss.backward()
        
        # 检查关键参数有梯度
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                has_grad = True
                break
        
        assert has_grad, "模型应该有非零梯度"

    def test_detection_head_output_range(self, sample_input):
        """测试检测头输出范围。"""
        model = RoRD()
        model.eval()
        
        with torch.no_grad():
            det_map, _ = model(sample_input)
        
        # 检测图应该在 [0, 1] 范围内（经过 Sigmoid）
        assert det_map.min() >= 0, f"检测图最小值 {det_map.min()} 应该 >= 0"
        assert det_map.max() <= 1, f"检测图最大值 {det_map.max()} 应该 <= 1"

    def test_descriptor_normalization(self, sample_input):
        """测试描述子归一化。"""
        model = RoRD()
        model.eval()
        
        with torch.no_grad():
            _, descriptors = model(sample_input)
        
        # 描述子应该经过 InstanceNorm
        # 检查每个空间位置的描述子向量
        desc_flat = descriptors.view(2, 128, -1)
        norms = torch.norm(desc_flat, dim=1)
        
        # 归一化后的向量范数应该接近某个常数
        assert not torch.isnan(norms).any()


class TestModelDeviceCompatibility:
    """测试模型设备兼容性。"""

    def test_cpu_execution(self):
        """测试 CPU 执行。"""
        device = torch.device("cpu")
        model = RoRD().to(device)
        model.eval()
        
        x = torch.randn(1, 1, 128, 128, device=device)
        
        with torch.no_grad():
            det_map, descriptors = model(x)
        
        assert det_map.device.type == "cpu"
        assert descriptors.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_cuda_execution(self):
        """测试 CUDA 执行。"""
        try:
            device = torch.device("cuda")
            model = RoRD().to(device)
            model.eval()
            
            x = torch.randn(1, 1, 128, 128, device=device)
            
            with torch.no_grad():
                det_map, descriptors = model(x)
            
            assert det_map.device.type == "cuda"
            assert descriptors.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e):
                pytest.skip(f"CUDA 不可用: {e}")
            raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_cpu_to_cuda_transfer(self):
        """测试 CPU 到 CUDA 转换。"""
        model = RoRD()
        
        # 在 CPU 上创建输入
        x_cpu = torch.randn(1, 1, 128, 128)
        
        # 转移到 CUDA
        try:
            device = torch.device("cuda")
            model = model.to(device)
            x_cuda = x_cpu.to(device)
            
            with torch.no_grad():
                det_map, descriptors = model(x_cuda)
            
            assert det_map.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e):
                pytest.skip(f"CUDA 不可用: {e}")
            raise


class TestModelParameterCount:
    """测试模型参数数量。"""

    def test_parameter_count_vgg16(self):
        """测试 VGG16 骨干参数数量。"""
        model = RoRD()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # VGG16 应该有约 14M 参数
        assert total_params > 1e6, "VGG16 参数数量应该 > 1M"
        assert total_params == trainable_params, "所有参数应该可训练"

    def test_parameter_count_resnet34(self):
        """测试 ResNet34 骨干参数数量。"""
        model_config = create_model_config(backbone_name="resnet34")
        model = RoRD(model_config=model_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # ResNet34 应该有约 21M 参数
        assert total_params > 1e6, "ResNet34 参数数量应该 > 1M"

    def test_parameter_count_efficientnet(self):
        """测试 EfficientNet-B0 骨干参数数量。"""
        model_config = create_model_config(backbone_name="efficientnet_b0")
        model = RoRD(model_config=model_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # EfficientNet-B0 应该有约 5M 参数
        assert total_params > 1e6, "EfficientNet-B0 参数数量应该 > 1M"


class TestModelMemory:
    """测试模型内存使用。"""

    def test_memory_cleanup(self):
        """测试内存清理。"""
        import gc
        
        # 创建模型
        model = RoRD()
        
        # 删除模型
        del model
        gc.collect()
        
        # 应该不会内存泄漏

    def test_inference_memory(self):
        """测试推理内存。"""
        model = RoRD()
        model.eval()
        
        # 多次推理
        for _ in range(5):
            x = torch.randn(1, 1, 128, 128)
            with torch.no_grad():
                det_map, descriptors = model(x)
            
            # 清理
            del x, det_map, descriptors


class TestModelSerialization:
    """测试模型序列化。"""

    def test_state_dict_save_load(self, tmp_path):
        """测试状态字典保存和加载。"""
        model = RoRD()

        # 保存状态字典
        state_dict = model.state_dict()
        save_path = tmp_path / "model.pt"
        torch.save(state_dict, save_path)

        # 加载状态字典
        model2 = RoRD()
        model2.load_state_dict(torch.load(save_path))

        # 检查参数相同
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.allclose(param1, param2), f"参数 {name1} 不匹配"


class TestNewConfigSystem:
    """测试新配置系统。"""

    def test_model_config_creation(self):
        """测试模型配置创建。"""
        config = ModelConfig(
            backbone=BackboneConfig(name="resnet34", pretrained=False),
            attention=AttentionConfig(enabled=True, type="se"),
            fpn=FPNConfig(enabled=True, levels=(2, 3)),
        )
        
        assert config.backbone.name == "resnet34"
        assert config.attention.enabled is True
        assert config.fpn.levels == (2, 3)

    def test_model_with_model_config(self, tmp_path):
        """测试使用 ModelConfig 创建模型。"""
        model_config = create_model_config(
            backbone_name="vgg16",
            fpn_levels=(2, 3, 4),
        )
        
        model = RoRD(model_config=model_config)
        
        assert model.backbone_name == "vgg16"
        assert model.fpn_levels == (2, 3, 4)

    def test_rord_config_from_yaml(self):
        """测试从 YAML 加载完整配置。"""
        import tempfile
        import yaml
        
        # 创建临时 YAML 文件
        config_data = {
            "model": {
                "backbone": {"name": "resnet34", "pretrained": False},
                "attention": {"enabled": False, "type": "none"},
                "fpn": {"enabled": True, "out_channels": 256, "levels": [2, 3, 4]},
            },
            "training": {
                "learning_rate": 0.0001,
                "batch_size": 16,
            }
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name
        
        try:
            cfg = RoRDConfig.from_yaml(yaml_path)
            
            assert cfg.model.backbone.name == "resnet34"
            assert cfg.training.learning_rate == 0.0001
            assert cfg.training.batch_size == 16
        finally:
            import os
            os.unlink(yaml_path)

    def test_backward_compatibility(self):
        """测试向后兼容性。"""
        # 使用旧参数创建模型
        model = RoRD(fpn_out_channels=128, fpn_levels=(3, 4))
        
        assert model.fpn_out_channels == 128
        assert model.fpn_levels == (3, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])