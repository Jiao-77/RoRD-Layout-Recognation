"""
统一配置系统。

使用 dataclass 定义配置结构，支持：
- 从 YAML 文件加载
- 类型安全的配置访问
- 默认值和验证

废弃:
    config.py: 请使用此模块代替
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ============================================================================
# 模型配置
# ============================================================================

@dataclass
class BackboneConfig:
    """骨干网络配置。"""
    name: Literal["vgg16", "resnet34", "efficientnet_b0"] = "vgg16"
    pretrained: bool = False


@dataclass
class AttentionConfig:
    """注意力机制配置。"""
    enabled: bool = False
    type: Literal["none", "se", "cbam"] = "none"
    places: List[str] = field(default_factory=list)
    reduction: int = 16
    spatial_kernel: int = 7


@dataclass
class FPNConfig:
    """FPN 配置。"""
    enabled: bool = True
    out_channels: int = 256
    levels: Tuple[int, ...] = (2, 3, 4)
    norm: Literal["bn", "ln", "none"] = "bn"


@dataclass
class ModelConfig:
    """模型配置。"""
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)


# ============================================================================
# 训练配置
# ============================================================================

@dataclass
class TrainingConfig:
    """训练配置。"""
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 50
    patch_size: int = 256
    scale_jitter_range: Tuple[float, float] = (0.8, 1.2)


# ============================================================================
# 匹配配置
# ============================================================================

@dataclass
class NMSConfig:
    """NMS 配置。"""
    enabled: bool = True
    radius: int = 4
    score_threshold: float = 0.5


@dataclass
class MatchingConfig:
    """匹配配置。"""
    keypoint_threshold: float = 0.5
    ransac_reproj_threshold: float = 5.0
    min_inliers: int = 15
    pyramid_scales: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.5])
    inference_window_size: int = 1024
    inference_stride: int = 768
    use_fpn: bool = True
    nms: NMSConfig = field(default_factory=NMSConfig)


# ============================================================================
# 评估配置
# ============================================================================

@dataclass
class EvaluationConfig:
    """评估配置。"""
    iou_threshold: float = 0.5


# ============================================================================
# 日志配置
# ============================================================================

@dataclass
class LoggingConfig:
    """日志配置。"""
    use_tensorboard: bool = True
    log_dir: str = "runs"
    experiment_name: str = "baseline"


# ============================================================================
# 路径配置
# ============================================================================

@dataclass
class PathsConfig:
    """路径配置。"""
    layout_dir: str = "data/layouts"
    save_dir: str = "outputs"
    val_img_dir: str = "data/val/images"
    val_ann_dir: str = "data/val/annotations"
    template_dir: str = "data/templates"
    model_path: str = "outputs/model_final.pth"


# ============================================================================
# 数据增强配置
# ============================================================================

@dataclass
class ElasticAugmentConfig:
    """弹性变换增强配置。"""
    enabled: bool = False
    alpha: int = 40
    sigma: int = 6
    alpha_affine: int = 6
    prob: float = 0.3


@dataclass
class PhotometricAugmentConfig:
    """光度变换增强配置。"""
    brightness_contrast: bool = True
    gauss_noise: bool = True


@dataclass
class AugmentConfig:
    """数据增强配置。"""
    elastic: ElasticAugmentConfig = field(default_factory=ElasticAugmentConfig)
    photometric: PhotometricAugmentConfig = field(default_factory=PhotometricAugmentConfig)


# ============================================================================
# 数据源配置
# ============================================================================

@dataclass
class DataSourceConfig:
    """数据源配置。"""
    enabled: bool = True
    ratio: float = 1.0
    png_dir: str = ""


@dataclass
class DiffusionTrainingConfig:
    """扩散模型训练配置。"""
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    image_size: int = 256
    timesteps: int = 1000
    augment: bool = True


@dataclass
class DiffusionGenerationConfig:
    """扩散生成配置。"""
    num_samples: int = 200
    timesteps: int = 1000


@dataclass
class DiffusionSourceConfig:
    """扩散数据源配置。"""
    enabled: bool = False
    model_dir: str = "models/diffusion"
    png_dir: str = "data/diffusion_generated"
    ratio: float = 0.0
    training: DiffusionTrainingConfig = field(default_factory=DiffusionTrainingConfig)
    generation: DiffusionGenerationConfig = field(default_factory=DiffusionGenerationConfig)


@dataclass
class DataSourcesConfig:
    """数据源配置。"""
    real: DataSourceConfig = field(default_factory=DataSourceConfig)
    diffusion: DiffusionSourceConfig = field(default_factory=DiffusionSourceConfig)


# ============================================================================
# 主配置
# ============================================================================

@dataclass
class RoRDConfig:
    """
    RoRD 项目主配置。
    
    使用示例:
        # 从 YAML 加载
        cfg = RoRDConfig.from_yaml("configs/base_config.yaml")
        
        # 访问配置
        print(cfg.model.backbone.name)
        print(cfg.training.learning_rate)
        
        # 创建模型
        model = RoRD(cfg.model)
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RoRDConfig":
        """
        从 YAML 文件加载配置。
        
        Args:
            path: YAML 配置文件路径
            
        Returns:
            RoRDConfig 实例
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        
        dict_cfg = OmegaConf.load(path)
        return cls.from_dictconfig(dict_cfg, path.parent)

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig, base_dir: Optional[Path] = None) -> "RoRDConfig":
        """
        从 DictConfig 创建配置。
        
        Args:
            cfg: OmegaConf DictConfig 对象
            base_dir: 配置文件所在目录（用于解析相对路径）
            
        Returns:
            RoRDConfig 实例
        """
        def get_nested(cfg_obj: Any, *keys: str, default: Any = None) -> Any:
            """安全获取嵌套配置值。"""
            result = cfg_obj
            for key in keys:
                if result is None:
                    return default
                if isinstance(result, dict):
                    result = result.get(key, default)
                elif hasattr(result, key):
                    result = getattr(result, key, default)
                else:
                    return default
            return result if result is not None else default

        # 解析模型配置
        backbone_cfg = BackboneConfig(
            name=get_nested(cfg, "model", "backbone", "name", default="vgg16"),
            pretrained=get_nested(cfg, "model", "backbone", "pretrained", default=False),
        )
        
        attention_cfg = AttentionConfig(
            enabled=get_nested(cfg, "model", "attention", "enabled", default=False),
            type=get_nested(cfg, "model", "attention", "type", default="none"),
            places=list(get_nested(cfg, "model", "attention", "places", default=[])),
            reduction=get_nested(cfg, "model", "attention", "reduction", default=16),
            spatial_kernel=get_nested(cfg, "model", "attention", "spatial_kernel", default=7),
        )
        
        fpn_levels = get_nested(cfg, "model", "fpn", "levels", default=[2, 3, 4])
        fpn_cfg = FPNConfig(
            enabled=get_nested(cfg, "model", "fpn", "enabled", default=True),
            out_channels=get_nested(cfg, "model", "fpn", "out_channels", default=256),
            levels=tuple(fpn_levels) if fpn_levels else (2, 3, 4),
            norm=get_nested(cfg, "model", "fpn", "norm", default="bn"),
        )
        
        model_cfg = ModelConfig(
            backbone=backbone_cfg,
            attention=attention_cfg,
            fpn=fpn_cfg,
        )

        # 解析训练配置
        scale_range = get_nested(cfg, "training", "scale_jitter_range", default=[0.8, 1.2])
        training_cfg = TrainingConfig(
            learning_rate=float(get_nested(cfg, "training", "learning_rate", default=5e-5)),
            batch_size=int(get_nested(cfg, "training", "batch_size", default=8)),
            num_epochs=int(get_nested(cfg, "training", "num_epochs", default=50)),
            patch_size=int(get_nested(cfg, "training", "patch_size", default=256)),
            scale_jitter_range=tuple(scale_range) if scale_range else (0.8, 1.2),
        )

        # 解析匹配配置
        nms_cfg = NMSConfig(
            enabled=get_nested(cfg, "matching", "nms", "enabled", default=True),
            radius=get_nested(cfg, "matching", "nms", "radius", default=4),
            score_threshold=get_nested(cfg, "matching", "nms", "score_threshold", default=0.5),
        )
        
        matching_cfg = MatchingConfig(
            keypoint_threshold=float(get_nested(cfg, "matching", "keypoint_threshold", default=0.5)),
            ransac_reproj_threshold=float(get_nested(cfg, "matching", "ransac_reproj_threshold", default=5.0)),
            min_inliers=int(get_nested(cfg, "matching", "min_inliers", default=15)),
            pyramid_scales=list(get_nested(cfg, "matching", "pyramid_scales", default=[0.75, 1.0, 1.5])),
            inference_window_size=int(get_nested(cfg, "matching", "inference_window_size", default=1024)),
            inference_stride=int(get_nested(cfg, "matching", "inference_stride", default=768)),
            use_fpn=get_nested(cfg, "matching", "use_fpn", default=True),
            nms=nms_cfg,
        )

        # 解析评估配置
        evaluation_cfg = EvaluationConfig(
            iou_threshold=float(get_nested(cfg, "evaluation", "iou_threshold", default=0.5)),
        )

        # 解析日志配置
        logging_cfg = LoggingConfig(
            use_tensorboard=get_nested(cfg, "logging", "use_tensorboard", default=True),
            log_dir=get_nested(cfg, "logging", "log_dir", default="runs"),
            experiment_name=get_nested(cfg, "logging", "experiment_name", default="baseline"),
        )

        # 解析路径配置
        def resolve_path(path_val: str) -> str:
            """解析相对路径。"""
            if not path_val or not base_dir:
                return path_val
            p = Path(path_val)
            if p.is_absolute():
                return str(p)
            return str((base_dir / p).resolve())

        paths_cfg = PathsConfig(
            layout_dir=resolve_path(get_nested(cfg, "paths", "layout_dir", default="data/layouts")),
            save_dir=resolve_path(get_nested(cfg, "paths", "save_dir", default="outputs")),
            val_img_dir=resolve_path(get_nested(cfg, "paths", "val_img_dir", default="data/val/images")),
            val_ann_dir=resolve_path(get_nested(cfg, "paths", "val_ann_dir", default="data/val/annotations")),
            template_dir=resolve_path(get_nested(cfg, "paths", "template_dir", default="data/templates")),
            model_path=resolve_path(get_nested(cfg, "paths", "model_path", default="outputs/model_final.pth")),
        )

        # 解析增强配置
        elastic_cfg = ElasticAugmentConfig(
            enabled=get_nested(cfg, "augment", "elastic", "enabled", default=False),
            alpha=get_nested(cfg, "augment", "elastic", "alpha", default=40),
            sigma=get_nested(cfg, "augment", "elastic", "sigma", default=6),
            alpha_affine=get_nested(cfg, "augment", "elastic", "alpha_affine", default=6),
            prob=get_nested(cfg, "augment", "elastic", "prob", default=0.3),
        )
        
        photometric_cfg = PhotometricAugmentConfig(
            brightness_contrast=get_nested(cfg, "augment", "photometric", "brightness_contrast", default=True),
            gauss_noise=get_nested(cfg, "augment", "photometric", "gauss_noise", default=True),
        )
        
        augment_cfg = AugmentConfig(
            elastic=elastic_cfg,
            photometric=photometric_cfg,
        )

        # 解析数据源配置
        real_cfg = DataSourceConfig(
            enabled=get_nested(cfg, "data_sources", "real", "enabled", default=True),
            ratio=get_nested(cfg, "data_sources", "real", "ratio", default=1.0),
            png_dir=get_nested(cfg, "data_sources", "real", "png_dir", default=""),
        )
        
        diffusion_training_cfg = DiffusionTrainingConfig(
            epochs=get_nested(cfg, "data_sources", "diffusion", "training", "epochs", default=100),
            batch_size=get_nested(cfg, "data_sources", "diffusion", "training", "batch_size", default=8),
            lr=get_nested(cfg, "data_sources", "diffusion", "training", "lr", default=1e-4),
            image_size=get_nested(cfg, "data_sources", "diffusion", "training", "image_size", default=256),
            timesteps=get_nested(cfg, "data_sources", "diffusion", "training", "timesteps", default=1000),
            augment=get_nested(cfg, "data_sources", "diffusion", "training", "augment", default=True),
        )
        
        diffusion_generation_cfg = DiffusionGenerationConfig(
            num_samples=get_nested(cfg, "data_sources", "diffusion", "generation", "num_samples", default=200),
            timesteps=get_nested(cfg, "data_sources", "diffusion", "generation", "timesteps", default=1000),
        )
        
        diffusion_cfg = DiffusionSourceConfig(
            enabled=get_nested(cfg, "data_sources", "diffusion", "enabled", default=False),
            model_dir=get_nested(cfg, "data_sources", "diffusion", "model_dir", default="models/diffusion"),
            png_dir=get_nested(cfg, "data_sources", "diffusion", "png_dir", default="data/diffusion_generated"),
            ratio=get_nested(cfg, "data_sources", "diffusion", "ratio", default=0.0),
            training=diffusion_training_cfg,
            generation=diffusion_generation_cfg,
        )
        
        data_sources_cfg = DataSourcesConfig(
            real=real_cfg,
            diffusion=diffusion_cfg,
        )

        return cls(
            model=model_cfg,
            training=training_cfg,
            matching=matching_cfg,
            evaluation=evaluation_cfg,
            logging=logging_cfg,
            paths=paths_cfg,
            augment=augment_cfg,
            data_sources=data_sources_cfg,
        )

    def to_dictconfig(self) -> DictConfig:
        """
        转换为 DictConfig 对象。
        
        Returns:
            DictConfig 对象
        """
        import dataclasses
        
        def dataclass_to_dict(obj: Any) -> Any:
            """递归转换 dataclass 为字典。"""
            if dataclasses.is_dataclass(obj):
                return {k: dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            return obj
        
        return OmegaConf.create(dataclass_to_dict(self))


# ============================================================================
# 便捷函数
# ============================================================================

def load_config(path: Union[str, Path]) -> RoRDConfig:
    """
    加载配置文件的便捷函数。
    
    Args:
        path: YAML 配置文件路径
        
    Returns:
        RoRDConfig 实例
    """
    return RoRDConfig.from_yaml(path)


def to_absolute_path(path_str: str, base_dir: Union[str, Path]) -> Path:
    """
    解析相对路径为绝对路径。
    
    Args:
        path_str: 路径字符串
        base_dir: 基准目录
        
    Returns:
        解析后的绝对路径
    """
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir) / path).resolve()