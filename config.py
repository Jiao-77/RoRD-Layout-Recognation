"""Legacy config shim loading values from YAML."""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


_BASE_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "base_config.yaml"
_CFG = OmegaConf.load(_BASE_CONFIG_PATH)

# --- 训练参数 ---
LEARNING_RATE = float(_CFG.training.learning_rate)
BATCH_SIZE = int(_CFG.training.batch_size)
NUM_EPOCHS = int(_CFG.training.num_epochs)
PATCH_SIZE = int(_CFG.training.patch_size)
SCALE_JITTER_RANGE = tuple(float(x) for x in _CFG.training.scale_jitter_range)

# --- 匹配与评估参数 ---
KEYPOINT_THRESHOLD = float(_CFG.matching.keypoint_threshold)
RANSAC_REPROJ_THRESHOLD = float(_CFG.matching.ransac_reproj_threshold)
MIN_INLIERS = int(_CFG.matching.min_inliers)
PYRAMID_SCALES = [float(s) for s in _CFG.matching.pyramid_scales]
INFERENCE_WINDOW_SIZE = int(_CFG.matching.inference_window_size)
INFERENCE_STRIDE = int(_CFG.matching.inference_stride)
IOU_THRESHOLD = float(_CFG.evaluation.iou_threshold)

# --- 文件路径 ---
LAYOUT_DIR = str((_BASE_CONFIG_PATH.parent / _CFG.paths.layout_dir).resolve()) if not Path(_CFG.paths.layout_dir).is_absolute() else _CFG.paths.layout_dir
SAVE_DIR = str((_BASE_CONFIG_PATH.parent / _CFG.paths.save_dir).resolve()) if not Path(_CFG.paths.save_dir).is_absolute() else _CFG.paths.save_dir
VAL_IMG_DIR = str((_BASE_CONFIG_PATH.parent / _CFG.paths.val_img_dir).resolve()) if not Path(_CFG.paths.val_img_dir).is_absolute() else _CFG.paths.val_img_dir
VAL_ANN_DIR = str((_BASE_CONFIG_PATH.parent / _CFG.paths.val_ann_dir).resolve()) if not Path(_CFG.paths.val_ann_dir).is_absolute() else _CFG.paths.val_ann_dir
TEMPLATE_DIR = str((_BASE_CONFIG_PATH.parent / _CFG.paths.template_dir).resolve()) if not Path(_CFG.paths.template_dir).is_absolute() else _CFG.paths.template_dir
MODEL_PATH = str((_BASE_CONFIG_PATH.parent / _CFG.paths.model_path).resolve()) if not Path(_CFG.paths.model_path).is_absolute() else _CFG.paths.model_path