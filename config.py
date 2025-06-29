# config.py

# --- 训练参数 ---
LEARNING_RATE = 5e-5  # 降低学习率，提高训练稳定性
BATCH_SIZE = 8  # 增加批次大小，提高训练效率
NUM_EPOCHS = 50  # 增加训练轮数
PATCH_SIZE = 256
# (优化) 训练时尺度抖动范围 - 缩小范围提高稳定性
SCALE_JITTER_RANGE = (0.8, 1.2) 

# --- 匹配与评估参数 ---
KEYPOINT_THRESHOLD = 0.5
RANSAC_REPROJ_THRESHOLD = 5.0
MIN_INLIERS = 15
IOU_THRESHOLD = 0.5
# (新增) 推理时模板匹配的图像金字塔尺度
PYRAMID_SCALES = [0.75, 1.0, 1.5]
# (新增) 推理时处理大版图的滑动窗口参数
INFERENCE_WINDOW_SIZE = 1024
INFERENCE_STRIDE = 768 # 小于INFERENCE_WINDOW_SIZE以保证重叠

# --- 文件路径 ---
# (路径保持不变, 请根据您的环境修改)
LAYOUT_DIR = 'path/to/layouts'
SAVE_DIR = 'path/to/save'
VAL_IMG_DIR = 'path/to/val/images'
VAL_ANN_DIR = 'path/to/val/annotations'
TEMPLATE_DIR = 'path/to/templates'
MODEL_PATH = 'path/to/save/model_final.pth'