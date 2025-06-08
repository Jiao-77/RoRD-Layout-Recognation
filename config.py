# config.py

# --- 训练参数 ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 20
PATCH_SIZE = 256
# (新增) 训练时尺度抖动范围
SCALE_JITTER_RANGE = (0.7, 1.5) 

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