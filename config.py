# config.py

# --- 训练参数 ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 20  # 增加了训练轮数
PATCH_SIZE = 256

# --- 匹配与评估参数 ---
# 关键点检测的置信度阈值
KEYPOINT_THRESHOLD = 0.5
# RANSAC 重投影误差阈值（像素）
RANSAC_REPROJ_THRESHOLD = 5.0
# RANSAC 判定为有效匹配所需的最小内点数
MIN_INLIERS = 15 # 适当提高以增加匹配的可靠性
# IoU (Intersection over Union) 阈值，用于评估
IOU_THRESHOLD = 0.5

# --- 文件路径 ---
# 训练数据目录
LAYOUT_DIR = 'path/to/layouts'
# 模型保存目录
SAVE_DIR = 'path/to/save'
# 验证集图像目录
VAL_IMG_DIR = 'path/to/val/images'
# 验证集标注目录
VAL_ANN_DIR = 'path/to/val/annotations'
# 模板图像目录
TEMPLATE_DIR = 'path/to/templates'
# 默认加载的模型路径
MODEL_PATH = 'path/to/save/model_final.pth'