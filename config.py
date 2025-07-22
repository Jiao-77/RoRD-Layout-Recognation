# config.py

# --- Training Parameters ---
LEARNING_RATE = 5e-5  # Reduce learning rate for improved training stability
BATCH_SIZE = 8  # Increase batch size for improved training efficiency
NUM_EPOCHS = 50  # Increase training epochs
PATCH_SIZE = 256
# (Optimization) Scale jitter range during training - reduced range for improved stability
SCALE_JITTER_RANGE = (0.8, 1.2) 

# --- Matching and Evaluation Parameters ---
KEYPOINT_THRESHOLD = 0.5
RANSAC_REPROJ_THRESHOLD = 5.0
MIN_INLIERS = 15
IOU_THRESHOLD = 0.5
# (New) Image pyramid scales for template matching during inference
PYRAMID_SCALES = [0.75, 1.0, 1.5]
# (New) Sliding window parameters for processing large layouts during inference
INFERENCE_WINDOW_SIZE = 1024
INFERENCE_STRIDE = 768 # Less than INFERENCE_WINDOW_SIZE to ensure overlap

# --- File Paths ---
# (Paths remain unchanged, please modify according to your environment)
LAYOUT_DIR = 'path/to/layouts'
SAVE_DIR = 'path/to/save'
VAL_IMG_DIR = 'path/to/val/images'
VAL_ANN_DIR = 'path/to/val/annotations'
TEMPLATE_DIR = 'path/to/templates'
MODEL_PATH = 'path/to/save/model_final.pth'