基于 AI 的集成电路版图识别：RoRD 模型
描述
本项目实现了 RoRD（Rotation-Robust Descriptors）模型，用于集成电路（IC）版图识别。RoRD 是一种先进的局部特征匹配方法，具有旋转鲁棒性，特别适合于 IC 版图，因为它们可能以各种方向出现（0°、90°、180°、270°及其镜像）。项目通过自监督学习和随机旋转增强，解决了数据稀缺性、几何多变性、动态扩展性和结构复杂性等挑战。
项目包括：

模型实现：适用于 IC 版图的 RoRD 模型，使用 PyTorch，基于 D2-Net 架构。
数据加载：自定义数据集类 ICLayoutDataset，用于加载光栅化的 IC 版图图像。
训练脚本：通过随机旋转进行自监督训练，确保模型对旋转鲁棒。
评估脚本：在验证集上评估模型性能，计算精确率、召回率和 F1 分数。
匹配工具：使用训练好的模型进行模板匹配，支持多实例匹配和可视化。

安装
环境要求

Python 3.8 或更高版本
CUDA（可选，用于 GPU 加速）

依赖安装
使用 uv 安装依赖库：
uv add torch torchvision opencv-python numpy Pillow
uv lock
uv sync

或者使用 pip：
pip install torch torchvision opencv-python numpy Pillow

使用方法
项目结构
ic_layout_recognition/
├── data/
│   ├── ic_dataset.py
├── utils/
│   ├── transforms.py
├── models/
│   ├── rord.py
├── train.py
├── evaluate.py
├── match.py
├── requirements.txt
└── README.md

训练
运行以下命令训练模型：
python train.py --data_dir path/to/layouts --save_dir path/to/save


--data_dir：包含 PNG 格式 IC 版图图像的目录。
--save_dir：模型权重保存目录。训练过程使用自监督学习，通过随机旋转生成训练对，优化关键点检测和描述子生成。

评估
运行以下命令评估模型性能：
python evaluate.py --model_path path/to/model.pth --val_dir path/to/val/images --annotations_dir path/to/val/annotations --templates path/to/templates


--model_path：训练好的模型权重路径。
--val_dir：验证集图像目录。
--annotations_dir：JSON 格式的真实标注目录。
--templates：模板图像路径列表。评估结果包括精确率、召回率和 F1 分数，基于 IoU（Intersection over Union）阈值。

模板匹配
运行以下命令进行模板匹配：
python match.py --model_path path/to/model.pth --layout_path path/to/layout.png --template_path path/to/template.png --output_path path/to/output.png


--layout_path：版图图像路径。
--template_path：模板图像路径。
--output_path：可视化结果保存路径（可选）。匹配过程使用 RoRD 模型提取关键点和描述子，通过互最近邻（MNN）匹配和 RANSAC 几何验证，生成边界框并支持多实例匹配。

数据准备
训练数据

格式：PNG 格式的 IC 版图图像，从 GDSII 或 OASIS 文件光栅化。
要求：数据集应包含多个版图图像，建议分辨率适中（如 1024x1024）。
路径：存储在 path/to/layouts 目录中。

验证数据

图像：PNG 格式的验证集图像，存储在 path/to/val/images。
注释：JSON 格式的真实标注，存储在 path/to/val/annotations，示例：{
    "boxes": [
        {"template": "template1.png", "x": 100, "y": 200, "width": 50, "height": 50},
        {"template": "template2.png", "x": 300, "y": 400, "width": 60, "height": 60}
    ]
}


模板：模板图像存储在 path/to/templates，文件名需与注释中的 template 字段一致。

模型
RoRD 模型基于 D2-Net 架构，使用 VGG-16 作为骨干网络。它包括：

检测头：用于关键点检测，输出概率图。
描述子头：生成旋转鲁棒的 128 维描述子，适配 IC 版图的 8 个离散旋转方向。模型通过自监督学习训练，使用随机旋转（0°~360°）生成训练对，优化检测重复性和描述子相似性。gi


结果
[待补充：如果有预训练模型或基准测试结果，请在此列出。例如：]

预训练模型：[链接](待补充)
验证集评估指标：精确率：X，召回率：Y，F1 分数：Z

贡献
欢迎贡献代码或提出建议！请遵循以下步骤：

Fork 本仓库。
创建新分支（git checkout -b feature/your-feature）。
提交更改（git commit -m "Add your feature"）。
推送到分支（git push origin feature/your-feature）。
提交 Pull Request。

许可证
本项目采用 MIT 许可证。
联系
如有问题或建议，请通过 [您的电子邮件] 联系或在 GitHub 上提交 issue。

AI-based Integrated Circuit Layout Recognition with RoRD
Description
This project implements the RoRD (Rotation-Robust Descriptors) model for integrated circuit (IC) layout recognition. RoRD is a state-of-the-art method for local feature matching that is robust to rotations, making it particularly suitable for IC layouts which can be oriented in various directions (0°, 90°, 180°, 270°, and their mirrors). The project addresses challenges such as data scarcity, geometric variability, dynamic scalability, and structural complexity through self-supervised learning and random rotation augmentation.
The project includes:

Model Implementation: The RoRD model adapted for IC layouts, using PyTorch, based on the D2-Net architecture.
Data Loading: Custom dataset class ICLayoutDataset for loading rasterized IC layout images.
Training Script: Self-supervised training with random rotations to ensure rotation robustness.
Evaluation Script: Evaluates model performance on a validation set, computing precision, recall, and F1 score.
Matching Utility: Performs template matching with the trained model, supporting multi-instance matching and visualization.

Installation
Requirements

Python 3.8 or higher
CUDA (optional, for GPU acceleration)

Dependency Installation
Install dependencies using uv:
uv add torch torchvision opencv-python numpy Pillow
uv lock
uv sync

Alternatively, use pip:
pip install torch torchvision opencv-python numpy Pillow

Usage
Project Structure
ic_layout_recognition/
├── data/
│   ├── ic_dataset.py
├── utils/
│   ├── transforms.py
├── models/
│   ├── rord.py
├── train.py
├── evaluate.py
├── match.py
├── requirements.txt
└── README.md

Training
Run the following command to train the model:
python train.py --data_dir path/to/layouts --save_dir path/to/save


--data_dir: Directory containing PNG format IC layout images.
--save_dir: Directory to save model weights.The training process uses self-supervised learning, generating training pairs with random rotations to optimize keypoint detection and descriptor generation.

Evaluation
Run the following command to evaluate model performance:
python evaluate.py --model_path path/to/model.pth --val_dir path/to/val/images --annotations_dir path/to/val/annotations --templates path/to/templates


--model_path: Path to the trained model weights.
--val_dir: Directory containing validation images.
--annotations_dir: Directory containing JSON format ground truth annotations.
--templates: List of template image paths.Evaluation metrics include precision, recall, and F1 score, based on IoU (Intersection over Union) thresholds.

Template Matching
Run the following command to perform template matching:
python match.py --model_path path/to/model.pth --layout_path path/to/layout.png --template_path path/to/template.png --output_path path/to/output.png


--layout_path: Path to the layout image.
--template_path: Path to the template image.
--output_path: Path to save visualization results (optional).The matching process extracts keypoints and descriptors using the RoRD model, performs mutual nearest neighbor (MNN) matching, and applies RANSAC for geometric verification, generating bounding boxes for multiple instances.

Data Preparation
Training Data

Format: PNG format IC layout images, rasterized from GDSII or OASIS files.
Requirements: The dataset should include multiple layout images, preferably with moderate resolution (e.g., 1024x1024).
Path: Stored in path/to/layouts.

Validation Data

Images: PNG format validation images, stored in path/to/val/images.
Annotations: JSON format ground truth annotations, stored in path/to/val/annotations, example:{
    "boxes": [
        {"template": "template1.png", "x": 100, "y": 200, "width": 50, "height": 50},
        {"template": "template2.png", "x": 300, "y": 400, "width": 60, "height": 60}
    ]
}


Templates: Template images stored in path/to/templates, with filenames matching the template field in annotations.

Model
The RoRD model is based on the D2-Net architecture, using VGG-16 as the backbone. It includes:

Detection Head: Outputs a probability map for keypoint detection.
Descriptor Head: Generates 128-dimensional rotation-robust descriptors, tailored for the 8 discrete rotation directions in IC layouts.The model is trained using self-supervised learning with random rotations (0°~360°), optimizing for detection repeatability and descriptor similarity.

Technical Comparison



Feature
U-Net
YOLO
Transformer (ViT)
SuperPoint
RoRD



Core Principle
Semantic Segmentation
Object Detection
Global Self-Attention
Self-Supervised Features
Rotation-Robust Features


Data Requirement
Large Pixel-Level Labels
Large Bounding Box Labels
Massive Pretraining Data
Synthetic Data
Synthetic Rotation Data


Rotation Robustness
Low
Low-Medium
Medium
Medium-High
Very High


Results
[To be added: If pre-trained models or benchmarks are available, list them here. For example:]

Pre-trained model: [link]
Validation set metrics: Precision: X, Recall: Y, F1 Score: Z

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Submit a Pull Request.

License
This project is licensed under the MIT License.
Contact
For questions or issues, please contact [your email] or open an issue on GitHub.
