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

```bash
uv add torch torchvision opencv-python numpy Pillow
uv lock
uv sync
```

或者使用 pip：

```bash
pip install torch torchvision opencv-python numpy Pillow
```

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

```bash
python train.py --data_dir path/to/layouts --save_dir path/to/save
```


--data_dir：包含 PNG 格式 IC 版图图像的目录。

--save_dir：模型权重保存目录。训练过程使用自监督学习，通过随机旋转生成训练对，优化关键点检测和描述子生成。


评估

运行以下命令评估模型性能：

```bash
python evaluate.py --model_path path/to/model.pth --val_dir path/to/val/images --annotations_dir path/to/val/annotations --templates path/to/templates
```


--model_path：训练好的模型权重路径。

--val_dir：验证集图像目录。

--annotations_dir：JSON 格式的真实标注目录。

--templates：模板图像路径列表。评估结果包括精确率、召回率和 F1 分数，基于 IoU（Intersection over Union）阈值。


模板匹配

运行以下命令进行模板匹配：

```bash
python match.py --model_path path/to/model.pth --layout_path path/to/layout.png --template_path path/to/template.png --output_path path/to/output.png
```


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

描述子头：生成旋转鲁棒的 128 维描述子，适配 IC 版图的 8 个离散旋转方向。模型通过自监督学习训练，使用随机旋转（0°~360°）生成训练对，优化检测重复性和描述子相似性。



结果

[待补充：如果有预训练模型或基准测试结果，请在此列出。例如：]


预训练模型：[链接](待补充)

验证集评估指标：精确率：X，召回率：Y，F1 分数：Z


