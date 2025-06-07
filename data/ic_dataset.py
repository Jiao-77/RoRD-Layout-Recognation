import os
from PIL import Image
from torch.utils.data import Dataset
import json

class ICLayoutDataset(Dataset):
    def __init__(self, image_dir, annotation_dir=None, transform=None):
        """
        初始化 IC 版图数据集。

        参数：
            image_dir (str): 存储 PNG 格式 IC 版图图像的目录路径。
            annotation_dir (str, optional): 存储 JSON 格式注释文件的目录路径。
            transform (callable, optional): 应用于图像的可选变换（如 Sobel 边缘检测）。
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        if annotation_dir:
            self.annotations = [f.replace('.png', '.json') for f in self.images]
        else:
            self.annotations = [None] * len(self.images)

    def __len__(self):
        """
        返回数据集中的图像数量。

        返回：
            int: 数据集大小。
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取指定索引的图像和注释。

        参数：
            idx (int): 图像索引。

        返回：
            tuple: (image, annotation)，image 为处理后的图像，annotation 为注释字典或空字典。
        """
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        if self.transform:
            image = self.transform(image)
        
        annotation = {}
        if self.annotation_dir and self.annotations[idx]:
            ann_path = os.path.join(self.annotation_dir, self.annotations[idx])
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    annotation = json.load(f)
        
        return image, annotation