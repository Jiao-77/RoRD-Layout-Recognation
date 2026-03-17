import os
import json
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

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


class ICLayoutTrainingDataset(Dataset):
    """自监督训练用的 IC 版图数据集，带数据增强与几何配准标签。"""

    def __init__(
        self,
        image_dir: str,
        patch_size: int = 256,
        transform=None,
        scale_range: Tuple[float, float] = (1.0, 1.0),
        use_albu: bool = False,
        albu_params: Optional[dict] = None,
    ) -> None:
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith('.png')
        ]
        self.patch_size = patch_size
        self.transform = transform
        self.scale_range = scale_range
        # 可选的 albumentations 管道
        self.albu = None
        if use_albu:
            try:
                import albumentations as A  # 延迟导入，避免环境未安装时报错
                p = albu_params or {}
                elastic_prob = float(p.get("prob", 0.3))
                alpha = float(p.get("alpha", 40))
                sigma = float(p.get("sigma", 6))
                alpha_affine = float(p.get("alpha_affine", 6))
                use_bc = bool(p.get("brightness_contrast", True))
                use_noise = bool(p.get("gauss_noise", True))
                transforms_list = [
                    A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, p=elastic_prob),
                ]
                if use_bc:
                    transforms_list.append(A.RandomBrightnessContrast(p=0.5))
                if use_noise:
                    transforms_list.append(A.GaussNoise(var_limit=(5.0, 20.0), p=0.3))
                self.albu = A.Compose(transforms_list)
            except Exception:
                self.albu = None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('L')
        width, height = image.size

        # 随机尺度抖动
        scale = float(np.random.uniform(self.scale_range[0], self.scale_range[1]))
        # 确保 scale 有下界，避免除零和过小的 crop_size
        scale = max(scale, 0.1)  # 最小 scale 为 0.1
        crop_size = int(self.patch_size / scale)
        crop_size = max(crop_size, 1)  # 确保 crop_size 至少为 1
        crop_size = min(crop_size, width, height)

        if crop_size <= 0:
            raise ValueError("crop_size must be positive; check scale_range configuration")

        x = np.random.randint(0, max(width - crop_size + 1, 1))
        y = np.random.randint(0, max(height - crop_size + 1, 1))
        patch = image.crop((x, y, x + crop_size, y + crop_size))
        patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)

        # photometric/elastic（在几何 H 之前）
        patch_np_uint8 = np.array(patch)
        if self.albu is not None:
            patch_np_uint8 = self.albu(image=patch_np_uint8)["image"]
            patch = Image.fromarray(patch_np_uint8)
        else:
            # 优化：合并光度增强为单次 numpy 操作，避免多次 point() 遍历
            patch_np = patch_np_uint8.astype(np.float32)
            
            # 亮度调整
            if np.random.random() < 0.5:
                brightness_factor = np.random.uniform(0.8, 1.2)
                patch_np = patch_np * brightness_factor
            
            # 对比度调整
            if np.random.random() < 0.5:
                contrast_factor = np.random.uniform(0.8, 1.2)
                patch_np = (patch_np - 128) * contrast_factor + 128
            
            # 噪声添加
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 5, patch_np.shape)
                patch_np = patch_np + noise
            
            # 裁剪并转换回 uint8
            patch_np = np.clip(patch_np, 0, 255)
            patch_np_uint8 = patch_np.astype(np.uint8)
            patch = Image.fromarray(patch_np_uint8)

        # 随机旋转与镜像（8个离散变换）
        theta_deg = int(np.random.choice([0, 90, 180, 270]))
        is_mirrored = bool(np.random.choice([True, False]))
        center_x, center_y = self.patch_size / 2.0, self.patch_size / 2.0
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), theta_deg, 1.0)

        if is_mirrored:
            translate_to_origin = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])
            mirror = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
            translate_back = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])
            mirror_matrix = translate_back @ mirror @ translate_to_origin
            rotation_matrix_h = np.vstack([rotation_matrix, [0, 0, 1]])
            homography = (rotation_matrix_h @ mirror_matrix).astype(np.float32)
        else:
            homography = np.vstack([rotation_matrix, [0, 0, 1]]).astype(np.float32)

        transformed_patch_np = cv2.warpPerspective(patch_np_uint8, homography, (self.patch_size, self.patch_size))
        transformed_patch = Image.fromarray(transformed_patch_np)

        if self.transform:
            patch_tensor = self.transform(patch)
            transformed_tensor = self.transform(transformed_patch)
        else:
            patch_tensor = torch.from_numpy(np.array(patch)).float().unsqueeze(0) / 255.0
            transformed_tensor = torch.from_numpy(np.array(transformed_patch)).float().unsqueeze(0) / 255.0

        H_tensor = torch.from_numpy(homography[:2, :]).float()
        return patch_tensor, transformed_tensor, H_tensor