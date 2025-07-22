import os
from PIL import Image
from torch.utils.data import Dataset
import json

class ICLayoutDataset(Dataset):
    def __init__(self, image_dir, annotation_dir=None, transform=None):
        """
        Initialize the IC layout dataset.

        Args:
            image_dir (str): Directory path containing PNG format IC layout images.
            annotation_dir (str, optional): Directory path containing JSON format annotation files.
            transform (callable, optional): Optional transform to apply to images (e.g., Sobel edge detection).
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
        Return the number of images in the dataset.

        Returns:
            int: Dataset size.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get image and annotation at specified index.

        Args:
            idx (int): Image index.

        Returns:
            tuple: (image, annotation), where image is the processed image and annotation is the annotation dict or empty dict.
        """
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        
        annotation = {}
        if self.annotation_dir and self.annotations[idx]:
            ann_path = os.path.join(self.annotation_dir, self.annotations[idx])
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    annotation = json.load(f)
        
        return image, annotation