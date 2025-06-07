import cv2
import numpy as np
from PIL import Image

class SobelTransform:
    def __call__(self, image):
        """
        应用 Sobel 边缘检测，增强 IC 版图的几何边界。

        参数：
            image (PIL.Image): 输入图像（灰度图）。

        返回：
            PIL.Image: 边缘增强后的图像。
        """
        img_np = np.array(image)
        sobelx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.hypot(sobelx, sobely)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        return Image.fromarray(sobel)