import cv2
import numpy as np
from PIL import Image

class SobelTransform:
    def __call__(self, image):
        """
        Apply Sobel edge detection to enhance geometric boundaries of IC layouts.

        Args:
            image (PIL.Image): Input image (grayscale).

        Returns:
            PIL.Image: Edge-enhanced image.
        """
        img_np = np.array(image)
        sobelx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.hypot(sobelx, sobely)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        return Image.fromarray(sobel)