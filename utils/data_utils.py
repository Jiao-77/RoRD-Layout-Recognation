from torchvision import transforms
from .transforms import SobelTransform

def get_transform():
    """
    获取统一的图像预处理管道。
    确保训练、评估和推理使用完全相同的预处理。
    """
    return transforms.Compose([
        SobelTransform(),  # 应用 Sobel 边缘检测
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # 适配 VGG 的三通道输入
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])