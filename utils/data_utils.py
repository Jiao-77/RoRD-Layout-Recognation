from torchvision import transforms
from .transforms import SobelTransform

def get_transform():
    """
    Get unified image preprocessing pipeline.
    Ensure training, evaluation, and inference use exactly the same preprocessing.
    """
    return transforms.Compose([
        SobelTransform(),  # Apply Sobel edge detection
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Adapt to VGG's three-channel input
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])