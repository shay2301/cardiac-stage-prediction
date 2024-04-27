from torchvision import transforms
import torch
import PIL
import random

class SameSeed:
    def __init__(self, seed=42):
        self.seed = seed

    def __call__(self, img):
        torch.manual_seed(self.seed)
        return img

class RandomChoiceTransform:
    def __init__(self, transforms, seed=42):
        self.transforms = transforms
        self.seed = seed

    def __call__(self, img):
        random.seed(self.seed)
        transform = random.choice(self.transforms)
        return transform(img)
    
input_transform = transforms.Compose([
    SameSeed(),
    RandomChoiceTransform([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0, 360), resample=PIL.Image.BILINEAR),
    ], seed=42),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

target_transform = transforms.Compose([
    SameSeed(),
    RandomChoiceTransform([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0, 360), resample=PIL.Image.BILINEAR),
    ], seed=42),
    transforms.ToTensor(),
])