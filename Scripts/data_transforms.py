from torchvision import transforms
import numpy as np
import torch
import random
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

# def normalize_image(image):
#     image = image.astype(np.float32)
#     image = (image - np.min(image)) / (np.max(image) - np.min(image))
#     return image

class RandomAddNoise:
    def __init__(self, mean=0.0, std=0.01, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                img = self.normalize_image(img)
                noise = (np.random.randn(*img.shape) * self.std + self.mean).astype(np.float32)
                noisy_img = img.astype(np.float32) + noise
                noisy_img = self.normalize_image(noisy_img)
                return Image.fromarray((noisy_img * 255).astype(np.uint8))
            elif isinstance(img, torch.Tensor):
                noise = torch.randn(img.size(), dtype=torch.float32) * self.std + self.mean
                noisy_img = img + noise
                noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
                return transforms.ToPILImage()(noisy_img)
            elif isinstance(img, Image.Image):
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = self.normalize_image(img_array)
                noise = (np.random.randn(*img_array.shape) * self.std + self.mean).astype(np.float32)
                noisy_img_array = img_array + noise
                noisy_img_array = self.normalize_image(noisy_img_array)
                return Image.fromarray((noisy_img_array * 255).astype(np.uint8))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        else:
            if isinstance(img, np.ndarray):
                return Image.fromarray(img.astype(np.uint8))
            elif isinstance(img, torch.Tensor):
                return transforms.ToPILImage()(img)
            elif isinstance(img, Image.Image):
                return img
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            
    @staticmethod
    def normalize_image(img):
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val + 1e-5)

class SameSeed:
    def __init__(self, seed=42):
        self.seed = seed

    def __call__(self, img):
        torch.manual_seed(self.seed)
        return img

base_transform = transforms.Compose([
    transforms.ToPILImage(),
    # RandomAddNoise(mean=0.0, std=0.1),
    transforms.ToTensor(),
])

horizontal_flip_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.hflip(x),
    transforms.ToTensor(),
])

vertical_flip_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.vflip(x),
    transforms.ToTensor(),
])

rotate_15_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 15),
    transforms.ToTensor(),
])

rotate_30_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 30),
    transforms.ToTensor(),
])

rotate_45_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 45),
    transforms.ToTensor(),
])

rotate_60_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 60),
    transforms.ToTensor(),
])

rotate_75_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 75),
    transforms.ToTensor(),
])


rotate_90_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 90),
    transforms.ToTensor(),
])

rotate_105_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 105),
    transforms.ToTensor(),
])

rotate_120_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 120),
    transforms.ToTensor(),
])

rotate_135_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 135),
    transforms.ToTensor(),
])

rotate_150_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 150),
    transforms.ToTensor(),
])

rotate_165_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 165),
    transforms.ToTensor(),
])

rotate_180_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 180),
    transforms.ToTensor(),
])

rotate_195_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 195),
    transforms.ToTensor(),
])

rotate_210_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 210),
    transforms.ToTensor(),
])

rotate_225_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 225),
    transforms.ToTensor(),
])

rotate_240_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 240),
    transforms.ToTensor(),
])

rotate_255_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 255),
    transforms.ToTensor(),
])

rotate_270_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: F.rotate(x, 270),
    transforms.ToTensor(),
])


noise_transform = transforms.Compose([
    transforms.ToPILImage(),
    RandomAddNoise(mean=0.0, std=0.1, p=1.0),
    transforms.ToTensor(),
])

video_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1283,), (0.1901,))
])

class VideoTransforms:
    """ Transform to convert video frames for input into 3D CNN, handling single-channel grayscale images. """
    def __init__(self):
        self.transform = transforms.Compose([
            SameSeed(),
            transforms.RandomRotation((0, 360)),
            #  transforms.Resize((80, 80)),
            transforms.Normalize((0.1283,), (0.1901,))
        ])

    def __call__(self, clip):
        # Assume clip is a numpy array of shape [H, W, Max_frames]
        # Convert to [Max_frames, H, W] to simulate a batch of grayscale images
        clip = np.transpose(clip, (2, 0, 1))
        transformed_clip = []
        for frame in clip:
            # Ensure frame is in [1, H, W] shape as a single-channel tensor
            frame_tensor = torch.from_numpy(frame).unsqueeze(0).float()

            # Apply the composed transforms
            transformed_frame = self.transform(frame_tensor)
            transformed_clip.append(transformed_frame)

        # Stack all frames along a new dimension to create a single tensor

        return torch.transpose(torch.stack(transformed_clip, dim=0), 1, 0)