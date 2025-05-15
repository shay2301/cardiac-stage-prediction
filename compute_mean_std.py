import os
import torch
from torchvision import transforms
from Scripts import dataloader
from torch.utils.data import DataLoader


def compute_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for batch in loader:
        frames = torch.cat((batch['frames']['esv'], batch['frames']['edv']), dim=0)
        batch_samples = frames.size(0)  # batch size (the last batch can have smaller size)
        frames = frames.view(batch_samples, frames.size(1), -1)
        mean += frames.mean(2).sum(0)
        std += frames.std(2).sum(0)
        total_images_count += batch_samples
        print(total_images_count)

    mean /= total_images_count
    std /= total_images_count
    return mean, std

root_dir = os.path.dirname(__file__)

input_transform = transforms.Compose([
    transforms.ToTensor()
])
# Assuming you have a DataLoader for your dataset
train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, input_transform=input_transform, target_transform=None, split_group='train')
train_loader = DataLoader(train_dataset, batch_size=10)
mean, std = compute_mean_std(train_loader)
print(f'Mean: {mean}, Std: {std}')