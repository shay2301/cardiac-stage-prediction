import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from Scripts import unet
from Scripts import dataloader
from Scripts.dataloader import imshow
import os

root_dir = os.path.dirname(__file__)
csv_full_path = os.path.join(root_dir, 'Database', 'training_database.csv')

# Assuming you have transformations for your frames
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=6)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = unet.UNet(n_channels=1, n_classes=1)

# Usage example
for batch in dataloader:
    # imshow(batch['frames']['esv'][0], batch['masks']['esv'][0], batch['FileName'][0])
    imshow(batch, alpha=0.9)
    # print(batch['FileName'])
    # print(batch['frames'], batch['masks'])