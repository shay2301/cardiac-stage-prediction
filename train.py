import pandas as pd
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from Scripts import unet
from Scripts import dataloader
from Scripts.dataloader import imshow
import os

class DiceLoss(nn.Module):
    def forward(self, input, target):
        smooth = 1.
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))

class iou(nn.Module):
    def forward(self, input, target):
        smooth = 1.
        input = input.view(-1)
        target = target.view(-1)
        intersection = (input * target).sum()
        return (intersection + smooth) / (input.sum() + target.sum() - intersection + smooth)

root_dir = os.path.dirname(__file__)
csv_full_path = os.path.join(root_dir, 'Database', 'training_database.csv')

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=transform, split_group='train')
train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)

val_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=transform, split_group='val')
val_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=True)

test_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=transform, split_group='test')
test_dataloader = DataLoader(test_dataset, batch_size=6)

model = unet.UNet(n_channels=1, n_classes=1)

optimizer = Adam(model.parameters(), lr=0.001)

criterion = nn.BCEWithLogitsLoss()
# criterion = DiceLoss()
# criterion = iou()

trained_model = unet.train(model, optimizer, criterion, train_dataloader, epochs=1)