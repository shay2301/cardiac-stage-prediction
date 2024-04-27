import pandas as pd
from torch import nn
from torch import save, load
from torch.optim import Adam
from torchvision import transforms
from Scripts.data_transforms import input_transform, target_transform
from torch.utils.data import Dataset, DataLoader
from Scripts import unet
from Scripts import dataloader
from Scripts.dataloader import imshow
import os

root_dir = os.path.dirname(__file__)
csv_full_path = os.path.join(root_dir, 'Database', 'training_database.csv')

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x/255)
])

batch_size = 64

train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, input_transform=input_transform, target_transform=target_transform, split_group='train')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, input_transform=input_transform, target_transform=target_transform, split_group='val')
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, input_transform=input_transform, target_transform=target_transform, split_group='test')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model = unet.UNet(n_channels=1, n_classes=1)

optimizer = Adam(model.parameters(), lr=0.01)

# criterion = nn.BCEWithLogitsLoss()
criterion = unet.DiceLoss()
# criterion = unet.iou()


trained_model = unet.train(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=6)

save(trained_model.state_dict(), os.path.join(root_dir, 'models', 'model_dice_6epochs.pth'))
