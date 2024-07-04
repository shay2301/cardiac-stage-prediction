import os
import pandas as pd
from torch import nn, save, load
from torch.optim import SGD
from torchvision import transforms
from Scripts.data_transforms import base_transform, horizontal_flip_transform, vertical_flip_transform, rotate_90_transform, rotate_180_transform, rotate_270_transform, noise_transform
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from Scripts.models import unet
from Scripts import dataloader

root_dir = os.path.dirname(__file__)
csv_full_path = os.path.join(root_dir, 'Database', 'training_database.csv')

batch_size = 64

base_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=base_transform, split_group='train')
horizontal_flipped_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=horizontal_flip_transform, split_group='train')
vertical_flipped_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=vertical_flip_transform, split_group='train')
rotated_90_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_90_transform, split_group='train')
rotated_180_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_180_transform, split_group='train')
rotated_270_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_270_transform, split_group='train')
noised_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=noise_transform, split_group='train', noise_flag=True)

train_dataset = ConcatDataset([base_train_dataset, 
                               horizontal_flipped_train_dataset, 
                               vertical_flipped_train_dataset, 
                               rotated_90_train_dataset, 
                               rotated_180_train_dataset, 
                               rotated_270_train_dataset,
                               noised_train_dataset])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=base_transform, split_group='val')
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=base_transform, split_group='test')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model = unet.UNet(n_channels=1, n_classes=1)

optimizer = SGD(model.parameters(), lr=0.1)

# criterion = nn.BCEWithLogitsLoss()
# criterion = unet.DiceLoss()
# criterion = unet.iou()
criterion = unet.CombinedBCEDiceLoss()
# criterion = unet.TverskyLoss()
# criterion = unet.CustomLossWithFPR()

trained_model = unet.train(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=6)

model_name = 'model_dice_6epochs_augmented.pth'
save(trained_model.state_dict(), os.path.join(root_dir, 'models', model_name))
save(trained_model.logs_di, os.path.join(root_dir, 'models', 'logs_' + model_name))
