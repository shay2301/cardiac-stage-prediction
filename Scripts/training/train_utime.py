import pandas as pd
from torch import nn, save
from torch.optim import Adam
from torchvision import transforms
from Scripts.data_transforms import base_transform, horizontal_flip_transform, vertical_flip_transform, rotate_90_transform, rotate_180_transform, rotate_270_transform, noise_transform
from torch.utils.data import DataLoader, ConcatDataset
from Scripts.models import utime
from Scripts import dataloader
import sys
import os

# Add the path to the Scripts directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'Scripts'))

root_dir = os.path.dirname(__file__)
root_dir = os.path.split(os.path.split(root_dir)[0])[0]
csv_full_path = os.path.join(root_dir, 'Database', 'training_database.csv')

# Reduce batch size to avoid cuDNN issues
batch_size = 64

print("Initializing datasets and dataloaders...")
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

model = utime.EfficientUTime(n_channels=1, n_classes=1)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = utime.CombinedBCEDiceLoss()

print("Starting training...")
trained_model = utime.train(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=6)
print("Training completed successfully.")

model_dir = os.path.join(root_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

model_name = 'utime_dice_6epochs.pth'
save(trained_model.state_dict(), os.path.join(root_dir, 'models', model_name))
save(trained_model.logs_di, os.path.join(root_dir, 'models', 'logs_' + model_name))
print(f"Model saved as {model_name}.")

