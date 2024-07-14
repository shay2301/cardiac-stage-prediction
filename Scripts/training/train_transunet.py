import os
import pandas as pd
import torch
from torch import nn, save
from torch.optim import SGD
from torchvision import transforms
from Scripts.data_transforms import base_transform, horizontal_flip_transform, vertical_flip_transform, rotate_15_transform, rotate_30_transform, rotate_45_transform, rotate_60_transform, rotate_75_transform, rotate_90_transform, rotate_105_transform, rotate_120_transform, rotate_135_transform, rotate_150_transform, rotate_165_transform, rotate_180_transform, rotate_195_transform, rotate_210_transform, rotate_225_transform, rotate_240_transform, rotate_255_transform, rotate_270_transform, noise_transform
from torch.utils.data import DataLoader, ConcatDataset
from Scripts.models import transunet
from Scripts import dataloader

root_dir = os.path.dirname(__file__)
root_dir = os.path.split(os.path.split(root_dir)[0])[0]
csv_full_path = os.path.join(root_dir, 'Database', 'training_database.csv')

batch_size = 64

base_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=base_transform, split_group='train', resize_flag=True)
horizontal_flipped_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=horizontal_flip_transform, split_group='train', resize_flag=True)
vertical_flipped_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=vertical_flip_transform, split_group='train', resize_flag=True)
rotated_15_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_15_transform, split_group='train', resize_flag=True)
rotated_30_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_30_transform, split_group='train', resize_flag=True)
rotated_45_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_45_transform, split_group='train', resize_flag=True)
rotated_60_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_60_transform, split_group='train', resize_flag=True)
rotated_75_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_75_transform, split_group='train', resize_flag=True)
rotated_90_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_90_transform, split_group='train', resize_flag=True)
rotated_105_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_105_transform, split_group='train', resize_flag=True)
rotated_120_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_120_transform, split_group='train', resize_flag=True)
rotated_135_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_135_transform, split_group='train', resize_flag=True)
rotated_150_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_150_transform, split_group='train', resize_flag=True)
rotated_165_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_165_transform, split_group='train', resize_flag=True)
rotated_180_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_180_transform, split_group='train', resize_flag=True)
rotated_195_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_195_transform, split_group='train', resize_flag=True)
rotated_210_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_210_transform, split_group='train', resize_flag=True)
rotated_225_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_225_transform, split_group='train', resize_flag=True)
rotated_240_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_240_transform, split_group='train', resize_flag=True)
rotated_255_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_255_transform, split_group='train', resize_flag=True)
rotated_270_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_270_transform, split_group='train', resize_flag=True)
noised_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=noise_transform, split_group='train', noise_flag=True, resize_flag=True)

train_dataset = ConcatDataset([base_train_dataset, 
                               horizontal_flipped_train_dataset, 
                               vertical_flipped_train_dataset, 
                               rotated_15_train_dataset,
                               rotated_30_train_dataset,
                               rotated_45_train_dataset,
                               rotated_60_train_dataset,
                               rotated_75_train_dataset,
                               rotated_90_train_dataset,
                               rotated_105_train_dataset,
                               rotated_120_train_dataset,
                               rotated_135_train_dataset,
                               rotated_150_train_dataset,
                               rotated_165_train_dataset,
                               rotated_180_train_dataset,
                               rotated_195_train_dataset,
                               rotated_210_train_dataset,
                               rotated_225_train_dataset,
                               rotated_240_train_dataset,
                               rotated_255_train_dataset,
                               rotated_270_train_dataset,
                               noised_train_dataset])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=base_transform, split_group='val', resize_flag=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=base_transform, split_group='test')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transunet.TransUNet112(img_size=224, patch_size=8, num_classes=1, in_channels=1)  # Specify in_channels=1
model.to(device)

optimizer = SGD(model.parameters(), lr=0.1)

# criterion = nn.BCEWithLogitsLoss()
# criterion = transunet.DiceLoss()
# criterion = transunet.iou()
criterion = transunet.CombinedBCEDiceLoss()
# criterion = transunet.TverskyLoss()
# criterion = transunet.CustomLossWithFPR()

trained_model = transunet.train(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=6)

model_name = 'transunet_6epochs.pth'
save(trained_model.state_dict(), os.path.join(root_dir, 'models', model_name))
save(trained_model.logs_di, os.path.join(root_dir, 'models', 'logs_' + model_name))

