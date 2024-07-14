import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn, optim
from Scripts import dataloader
from Scripts.data_transforms import base_transform, horizontal_flip_transform, vertical_flip_transform, rotate_15_transform, rotate_30_transform, rotate_45_transform, rotate_60_transform, rotate_75_transform, rotate_90_transform, rotate_105_transform, rotate_120_transform, rotate_135_transform, rotate_150_transform, rotate_165_transform, rotate_180_transform, rotate_195_transform, rotate_210_transform, rotate_225_transform, rotate_240_transform, rotate_255_transform, rotate_270_transform, noise_transform
from Scripts.models.vit import ViTForSegmentation, train, CombinedBCEDiceLoss, TverskyLoss

root_dir = os.path.dirname(__file__)
csv_full_path = os.path.join(root_dir, 'Database', 'training_database.csv')

batch_size = 64  # Reduced batch size to lower memory usage further

base_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=base_transform, split_group='train')
horizontal_flipped_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=horizontal_flip_transform, split_group='train')
vertical_flipped_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=vertical_flip_transform, split_group='train')
rotated_15_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_15_transform, split_group='train')
rotated_30_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_30_transform, split_group='train')
rotated_45_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_45_transform, split_group='train')
rotated_60_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_60_transform, split_group='train')
rotated_75_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_75_transform, split_group='train')
rotated_90_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_90_transform, split_group='train')
rotated_105_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_105_transform, split_group='train')
rotated_120_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_120_transform, split_group='train')
rotated_135_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_135_transform, split_group='train')
rotated_150_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_150_transform, split_group='train')
rotated_165_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_165_transform, split_group='train')
rotated_180_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_180_transform, split_group='train')
rotated_195_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_195_transform, split_group='train')
rotated_210_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_210_transform, split_group='train')
rotated_225_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_225_transform, split_group='train')
rotated_240_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_240_transform, split_group='train')
rotated_255_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_255_transform, split_group='train')
rotated_270_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=rotate_270_transform, split_group='train')
noised_train_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=noise_transform, split_group='train', noise_flag=True)

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

val_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=base_transform, split_group='val')
## print how big is the dataset
print(f"Training dataset size: {len(train_dataset)}")
full_training_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, persistent_workers=False)  # Reduced num_workers further
validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, persistent_workers=False)  # Reduced num_workers further

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, optimizer, and loss function
model = ViTForSegmentation(num_classes=1, patch_size=8).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = CombinedBCEDiceLoss()
# criterion = TverskyLoss()

# Train the model
train(model, full_training_dataloader, validation_dataloader, criterion, optimizer, device, epochs=6)

# Save the trained model
model_save_path = os.path.join(root_dir, 'models', 'vit_segmentation_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
