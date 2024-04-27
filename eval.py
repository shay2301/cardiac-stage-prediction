import pandas as pd
from torch import nn
from torch import no_grad, load
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

batch_size = 6

test_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, input_transform=input_transform, target_transform=target_transform, split_group='test')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

## import saved model
saved_model = unet.UNet(n_channels=1, n_classes=1)
saved_model.load_state_dict(load(os.path.join(root_dir, 'models', 'model_dice_6epochs.pth')))

for i, batch in enumerate(test_dataloader):
    esv_imgs = batch['frames']['esv']
    edv_imgs = batch['frames']['edv']
    esv_masks = batch['masks']['esv']
    edv_masks = batch['masks']['edv']
    filenames = batch['FileName']

    saved_model.eval()
    with no_grad():
        esv_outputs = saved_model(esv_imgs)
        edv_outputs = saved_model(edv_imgs)

    imshow(batch, [esv_outputs, edv_outputs], alpha=0.5)
