import pandas as pd
from torch import nn
from torch import sigmoid
from torch import no_grad, load
from torchvision import transforms
from Scripts.data_transforms import base_transform
from torch.utils.data import Dataset, DataLoader
from Scripts.models import unet
from Scripts import dataloader
from Scripts.dataloader import imshow
import os


root_dir = os.path.dirname(__file__)
csv_full_path = os.path.join(root_dir, 'Database', 'training_database.csv')

batch_size = 1

test_dataset = dataloader.VideoFrameDataset(root_dir=root_dir, transform=base_transform, split_group='test')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

## import saved model
saved_model = unet.UNet(n_channels=1, n_classes=1)
saved_model.load_state_dict(load(os.path.join(root_dir, 'models', 'model_dice_6epochs.pth')))

for i, batch in enumerate(test_dataloader):

    saved_model.eval()
    with no_grad():
        esv_outputs = sigmoid(saved_model(batch['frames']['esv'])) > 0.9
        edv_outputs = sigmoid(saved_model(batch['frames']['edv'])) > 0.9

    imshow(batch, alpha=0.5, threshold=0.8)
