import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class DiceLoss(nn.Module):
    def forward(self, input, target, smooth=10.):
        input = torch.sigmoid(input)
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
    
def train(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            frames_esv = batch['frames']['esv']
            frames_edv = batch['frames']['edv']
            masks_esv = batch['masks']['esv']
            masks_edv = batch['masks']['edv']
            frames = torch.cat((frames_esv, frames_edv), dim=0)
            masks = torch.cat((masks_esv, masks_edv), dim=0)
            optimizer.zero_grad()
            frames = frames.to(device)
            masks = masks.to(device).float()
            outputs = model(frames)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, train Dice: {loss.item()}')
        model.logs_di['epoch'].append(epoch)
        model.logs_di['Dice'].append(loss.item())
        model.logs_di['dataset'].append('train')
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                frames_esv = batch['frames']['esv']
                frames_edv = batch['frames']['edv']
                masks_esv = batch['masks']['esv']
                masks_edv = batch['masks']['edv']
                frames = torch.cat((frames_esv, frames_edv), dim=0)
                masks = torch.cat((masks_esv, masks_edv), dim=0)
                frames = frames.to(device)
                masks = masks.to(device).float()
                outputs = model(frames)
                loss = criterion(outputs, masks)
                if i % 10 == 0:
                    print(f'Epoch: {epoch}, Batch: {i}, val Dice: {loss.item()}')
        model.logs_di['epoch'].append(epoch)
        model.logs_di['Dice'].append(loss.item())
        model.logs_di['dataset'].append('val')
        print(f'Epoch: {epoch}, val Dice: {loss.item()}')
    print('Finished Training')
    return model

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.logs_di = {'epoch':[], 'Dice':[], 'dataset':[]}

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def predict(self, x):
        return F.softmax(self.forward(x), dim=1)