import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class DiceLoss(nn.Module):
    def forward(self, input, target, smooth=0.1):
        input = torch.sigmoid(input)
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))

class CombinedBCEDiceLoss(nn.Module):
    def __init__(self):
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, input, target):
        bce = self.bce_loss(input, target)
        dice = self.dice_loss(input, target)
        return bce + dice

class iou(nn.Module):
    def forward(self, input, target):
        smooth = 1.
        input = input.view(-1)
        target = target.view(-1)
        intersection = (input * target).sum()
        return (intersection + smooth) / (input.sum() + target.sum() - intersection + smooth)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        true_pos = (inputs * targets).sum()
        false_neg = ((1 - inputs) * targets).sum()
        false_pos = (inputs * (1 - targets)).sum()

        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky_index

class CustomLossWithFPR(nn.Module):
    def __init__(self):
        super(CustomLossWithFPR, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        false_positives = torch.sum((inputs == 1) & (targets == 0)).float()
        fpr_loss = false_positives / (torch.sum(targets == 0).float() + 1e-6)
        return bce_loss + fpr_loss

def train(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dice_loss = DiceLoss()
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            frames = torch.cat((batch['frames']['esv'], batch['frames']['edv']), dim=0)
            masks = torch.cat((batch['masks']['esv'], batch['masks']['edv']), dim=0)
            
            # Add sequence dimension if not present
            if frames.dim() == 4:
                frames = frames.unsqueeze(1)
                masks = masks.unsqueeze(1)
            
            optimizer.zero_grad()
            frames = frames.to(device)
            masks = masks.to(device).float()
            outputs = model(frames)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, train loss: {loss.item()}, train dice coeff: {1-dice_loss(outputs, masks).item()}')
        model.logs_di['epoch'].append(epoch)
        model.logs_di['Dice'].append(loss.item())
        model.logs_di['dataset'].append('train')
                
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                frames = torch.cat((batch['frames']['esv'], batch['frames']['edv']), dim=0)
                masks = torch.cat((batch['masks']['esv'], batch['masks']['edv']), dim=0)
                frames = frames.to(device)
                masks = masks.to(device).float()
                outputs = model(frames)
                loss = criterion(outputs, masks)
                if i % 10 == 0:
                    print(f'Epoch: {epoch}, Batch: {i}, val loss: {loss.item()}, val dice coeff: {1-dice_loss(outputs, masks).item()}')
        model.logs_di['epoch'].append(epoch)
        model.logs_di['Dice'].append(loss.item())
        model.logs_di['dataset'].append('val')
        print(f'Epoch: {epoch}, train loss: {loss.item()}, train dice coeff: {1-dice_loss(outputs, masks).item()}')
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
            nn.ReLU(inplace=True),
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

class UTime(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UTime, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.logs_di = {'epoch': [], 'Dice': [], 'dataset': []}

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        factor = 2 if bilinear else 1
        
        self.down4 = Down(512, 1024 // factor)
        self.lstm = nn.LSTM(1024 // factor * 7 * 7, 1024 // factor * 7 * 7, batch_first=True)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        b, s, c, h, w = x.size()
        x = x.view(b * s, c, h, w)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        b_s, c, h, w = x5.size()
        x5 = x5.view(b, s, -1)

        self.lstm.flatten_parameters()

        x5, _ = self.lstm(x5)

        lstm_output_channels = 1024 // factor
        lstm_output_features = lstm_output_channels * h * w

        x5 = x5.view(b_s, lstm_output_channels, h, w)

        x = self.up1(x5, x4.view(b_s, 512, h, w))
        x = self.up2(x, x3.view(b_s, 256, h * 2, w * 2))
        x = self.up3(x, x2.view(b_s, 128, h * 4, w * 4))
        x = self.up4(x, x1.view(b_s, 64, h * 8, w * 8))
        logits = self.outc(x)
        return logits.view(b, s, self.n_classes, h * 16, w * 16)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))
    
class EfficientUTime(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(EfficientUTime, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.logs_di = {'epoch': [], 'Dice': [], 'dataset': []}

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        
        self.conv_lstm = ConvLSTM(input_channels=256, hidden_channels=256, kernel_size=3, num_layers=1)
        
        self.up1 = Up(256 + 128, 128, bilinear)
        self.up2 = Up(128 + 64, 64, bilinear)
        self.up3 = Up(64 + 32, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # Handle both 4D and 5D inputs
        if x.dim() == 4:
            b, c, h, w = x.size()
            s = 1
            x = x.unsqueeze(1)  # Add a sequence dimension
        elif x.dim() == 5:
            b, s, c, h, w = x.size()
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D instead")

        x = x.view(b * s, c, h, w)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4 = x4.view(b, s, -1, h // 8, w // 8)
        x4, _ = self.conv_lstm(x4)
        x4 = x4.view(b * s, -1, h // 8, w // 8)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)

        return logits.view(b, s, self.n_classes, h, w)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, x):
        b, t, c, h, w = x.size()
        hidden_state = torch.zeros(b, self.hidden_channels, h, w).to(x.device)
        cell_state = torch.zeros(b, self.hidden_channels, h, w).to(x.device)
        
        output = []
        for i in range(t):
            hidden_state, cell_state = self.cell(x[:, i, :, :, :], (hidden_state, cell_state))
            output.append(hidden_state)
        
        return torch.stack(output, dim=1), (hidden_state, cell_state)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next