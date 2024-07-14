import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

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

def train(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dice_loss = DiceLoss()
    upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Define upsampling layer
    
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
            outputs = upsample(outputs)  # Upsample the model output to match the mask size
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                if isinstance(criterion, DiceLoss):
                    print(f'Epoch: {epoch}, Batch: {i}, train Dice loss: {loss.item()}')
                else:
                    print(f'Epoch: {epoch}, Batch: {i}, train loss: {loss.item()}, train dice coeff: {1-dice_loss(outputs, masks).item()}')
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
                outputs = upsample(outputs)  # Upsample the model output to match the mask size
                
                loss = criterion(outputs, masks)
                if i % 10 == 0:
                    if isinstance(criterion, DiceLoss):
                        print(f'Epoch: {epoch}, Batch: {i}, val Dice loss: {loss.item()}')
                    else:
                        print(f'Epoch: {epoch}, Batch: {i}, val loss: {loss.item()}, val dice coeff: {1-dice_loss(outputs, masks).item()}')
        model.logs_di['epoch'].append(epoch)
        model.logs_di['Dice'].append(loss.item())
        model.logs_di['dataset'].append('val')
        if isinstance(criterion, DiceLoss):
            print(f'Epoch: {epoch}, train Dice: {loss.item()}')
        else:
            print(f'Epoch: {epoch}, train loss: {loss.item()}, train dice coeff: {1-dice_loss(outputs, masks).item()}')
    print('Finished Training')
    return model

class TransUNet112(nn.Module):
    def __init__(self, img_size=112, patch_size=8, num_classes=1, in_channels=1):
        super(TransUNet112, self).__init__()
        self.patch_size = patch_size
        
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        config.num_channels = in_channels  # Set number of input channels to 1 for grayscale images
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', config=config, ignore_mismatched_sizes=True)
        
        # Replace the first convolutional layer to accommodate single-channel input
        self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size)
        )
        
        self.unet_decoder = UNetDecoder(img_size, patch_size, num_classes)
        self.logs_di = {'epoch': [], 'Dice': [], 'dataset': []}
    
    def forward(self, x):
        # Step 1: Extract features using ViT
        vit_output = self.vit(pixel_values=x).last_hidden_state
                
        # Step 2: Reshape and process with U-Net decoder
        seg_map = self.unet_decoder(vit_output)
        
        return seg_map

class UNetDecoder(nn.Module):
    def __init__(self, img_size, patch_size, num_classes):
        super(UNetDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=3, padding=1)
    
    def forward(self, x):
        # The shape of x is (batch_size, num_patches, hidden_size)
        batch_size, num_patches, hidden_size = x.shape

        # Remove the [CLS] token (first token)
        x = x[:, 1:, :]
        num_patches -= 1
        
        # Calculate the number of patches along each dimension
        patch_dim = int(num_patches ** 0.5)
        
        # Ensure the reshaping dimensions match
        assert patch_dim * patch_dim == num_patches, f"Number of patches ({num_patches}) is not a perfect square"
        
        # Reshape to (batch_size, hidden_size, patch_dim, patch_dim)
        x = x.permute(0, 2, 1).reshape(batch_size, hidden_size, patch_dim, patch_dim)
        
        # Debugging prints
        
        x = self.upsample(x)
        
        # Debugging prints
        
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        
        # Debugging prints
        
        return x
