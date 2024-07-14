import torch
from torch import nn
import timm

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

class ViTForSegmentation(nn.Module):
    def __init__(self, num_classes=1, patch_size=8, img_size=112):
        super(ViTForSegmentation, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=img_size)

        # Use the correct attribute to get the embedding dimension
        patch_embed_dim = self.backbone.patch_embed.proj.out_channels

        # Modify patch size
        self.backbone.patch_embed.proj = nn.Conv2d(1, patch_embed_dim, kernel_size=patch_size, stride=patch_size)

        # Update positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.backbone.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, patch_embed_dim))

        self.backbone.head = nn.Identity()  # Remove the classification head
        self.conv = nn.Conv2d(in_channels=patch_embed_dim, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.backbone.patch_embed(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x.permute(0, 2, 1).reshape(B, self.backbone.patch_embed.proj.out_channels, H // self.backbone.patch_embed.proj.kernel_size[0], W // self.backbone.patch_embed.proj.kernel_size[1])
        x = self.conv(x)
        x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            frames = batch['frames']['esv']
            masks = batch['masks']['esv']
            frames = frames.to(device)
            masks = masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{len(train_dataloader)}, Loss: {loss.item()} Dice: {1 - DiceLoss()(outputs, masks).item()}")
        print(f"Epoch {epoch+1} finished, Loss: {running_loss/len(train_dataloader)} Dice: {1 - DiceLoss()(outputs, masks).item()}")
        for i, batch in enumerate(val_dataloader):
            frames = batch['frames']['esv']
            masks = batch['masks']['esv']
            frames = frames.to(device)
            masks = masks.to(device).float()
            outputs = model(frames)
            loss = criterion(outputs, masks)
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{len(val_dataloader)}, Validation Loss: {loss.item()} Dice: {1 - DiceLoss()(outputs, masks).item()}")
        print(f"Validation Loss: {loss.item()} Dice: {1 - DiceLoss()(outputs, masks).item()}")
    print('Finished Training')
