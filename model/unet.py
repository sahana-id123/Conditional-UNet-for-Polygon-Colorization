import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, color_emb_dim=16):
        super().__init__()
        self.color_emb_dim = color_emb_dim

        # Color embedding
        self.color_embed = nn.Embedding(10, color_emb_dim)  # max 10 colors for now

        # Downsampling
        self.down1 = DoubleConv(in_channels + color_emb_dim, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)
        
        # Upsampling
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, color_idx):
        B, C, H, W = x.shape
        color_emb = self.color_embed(color_idx)  # (B, emb_dim)
        color_map = color_emb.unsqueeze(2).unsqueeze(3).expand(B, self.color_emb_dim, H, W)
        x = torch.cat([x, color_map], dim=1)

        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bn = self.bottleneck(self.pool2(d2))

        u2 = self.up2(bn)
        dec2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([u1, d1], dim=1))

        return torch.sigmoid(self.out(dec1))
