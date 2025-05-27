import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return F.relu(x + self.block(x))

class AttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_weights = self.attn(x)
        return x * attn_weights

#Encoder
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock(out_ch)
        )

    def forward(self, x):
        return self.conv(x)

#Decoder
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock(out_ch)
        )
        self.attn = AttentionBlock(out_ch)

    def forward(self, x, skip):
        
        x = self.up(x)
        #print(x.shape, skip.shape)
        
        if x.shape[2:] != skip.shape[2:]:
            skip = skip[:, :, :x.shape[2], :x.shape[3]]
            print(skip.shape)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attn(x)
        return x


class ResidualUNetWithAttention(nn.Module):
    def __init__(self, in_channels=9, out_channels=3, base_ch=64):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = ResidualBlock(base_ch * 8)

        self.up3 = UpBlock(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_bottleneck = self.bottleneck(x4)

        x = self.up3(x_bottleneck, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        return self.out_conv(x)
