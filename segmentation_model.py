import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SegmentationModel(nn.Module):
    def __init__(self, in_channels=1, depth=16, num_classes=1):
        super().__init__()

        self.enc1 = ConvBlock3D(in_channels, out_channels=depth)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(in_channels=depth, out_channels=depth*2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(in_channels=depth*2, out_channels=depth*4)

        self.up1 = nn.ConvTranspose3d(in_channels=depth*4, out_channels=depth*2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(in_channels=depth*4, out_channels=depth*2)
        self.up2 = nn.ConvTranspose3d(in_channels=depth*2, out_channels=depth, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(in_channels=depth*2, out_channels=depth)

        self.final_conv = nn.Conv3d(in_channels=depth, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))

        x4 = self.up1(x3)
        x4 = self.dec1(torch.cat([x4, x2], dim=1))
        x5 = self.up2(x4)
        x5 = self.dec2(torch.cat([x5, x1], dim=1))

        return self.final_conv(x5)