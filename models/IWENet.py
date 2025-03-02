import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Squeeze operation
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        # Excitation operation
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        # Scale the input features
        return x * y.expand_as(x)

# Double Convolution Block
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

# U-Net with SE Blocks (IWENet)
class IWENet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(IWENet, self).__init__()
        self.model_name = "IWENet"
        
        # Encoding path
        self.encoder1 = DoubleConv(in_channels, 64)
        self.se1 = SEBlock(64)
        
        self.encoder2 = DoubleConv(64, 128)
        self.se2 = SEBlock(128)
        
        self.encoder3 = DoubleConv(128, 256)
        self.se3 = SEBlock(256)
        
        self.encoder4 = DoubleConv(256, 512)
        self.se4 = SEBlock(512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Decoding path
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        se_enc1 = self.se1(enc1)
        enc2 = self.encoder2(self.pool(se_enc1))
        se_enc2 = self.se2(enc2)
        enc3 = self.encoder3(self.pool(se_enc2))
        se_enc3 = self.se3(enc3)
        enc4 = self.encoder4(self.pool(se_enc3))
        se_enc4 = self.se4(enc4)
        
        # Decoding path
        dec1 = self.upconv1(se_enc4)
        dec1 = torch.cat((dec1, se_enc3), dim=1)
        dec1 = self.decoder1(dec1)
        
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((dec2, se_enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec3 = self.upconv3(dec2)
        dec3 = torch.cat((dec3, se_enc1), dim=1)
        dec3 = self.decoder3(dec3)
        
        output = self.out_conv(dec3)
        return output  # 不再应用 sigmoid

