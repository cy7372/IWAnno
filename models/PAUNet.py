import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PAUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(PAUNet, self).__init__()
        self.model_name = 'paunet'

        # U-Net 编码器 (下采样)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # U-Net 解码器 (上采样)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128, 64)

        # 最后一层
        self.conv_last = nn.Conv2d(64, num_classes, kernel_size=1)

        # 像素注意力模块
        self.pixel_attention1 = self.pixel_attention(512)
        self.pixel_attention2 = self.pixel_attention(256)
        self.pixel_attention3 = self.pixel_attention(128)

    def conv_block(self, in_channels, out_channels):
        """卷积块：两个卷积层 + 批归一化 + ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def pixel_attention(self, in_channels):
        """像素注意力模块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码器部分 (下采样)
        e1 = self.encoder1(x)  # [B, 64, H, W]
        e2 = self.encoder2(F.max_pool2d(e1, kernel_size=2))  # [B, 128, H/2, W/2]
        e3 = self.encoder3(F.max_pool2d(e2, kernel_size=2))  # [B, 256, H/4, W/4]
        e4 = self.encoder4(F.max_pool2d(e3, kernel_size=2))  # [B, 512, H/8, W/8]

        # 解码器部分 (上采样 + 拼接跳跃连接)
        d4 = self.upconv4(e4)  # [B, 256, H/4, W/4]
        d4 = torch.cat((d4, e3), dim=1)  # 拼接跳跃连接
        d4 = self.pixel_attention1(d4) * d4  # 应用像素注意力
        d4 = self.decoder4(d4)  # [B, 256, H/4, W/4]

        d3 = self.upconv3(d4)  # [B, 128, H/2, W/2]
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.pixel_attention2(d3) * d3
        d3 = self.decoder3(d3)  # [B, 128, H/2, W/2]

        d2 = self.upconv2(d3)  # [B, 64, H, W]
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.pixel_attention3(d2) * d2
        d2 = self.decoder2(d2)  # [B, 64, H, W]

        # 最后一层输出
        out = self.conv_last(d2)  # [B, 1, H, W]

        return out
