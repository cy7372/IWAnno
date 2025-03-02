import torch
import torch.nn as nn
from .FLaTO import FLaTO  # 假设 FLaTO 存在于当前目录的 FLaTO.py 中

# 定义适配器模块
class AdapterModule(nn.Module):
    def __init__(self, in_channels, adapter_channels=64):
        super(AdapterModule, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, adapter_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(adapter_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.adapter(x)  # 使用残差连接以保留输入特征

# 继承 FLaTO 模型以创建适配器增强的迁移学习模型
class FLaTOada(FLaTO):
    def __init__(self, in_channels, img_size=224, num_classes=1, vis=False, adapter_channels=64):
        super(FLaTOada, self).__init__(in_channels, img_size, num_classes)  # 调用父类构造函数

        # 冻结原始 FLaTO 模型的 Backbone 层以支持迁移学习
        for param in self.backbone.parameters():
            param.requires_grad = False  # 在迁移学习中通常会冻结原来的 backbone

        # 为每个 Backbone 特征输出添加一个适配器模块
        backbone_out_channels = self.backbone.feature_info.channels()  # [64, 256, 512, 1024]
        self.adapters = nn.ModuleList([AdapterModule(ch, adapter_channels) for ch in backbone_out_channels])

    def forward(self, x):
        input_size = x.size()[2:]  # 输入图像尺寸

        # Backbone 前向传播，获取特征
        features = self.backbone(x)  # 形状: [(B, 64, 112, 112), (B, 256, 56, 56), ...]

        # 使用适配器模块对每层 Backbone 特征进行调整
        adapted_features = [adapter(f) for adapter, f in zip(self.adapters, features)]

        # Transformer 输入最后一层适配器特征
        x = adapted_features[-1]  # 使用适配后的最后一层特征
        x = self.transformer(x)

        # 将 Transformer 输出恢复为原始特征尺寸
        B, n_patch, hidden = x.size()
        h, w = int(n_patch ** 0.5), int(n_patch ** 0.5)
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        # 获取跳跃连接特征并使用 Decoder 进行解码
        skips = adapted_features[:-1][::-1] + [None]  # 添加 None 以匹配 Decoder 层数
        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x, skip=skips[i])

        logits = self.segmentation_head(x)
        logits = nn.functional.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return logits
