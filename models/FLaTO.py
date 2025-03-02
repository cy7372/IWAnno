import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
import copy
import timm

# ResNetV2 Backbone
class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block."""
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet model."""
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0]+1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1]+1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2]+1)],
            ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*16, cout=width*32, cmid=width*8, stride=2))] +
                [(f'unit{i}', PreActBottleneck(cin=width*32, cout=width*32, cmid=width*8)) for i in range(2, block_units[3]+1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        x = self.root(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for i in range(len(self.body)):
            x = self.body[i](x)
            features.append(x)
        return x, features  # 不再反转特征列表
    
    
class Embeddings(nn.Module):
    """Construct the embeddings from patch and position embeddings."""
    def __init__(self, img_size, patch_size, hidden_size, in_channels):
        super(Embeddings, self).__init__()

        # 计算 n_patches：总共有多少个 patch
        assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
        n_patches = (img_size // patch_size) * (img_size // patch_size)

        # 调试输出 n_patches
        print(f"n_patches: {n_patches}, img_size: {img_size}, patch_size: {patch_size}")

        # 初始化 position_embeddings 为一个 learnable 参数
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))  # 初始化为 Parameter
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 将输入图像切分为 patch
        x = self.patch_embeddings(x)  # (B, hidden_size, H/patch_size, W/patch_size)
        
        # 展平并转置为 [B, n_patches, hidden_size]
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, hidden_size)

        # 调试输出
        # print(f"x.shape: {x.shape}, position_embeddings.shape: {self.position_embeddings.shape}")

        # 动态调整 position_embeddings 的大小
        n_patches = x.size(1)  # 获取当前输入的 n_patches 数量
        self.position_embeddings = nn.Parameter(self.position_embeddings[:, :n_patches, :])

        # 添加位置编码
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings





class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        # Initialize weights
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.normal_(self.query.bias, std=1e-6)
        nn.init.normal_(self.key.bias, std=1e-6)
        nn.init.normal_(self.value.bias, std=1e-6)
        nn.init.normal_(self.out.bias, std=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [B, seq_len, num_heads, head_size]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_size]

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)  # [B, num_heads, seq_len, head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [B, num_heads, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)  # [B, num_heads, seq_len, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [B, seq_len, num_heads, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [B, seq_len, hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout_rate, attention_dropout_rate):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, mlp_dim, dropout_rate, attention_dropout_rate):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded

class Transformer(nn.Module):
    def __init__(self, img_size, patch_size, hidden_size, num_layers, num_heads, mlp_dim, dropout_rate, attention_dropout_rate, in_channels):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size, patch_size, hidden_size, in_channels)
        self.encoder = Encoder(num_layers, hidden_size, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=True) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)

class FLaTO(nn.Module):
    def __init__(self,in_channels, img_size, num_classes, vis=False):
        super(FLaTO, self).__init__()  # 传递 num_classes 参数
        self.vis = vis
        self.model_name = "FLaTO"
        self.num_classes = num_classes  # 动态设置 num_classes
        

        # 配置参数
        config = {
            'num_layers': 12,
            'hidden_size': 768,
            'num_heads': 12,
            'mlp_dim': 3072,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.0,
            'decoder_channels': [256, 128, 64, 16],
            'n_skip': 3,
            'patch_size': 1,
        }

        # Backbone
        backbone_model = 'resnet50'
        self.backbone = timm.create_model(backbone_model, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3))
        backbone_out_channels = self.backbone.feature_info.channels()

        # Transformer
        transformer_in_channels = backbone_out_channels[-1]
        transformer_img_size = img_size // 16
        self.transformer = Transformer(
            img_size=transformer_img_size,
            patch_size=config['patch_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            mlp_dim=config['mlp_dim'],
            dropout_rate=config['dropout_rate'],
            attention_dropout_rate=config['attention_dropout_rate'],
            in_channels=transformer_in_channels
        )

        # Decoder
        self.decoder_channels = config['decoder_channels']
        self.conv_more = nn.Conv2d(config['hidden_size'], self.decoder_channels[0], kernel_size=3, padding=1)

        in_channels = [self.decoder_channels[0]] + list(self.decoder_channels[:-1])
        out_channels = self.decoder_channels

        skip_channels = backbone_out_channels[:-1][::-1]
        skip_channels += [0]
        if config['n_skip'] != 0:
            for i in range(4 - config['n_skip']):
                skip_channels[-(i + 1)] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = []
        for in_ch, out_ch, skip_ch in zip(in_channels, out_channels, skip_channels):
            blocks.append(DecoderBlock(in_ch, out_ch, skip_ch))
        self.blocks = nn.ModuleList(blocks)

        # Segmentation Head
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=self.num_classes,  # 确保使用 num_classes 设置输出通道数
            kernel_size=3,
        )

        # 初始化权重
        self._init_weights()

    def forward(self, x):
        input_size = x.size()[2:]

        # Backbone
        features = self.backbone(x)

        # Transformer
        x = features[-1]
        x = self.transformer(x)
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        # Decoder
        skips = features[:-1][::-1]
        skips += [None]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i]
            x = decoder_block(x, skip=skip)

        logits = self.segmentation_head(x)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        # 调试：打印 logits 的形状，确认其通道数是否等于 num_classes
        # print(f"Logits shape: {logits.shape}, Expected number of classes: {self.segmentation_head[0].out_channels}")
        return logits

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None and m.bias.requires_grad:
                    nn.init.zeros_(m.bias)
