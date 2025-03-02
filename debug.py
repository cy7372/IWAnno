import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import random

import dancher_tools_segmentation as dt

def check_dataloader_integrity(data_loader, num_samples=5):
    """
    检查数据加载器是否正确加载数据，包括每个 batch 中图像和掩码的一致性。
    
    Args:
        data_loader (DataLoader): 数据加载器。
        num_samples (int): 要检查的样本数量。
    
    Returns:
        list: 错误信息列表。
    """
    errors = []

    # 遍历数据加载器中的数据
    iterator = iter(data_loader)
    for _ in range(num_samples):
        images, masks = next(iterator)

        # 检查图像和掩码的形状一致性
        if images.shape[0] != masks.shape[0]:
            errors.append(f"Batch 中图像和掩码数量不匹配: 图像 {images.shape[0]}，掩码 {masks.shape[0]}")

        # 检查图像和掩码的尺寸是否一致
        for i in range(images.shape[0]):
            if images.shape[2:] != masks.shape[2:]:
                errors.append(f"图像和掩码的尺寸不一致: 图像 {images.shape[2:]}，掩码 {masks.shape[2:]}")

        # 检查图像和掩码是否能够正常加载
        try:
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            mask = masks[0].cpu().numpy()[0]
        except Exception as e:
            errors.append(f"无法加载图像或掩码: 错误 {e}")
            continue
        
        # 显示样本（如果没有错误）
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis('off')

        plt.show()

    return errors


def validate_data_loaders(args):
    """
    检查训练集和测试集加载器的完整性。
    
    Args:
        args: 配置参数，包含数据加载器的设置。
    
    Returns:
        list: 错误信息列表。
    """
    # 获取数据加载器
    train_loader, test_loader = dt.utils.get_dataloaders(args)

    print("开始检查训练集数据加载器...")
    train_errors = check_dataloader_integrity(train_loader, num_samples=5)

    print("开始检查测试集数据加载器...")
    test_errors = check_dataloader_integrity(test_loader, num_samples=5)

    # 合并错误信息
    all_errors = train_errors + test_errors
    return all_errors


if __name__ == "__main__":
    # 加载配置文件
    config_path = 'configs/S1/FLaTO.yaml'
    args = dt.utils.get_config(config_path)

    # 检查数据加载器
    errors = validate_data_loaders(args)

    if errors:
        print("在数据加载器中发现了以下错误：")
        for error in errors:
            print(error)
    else:
        print("数据加载器没有发现错误。")
