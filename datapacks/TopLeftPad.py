import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from dancher_tools_segmentation.utils.data_loader import DatasetRegistry
from torchvision.transforms import Normalize


@DatasetRegistry.register_dataset('TopLeftPad')
class TopLeftPad(Dataset):
    """
    IW Full Scene Dataset 类，裁剪图像、处理颜色映射，并返回与 IWDataset 相同的格式。
    """
    dataset_name = 'TopLeftPad'

    # 定义颜色映射，将 RGB 颜色映射到类别索引
    color_map = {
        (0, 0, 0): 0,         # 背景
        (255, 255, 255): 1    # 内部波
    }

    # 类别名称
    class_name = ["background", "internal wave"]

    # 预先计算颜色映射编码表，避免每次调用时重复计算
    _COLOR_TO_INDEX = {((r << 16) | (g << 8) | b): index 
                       for (r, g, b), index in color_map.items()}

    def __init__(self, data):
        """
        初始化数据集。

        :param data: dict，包括 'images' 和 'masks'，均为 numpy arrays。
        """
        self.images = data['images']
        self.masks = data['masks']
        self.num_samples = self.images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        获取单个样本。

        :param idx: 索引
        :return: (image_tensor, mask_tensor)
        """
        image = self.images[idx]
        mask = self.masks[idx]

        # 转换为 PyTorch tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)

        return image, mask

    @classmethod
    def process_data(cls, filename, images_dir, masks_dir, img_size):
        """
        对单张图像及对应掩码进行预处理：
        - 若图像宽和高均小于 img_size（例如 224），则将图像置于左上角，其余区域填充为黑色；
        - 若图像中任一边大于等于 img_size，则按长边缩放（缩放后长边等于 img_size，另一边小于等于 img_size），
          再将图像置于左上角，空白区域填充为黑色。

        :param filename: 图像文件名
        :param images_dir: 图像目录路径
        :param masks_dir: 掩码目录路径
        :param img_size: 输出图像大小（例如 224）
        :return: (images_list, masks_list)
        """
        # 定义 Normalize 实例
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        
        image_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, os.path.splitext(filename)[0] + ".png")

        # 读取图像和掩码
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        if image is None or mask is None:
            raise FileNotFoundError(f"Failed to load image or mask: {filename}")

        # 确保图像为 RGB 格式
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 获取原始图像尺寸
        height, width = image.shape[:2]
        desired_size = img_size

        # 如果任一边大于等于 desired_size，则按长边缩放，
        # 否则保持原始尺寸（仅进行填充）
        if height >= desired_size or width >= desired_size:
            scale = desired_size / max(height, width)
            new_height = int(round(height * scale))
            new_width = int(round(width * scale))
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        else:
            new_height, new_width = height, width

        # 创建黑色背景，并将图像/掩码置于左上角
        padded_image = np.zeros((desired_size, desired_size, 3), dtype=image.dtype)
        padded_mask = np.zeros((desired_size, desired_size, 3), dtype=mask.dtype)
        padded_image[:new_height, :new_width] = image
        padded_mask[:new_height, :new_width] = mask

        # 图像归一化前的预处理
        processed_image = padded_image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).float()
        image_tensor = normalize(image_tensor)
        processed_image = image_tensor.permute(1, 2, 0).numpy()

        # 应用颜色映射处理掩码
        processed_mask = cls.apply_color_map(padded_mask)

        # 为保持与原格式一致，返回列表形式
        return [processed_image], [processed_mask]

    @classmethod
    def apply_color_map(cls, mask):
        """
        将掩码根据 color_map 转换为类别索引。

        :param mask: 输入掩码 (H, W, 3)
        :return: 编码后的掩码 (H, W)
        """
        # 计算每个像素的编码值
        encoded = ((mask[:, :, 0].astype(np.uint32) << 16) |
                   (mask[:, :, 1].astype(np.uint32) << 8) |
                    mask[:, :, 2].astype(np.uint32))
        # 初始化结果，默认值为 -1 表示未匹配
        indexed_mask = np.full(encoded.shape, -1, dtype=np.int32)
        for key, value in cls._COLOR_TO_INDEX.items():
            indexed_mask[encoded == key] = value

        if np.any(indexed_mask == -1):
            raise ValueError("Found unknown pixel values in mask!")

        return indexed_mask.astype(np.uint8)
