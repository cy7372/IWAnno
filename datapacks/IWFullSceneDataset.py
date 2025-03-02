import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from dancher_tools_segmentation.utils.data_loader import DatasetRegistry
from torchvision.transforms import Normalize


@DatasetRegistry.register_dataset('IWFullSceneDataset')
class IWFullSceneDataset(Dataset):
    """
    IW Full Scene Dataset 类，裁剪图像，处理颜色映射，并返回与 IWDataset 相同的格式。
    """
    dataset_name = 'IWFullSceneDataset'

    # 定义颜色映射，将 RGB 颜色映射到类别索引
    color_map = {
        (0, 0, 0): 0,         # 背景
        (255, 255, 255): 1    # 内部波
    }

    # 类别名称
    class_name = ["background", "internal wave"]

    def __init__(self, data):
        """
        初始化数据集。

        :param data: dict，包括 'images' 和 'masks'，numpy arrays。
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

        # 转换为 PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)

        return image, mask


    @classmethod
    def process_data(cls, filename, images_dir, masks_dir, img_size):
        """
        使用生成器逐步输出裁剪结果，避免一次性将所有裁剪块加载到内存。
        
        :param filename: 图像文件名
        :param images_dir: 图像目录路径
        :param masks_dir: 掩码目录路径
        :param img_size: 每个裁剪块的尺寸
        :yield: (cropped_image, processed_mask) 对，每次一块裁剪结果
        """
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, os.path.splitext(filename)[0] + ".png")

        # 读取图像和掩码
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        if image is None or mask is None:
            raise FileNotFoundError(f"Failed to load image or mask: {filename}")

        # 确保图像为 RGB 格式
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        height, width = image.shape[:2]
        step_size = img_size // 2

        if height < img_size or width < img_size:
            raise ValueError(f"Image {filename} is smaller than crop size {img_size}")

        # 逐块裁剪并立即输出
        for row in range(0, height - img_size + 1, step_size):
            for col in range(0, width - img_size + 1, step_size):
                cropped_image = image[row:row + img_size, col:col + img_size]
                cropped_mask = mask[row:row + img_size, col:col + img_size]

                # 应用颜色映射
                processed_mask = cls.apply_color_map(cropped_mask)

                # 图像归一化前的预处理
                cropped_image = cropped_image.astype(np.float32) / 255.0

                # 转换为张量并应用归一化
                cropped_image_tensor = torch.from_numpy(cropped_image).permute(2, 0, 1).float()
                cropped_image_tensor = normalize(cropped_image_tensor)

                # 转回 NumPy 格式
                cropped_image = cropped_image_tensor.permute(1, 2, 0).numpy()

                yield cropped_image, processed_mask


    @classmethod
    def apply_color_map(cls, mask):
        """
        将掩码根据 color_map 转换为类别索引。

        :param mask: 输入掩码 (H, W, 3)。
        :return: 编码后的掩码 (H, W)。
        """
        mask_encoded = (
            (mask[:, :, 0].astype(np.uint32) << 16) |
            (mask[:, :, 1].astype(np.uint32) << 8) |
            (mask[:, :, 2].astype(np.uint32))
        )
        color_keys_encoded = (
            (np.array(list(cls.color_map.keys()), dtype=np.uint32)[:, 0] << 16) |
            (np.array(list(cls.color_map.keys()), dtype=np.uint32)[:, 1] << 8) |
            (np.array(list(cls.color_map.keys()), dtype=np.uint32)[:, 2])
        )
        color_to_index = dict(zip(color_keys_encoded, cls.color_map.values()))

        indexed_mask = np.vectorize(lambda x: color_to_index.get(x, None))(mask_encoded)

        if np.any(indexed_mask == None):
            raise ValueError("Found unknown pixel values in mask!")

        return indexed_mask.astype(np.uint8)
    
    