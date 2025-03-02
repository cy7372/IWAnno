# datapacks/iw_dataset.py

import os
import cv2
import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from dancher_tools_segmentation.utils.data_loader import DatasetRegistry  # 确保正确的导入路径

@DatasetRegistry.register_dataset('iw_dataset')
class IWDataset(Dataset):
    """
    IW 数据分割数据集类，负责特定的数据处理逻辑。
    """

    dataset_name = 'iw_dataset'  # 确保与注册名一致

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

        :param data: dict，包括 'images' 和 'masks'，numpy arrays
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
                 - image_tensor: 形状为 (3, img_size, img_size)，类型为 torch.float32
                 - mask_tensor: 形状为 (img_size, img_size)，类型为 torch.long
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
        处理单个图像和掩码，并以生成器方式输出处理结果。
        这里使用 yield 替换原来的 return，使得数据处理更灵活（比如支持生成器接口），
        有助于上层代码以分块方式按需加载数据，从而降低内存峰值。

        :param filename: 图像文件名
        :param images_dir: 图像目录路径
        :param masks_dir: 掩码目录路径
        :param img_size: 输出图像大小（整数）
        :yield: (processed_image, processed_mask)
                - processed_image: 归一化后的图像，
                  形状为 (img_size, img_size, 3)，类型为 float32
                - processed_mask: 类别索引的掩码，
                  形状为 (img_size, img_size)，类型为 uint8
        """
        # 定义归一化
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

        # 构建图像路径并读取图像
        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 预处理图像：resize 和归一化
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]

        # 转换为张量并归一化
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        image_tensor = normalize(image_tensor)
        processed_image = image_tensor.permute(1, 2, 0).numpy()  # (H, W, C)

        # 构建掩码文件名并读取掩码
        mask_filename = filename.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(masks_dir, mask_filename)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        # 将掩码从 BGR 转为 RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # 二值化掩码
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, bin_mask = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)
        mask = cv2.merge([bin_mask, bin_mask, bin_mask])

        # 应用颜色映射
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
            problem_mask = (indexed_mask == None)
            problem_values = mask_encoded[problem_mask]
            unique_problem_values = np.unique(problem_values)
            print(f"[ERROR] Unknown pixel values in mask '{mask_path}': {unique_problem_values}")
            raise ValueError(
                f"apply_color_map found unknown color(s) in mask '{mask_path}': {unique_problem_values}"
            )

        processed_mask = indexed_mask.astype(np.uint8)

        # 缩放掩码
        zoom_factor = (img_size / processed_mask.shape[0], img_size / processed_mask.shape[1])
        processed_mask = zoom(processed_mask, zoom_factor, order=0)

        # 使用 yield 输出处理结果。
        # 对于 iw_dataset，每个图像仅生成一个处理结果，但使用生成器接口，
        # 保持与其他支持分块输出的数据集一致。
        yield processed_image, processed_mask
