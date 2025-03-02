import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from dancher_tools.utils.data_loader import DatasetRegistry

@DatasetRegistry.register_dataset('LoveDA')
class LoveDA(Dataset):
    def __init__(self, images_dir, masks_dir, image_filenames, img_size=224, transform=None):
        """
        LoveDA 数据集加载类。
        
        参数:
            images_dir (str): 图像文件夹路径。
            masks_dir (str): 掩码文件夹路径。
            image_filenames (list): 图像文件名列表。
            img_size (int): 图像大小，默认224。
            transform (torchvision.transforms, optional): 图像变换，默认为None。
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_filenames = image_filenames
        self.img_size = img_size
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 加载图像和掩码路径
        image_file_name = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_file_name)
        mask_path = os.path.join(self.masks_dir, image_file_name)

        # 加载和处理图像
        image = Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size))
        image = self.transform(image)

        # 加载和处理掩码（假设掩码已直接映射为类别编号）
        mask = Image.open(mask_path).resize((self.img_size, self.img_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()  # 转换为 long 型类别标签张量

        return image, mask
