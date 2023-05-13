"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/4/19 18:48
@Software: PyCharm 
@File : image_datasets.py
"""


import os

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# 递归函数，用于列出指定目录下所有图像文件的路径，并将这些路径存储在一个列表中返回
def list_images(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(list_images(full_path))
    return results



# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_size, image_path, mask_path, image_transformer=None, mask_transformer=None):
        self.image_size = image_size
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_transformer = image_transformer
        self.mask_transformer = mask_transformer

    def __len__(self):
        return len(list_images(self.image_path))

    def __getitem__(self, index):
        image_list = list_images(self.image_path)
        mask_list = list_images(self.mask_path)
        image = Image.open(image_list[index])
        mask = Image.open(mask_list[index])
        if self.image_transformer is not None:
            image = self.image_transformer(image)
        if self.mask_transformer is not None:
            mask = self.mask_transformer(mask)
        return image, mask


def load_data(
    *,
    batch_size,
    image_size=256,
    image_path,
    mask_path,
    image_transformer=None,
    mask_transformer=None,
    shuffle=True,
    num_workers=1,
):

    if not os.path.exists(image_path):
        raise ValueError("image directory does not exist")
    if not os.path.exists(mask_path):
        raise ValueError("mask directory does not exist")

    dataset = ImageDataset(
        image_size,
        image_path,
        mask_path,
        image_transformer=image_transformer,
        mask_transformer=mask_transformer,
    )

    print("Len of Dataset:", dataset.__len__())

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)



# 测试函数
if __name__ == "__main__":

    train_loader = load_data(
        batch_size=1,
        image_size=256,
        image_path="./data/CelebAMask-HQ/CelebA-HQ-img/",
        mask_path="./data/CelebAMask-HQ/CelebAMask-HQ-mask/",
        image_transformer=transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        mask_transformer=transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # transforms.Normalize((0,), (1 / 255.,))
            ]
        ),
        shuffle=False,
    )

    for i, (image, mask) in enumerate(train_loader):
        print(image.shape)
        print(mask.shape)
        plt.imshow(image.numpy().squeeze().transpose((1, 2, 0)))
        plt.show()
        plt.imshow(mask.numpy().squeeze(), cmap='gray')
        plt.show()
        break






