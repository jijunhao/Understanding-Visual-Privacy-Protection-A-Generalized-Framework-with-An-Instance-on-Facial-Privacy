"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/4/17 20:40
@Software: PyCharm 
@File : show.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    image="/media/node/SSD/jijunhao/CelebAMask-HQ/CelebA-HQ-img/1.jpg"
    mask = "/media/node/SSD/jijunhao/CelebAMask-HQ/CelebAMask-HQ-mask/1.png"
    mask_color="/media/node/SSD/jijunhao/CelebAMask-HQ/CelebAMask-HQ-mask-color/1.png"
    img = plt.imread(image)
    plt.imshow(img)
    plt.show()
    img = plt.imread(mask)
    plt.imshow(img, cmap='gray')
    plt.show()
    img = plt.imread(mask_color)
    plt.imshow(img)
    plt.show()