"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/4/17 20:36
@Software: PyCharm
@File : g_color_palette.py
"""
import os
from PIL import Image
import numpy as np
from utils import make_folder
from tqdm import trange

color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

folder_base = './data/CelebAMask-HQ/CelebAMask-HQ-mask'
folder_save = './data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette'
img_num = 30000

make_folder(folder_save)

# Create palette
palette = [c for color in color_list for c in color]

for k in trange(img_num):
    filename_save = os.path.join(folder_save, str(k) + '.png')
    if not os.path.exists(filename_save):
        filename = os.path.join(folder_base, str(k) + '.png')
        im_base = np.zeros((512, 512))
        if os.path.exists(filename):
            im = Image.open(filename)
            im = np.array(im)
            for idx, color in enumerate(color_list):
                im_base[im == idx] = idx

        result = Image.fromarray((im_base).astype(np.uint8))
        result = result.convert('P')
        result.putpalette(palette)
        result.save(filename_save)