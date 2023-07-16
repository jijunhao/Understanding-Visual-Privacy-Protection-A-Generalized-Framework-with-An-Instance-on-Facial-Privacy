"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/7/16 14:11
@Software: PyCharm 
@File : g_color_bw.py
"""
import os
from PIL import Image
import numpy as np
from utils import make_folder
from tqdm import trange
color_list = [[0, 0, 0]]+[[255,255,255]]*18

folder_base = './data/CelebAMask-HQ/CelebAMask-HQ-mask'
folder_save = './data/CelebAMask-HQ/CelebAMask-HQ-mask-bw'
img_num = 30000

make_folder(folder_save)

for k in trange(img_num):
    filename_save = os.path.join(folder_save, str(k) + '.png')
    if not os.path.exists(filename_save):  # 检查文件是否存在，如果不存在，则保存图片
        filename = os.path.join(folder_base, str(k) + '.png')
        im_base = np.zeros((512, 512, 3))
        if (os.path.exists(filename)):
            #print(filename)
            im = Image.open(filename)
            im = np.array(im)
            for idx, color in enumerate(color_list):
                im_base[im == idx] = color # 将标签idx（单通道数据）和三通道的颜色对应上，给数据一个色彩填充

        result = Image.fromarray((im_base).astype(np.uint8))
        #print (filename_save)
        result.save(filename_save)