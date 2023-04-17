"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/4/17 20:33
@Software: PyCharm 
@File : g_mask.py
"""

from tqdm import trange


import os
import cv2
import numpy as np
from utils import make_folder

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

# 输入数据集的名称一定要检查一下，这个名称错了不会报错，但是输出的mask里面全部都是空值
folder_base = '/media/node/SSD/jijunhao/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
folder_save = '/media/node/SSD/jijunhao/CelebAMask-HQ/CelebAMask-HQ-mask'
img_num = 30000 # 数据集一共有30000张图片

make_folder(folder_save)

for k in trange(img_num):
    folder_num = k // 2000 # 该图片的分割组件存放的目录，一共有15个目录，每个目录存了2000张分割结果（包含一张图片的面部各个组件分开的分割结果）
    im_base = np.zeros((512, 512))
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if (os.path.exists(filename)):
            #print (label, idx+1)
            im = cv2.imread(filename)
            im = im[:, :, 0]  # 取出图像第一个通道的值（分割图像只有一个通道，但是是部分的组件）
            im_base[im != 0] = (idx + 1) # 将该部分的值赋予一个idx+1的数值，实现分割，后期填充上颜色就变成我们看到的最终分割结果了

    filename_save = os.path.join(folder_save, str(k) + '.png')
    # print (filename_save)
    cv2.imwrite(filename_save, im_base) # 保存图片
