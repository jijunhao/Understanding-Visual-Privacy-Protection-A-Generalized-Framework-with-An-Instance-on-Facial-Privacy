"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/8/4 0:41
@Software: PyCharm 
@File : ID.py
"""
import numpy as np
import torch
import cv2
import os
from models import identity
TestFace = indentity.TestFace()

data = cv2.imread("")
data = np.array(data, dtype=np.float32) / 255.0
data = np.transpose(data, (2, 0, 1))
data = torch.from_numpy(data).unsqueeze(0).to('cuda')
data1 = cv2.imread("/home/jijunhao/diffusion/datasets/image/image_512_downsampled_from_hq_1024/"+str(0)+".jpg")
data1 = np.array(data1, dtype=np.float32) / 255.0
data1 = np.transpose(data1, (2, 0, 1))
data1 = torch.from_numpy(data1).unsqueeze(0).to('cuda')

data2 = cv2.imread("/home/jijunhao/diffusion/datasets/image/image_512_downsampled_from_hq_1024/"+str(26)+".jpg")
data2 = np.array(data2, dtype=np.float32) / 255.0
data2 = np.transpose(data2, (2, 0, 1))
data2 = torch.from_numpy(data2).unsqueeze(0).to('cuda')


ps = TestFace.test_verification(data,data1)
po = TestFace.test_verification(data,data2)

print(ps,po,ps/(ps+po))