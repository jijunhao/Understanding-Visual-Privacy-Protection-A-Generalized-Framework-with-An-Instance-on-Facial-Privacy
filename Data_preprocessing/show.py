"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/4/17 20:40
@Software: PyCharm 
@File : show.py
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
if __name__ == '__main__':
    image = "./data/CelebAMask-HQ/CelebA-HQ-img/0.jpg"
    mask = "./data/CelebAMask-HQ/CelebAMask-HQ-mask/0.png"
    mask_color = "./data/CelebAMask-HQ/CelebAMask-HQ-mask-color/0.png"

    # 读取图像数据
    img1 = plt.imread(image)
    img2 = plt.imread(mask)
    img3 = plt.imread(mask_color)

    print(img1.shape, img2.shape,img3.shape)


    # # 创建具有 1 行和 3 列的子图
    # fig, axs = plt.subplots(1, 3, figsize=(8, 3))
    #
    # # 在每个子图中显示相应的图像
    # axs[0].imshow(img1)
    # axs[1].imshow(img2, cmap='gray')
    # axs[2].imshow(img3)
    #
    # # 设置子图标题
    # axs[0].set_title('Image')
    # axs[1].set_title('Mask')
    # axs[2].set_title('Mask Color')
    #
    # # 取消坐标轴
    # axs[0].set_axis_off()
    # axs[1].set_axis_off()
    # axs[2].set_axis_off()

    # 文件路径
    image_path = "./data/CelebAMask-HQ/CelebA-HQ-img/9801.jpg"
    mask_path = "./data/CelebAMask-HQ/CelebAMask-HQ-mask-bw/9801.png"

    # 读取图像数据
    image = Image.open(image_path)
    image = image.resize((512, 512))


    mask = plt.imread(mask_path)

    image = np.array(image)
    image[mask == 0] = 255
    #image = image*mask+image*(1-mask)



    # 显示处理后的图像
    #plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')  # 去掉坐标轴

    # 显示图像
    plt.show()

    # 保存图像为矢量图
    plt.savefig('./outputs/show0.png', bbox_inches='tight', pad_inches=0)