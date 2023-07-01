"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/4/17 20:40
@Software: PyCharm 
@File : show.py
"""

import matplotlib.pyplot as plt


if __name__ == '__main__':
    image = "./data/CelebAMask-HQ/CelebA-HQ-img/0.jpg"
    mask = "./data/CelebAMask-HQ/CelebAMask-HQ-mask/0.png"
    mask_color = "./data/CelebAMask-HQ/CelebAMask-HQ-mask-color/0.png"

    # 读取图像数据
    img1 = plt.imread(image)
    img2 = plt.imread(mask)
    img3 = plt.imread(mask_color)

    # 创建具有 1 行和 3 列的子图
    fig, axs = plt.subplots(1, 3, figsize=(8, 3))

    # 在每个子图中显示相应的图像
    axs[0].imshow(img1)
    axs[1].imshow(img2, cmap='gray')
    axs[2].imshow(img3)

    # 设置子图标题
    axs[0].set_title('Image')
    axs[1].set_title('Mask')
    axs[2].set_title('Mask Color')

    # 取消坐标轴
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    axs[2].set_axis_off()

    # 显示图像
    plt.show()

    # 保存图像为矢量图
    plt.savefig('./outputs/show0.svg', format='svg')