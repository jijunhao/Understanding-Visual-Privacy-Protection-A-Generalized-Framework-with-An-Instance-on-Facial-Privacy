"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/4/8 9:14
@Software: PyCharm 
@File : DCGAN.py
"""


import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from tqdm import tqdm



# 定义生成器和判别器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 128 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 128 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# 定义训练函数
def train(dataloader, discriminator, generator, optimizer_D, optimizer_G, criterion, num_epochs):
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(tqdm(dataloader)):
            # 将图像数据和标签移到设备上
            real_imgs = real_imgs.to(device)
            real_labels = torch.ones((real_imgs.size(0), 1)).to(device)
            fake_labels = torch.zeros((real_imgs.size(0), 1)).to(device)
            # 训练判别器
            optimizer_D.zero_grad()
            # 训练判别器识别真实图像
            real_outputs = discriminator(real_imgs)
            loss_D_real = criterion(real_outputs, real_labels)
            # 训练判别器识别虚假图像
            z = torch.randn((real_imgs.size(0), latent_dim)).to(device)
            fake_imgs = generator(z)
            fake_outputs = discriminator(fake_imgs.detach())
            loss_D_fake = criterion(fake_outputs, fake_labels)
            # 计算总损失并更新判别器参数
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()
            # 训练生成器
            optimizer_G.zero_grad()
            # 训练生成器生成虚假图像并让判别器认为是真实图像
            output = discriminator(fake_imgs)
            loss_G = criterion(output, real_labels)
            # 更新生成器参数
            loss_G.backward()
            optimizer_G.step()

        # 输出训练结果
        tqdm.write('Epoch [{}/{}], Step [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, len(dataloader), loss_D.item(), loss_G.item()))
        if epoch%100==0:
            z = torch.randn((real_imgs.size(0), latent_dim)).to(device)
            fake_imgs = generator(z)
            save_image(fake_imgs, f"../output/epoch_{epoch}.png", normalize=True)

        # 保存生成器模型
        torch.save(generator.state_dict(), '../resources/generator.pth')
        torch.save(discriminator.state_dict(), '../resources/discriminator.pth')


if __name__ == '__main__':
    # 定义超参数
    batch_size = 64
    latent_dim = 100

    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    num_epochs = 1000

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载数据集
    dataset = ImageFolder(root='/media/node/SSD/jijunhao/data/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化生成器和判别器
    # generator.load_state_dict(torch.load('generator.pth')).to(device)
    # discriminator.load_state_dict(torch.load('discriminator.pth')).to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.load_state_dict(torch.load('../resources/generator.pth'))
    discriminator.load_state_dict(torch.load('../resources/discriminator.pth'))

    # 定义优化器和损失函数
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # 定义真实标签和虚假标签
    real_label = 1
    fake_label = 0

    # 定义损失函数
    criterion = nn.BCELoss()

    # 训练模型
    train(dataloader, discriminator, generator, optimizer_D, optimizer_G, criterion, num_epochs)
