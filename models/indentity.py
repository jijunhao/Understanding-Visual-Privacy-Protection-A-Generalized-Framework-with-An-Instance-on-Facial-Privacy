"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/5/14 16:07
@Software: PyCharm 
@File : indentity.py
"""
import torch
import torch.nn.functional as F
from backbone import facenet,irse,ir152

import cv2
import numpy as np

class TestFace():
    def __init__(self):
        super(TestFace, self).__init__()
        self.gpu = True
        self.device = torch.device('cuda')if self.gpu else  torch.device('cpu')
        self.facemodels()

    def facemodels(self):
        self.targe_models = {}
        selected=['ir152','irse50','facenet','mobile_face']
        #selected=['facenet']
        for model in selected:
            if model == 'ir152':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                fr_model = ir152.IR_152((112, 112))
                if self.gpu:
                    fr_model.load_state_dict(torch.load('resources/ir152.pth'))
                else:
                    fr_model.load_state_dict(torch.load('resources/ir152.pth',map_location='cpu'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)
            if model == 'irse50':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                fr_model = irse.Backbone(50, 0.6, 'ir_se')
                if self.gpu:
                    fr_model.load_state_dict(torch.load('resources/irse50.pth'))
                else:
                    fr_model.load_state_dict(torch.load('resources/irse50.pth',map_location='cpu'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)
            if model == 'facenet':
                self.targe_models[model] = []
                self.targe_models[model].append((160, 160))
                fr_model = facenet.InceptionResnetV1(num_classes=8631, device=self.device)
                if self.gpu:
                    fr_model.load_state_dict(torch.load('resources/facenet.pth'))

                else:
                    fr_model.load_state_dict(torch.load('resources/facenet.pth',map_location='cpu'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)
            if model == 'mobile_face':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                fr_model = irse.MobileFaceNet(512)
                if self.gpu:
                    fr_model.load_state_dict(torch.load('resources/mobile_face.pth'))

                else:
                    fr_model.load_state_dict(torch.load('resources/mobile_face.pth',map_location='cpu'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)

    def test_verification(self, img_a, img_b):

        self.facemodels()
        targeted_loss_list = []

        for model_name in self.targe_models.keys():
            target_loss_A = self.cal_loss(img_a, img_b, model_name, self.targe_models)
            targeted_loss_list.append(target_loss_A)

        targeted_loss_list=torch.stack(targeted_loss_list)

        return targeted_loss_list

    def cos_simi(self, emb_1, emb_2):
        sims=torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1)
        return torch.mean((sims))

    def cal_loss(self, source, target, model_name, target_models):  # 计算损失
        input_size = target_models[model_name][0]
        fr_model = target_models[model_name][1]
        source_resize = F.interpolate(source, size=input_size, mode='bilinear')
        target_resize = F.interpolate(target, size=input_size, mode='bilinear')
        emb_source = fr_model(source_resize)
        emb_target = fr_model(target_resize).detach()
        cos_loss = torch.cosine_similarity(emb_source, emb_target)
        return cos_loss

"""
if __name__ == '__main__':
    TestFace= TestFace()
    # 读取第一张图片
    data1 = cv2.imread("./data/CelebAMask-HQ/CelebA-HQ-img/0.jpg")
    data1 = np.array(data1, dtype=np.float32) / 255.0
    data1 = np.transpose(data1, (2, 0, 1))
    data1 = torch.from_numpy(data1).unsqueeze(0).to('cuda')
    # 读取第二张图片
    data2 = cv2.imread("./data/CelebAMask-HQ/CelebA-HQ-img/1.jpg")
    data2 = np.array(data2, dtype=np.float32) / 255.0
    data2 = np.transpose(data2, (2, 0, 1))
    data2 = torch.from_numpy(data2).unsqueeze(0).to('cuda')


    loss = TestFace.test_verification(data1, data2)
    print(loss)

    loss = TestFace.test_verification(data1, data1)
    print(loss)
"""



