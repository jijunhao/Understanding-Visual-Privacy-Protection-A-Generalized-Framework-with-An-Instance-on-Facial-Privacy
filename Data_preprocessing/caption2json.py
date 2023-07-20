"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/7/19 9:34
@Software: PyCharm 
@File : caption2json.py
"""
import os
import json
from tqdm import trange


folder_base = './data/celeba-caption/'
folder_save = './datasets/text/'


text ={}

for i in trange(30000):
    filename = os.path.join(folder_base, str(i) + '.txt')
    # 打开文本文件并读取内容
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        # 拼接每一行
        joined_lines = ' '.join([line.strip() for line in lines])

        # 去除重复的句子
        unique_strings = list(set(joined_lines[:-1].split(". ")))


        # 重新拼接
        joined_lines = '. '.join([line.strip() for line in unique_strings])+'.'


        # 将拼接后的结果转换成JSON格式
        text[str(i) + ".jpg"] = {"text":joined_lines.strip()}  # 去除首尾的空格和换行符


json_filename = os.path.join(folder_save, "captions.json")
with open(json_filename, 'w') as json_file:
    json.dump(text, json_file, indent=4)





