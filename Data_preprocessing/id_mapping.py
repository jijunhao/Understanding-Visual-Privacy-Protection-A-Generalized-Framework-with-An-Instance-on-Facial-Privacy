"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/5/9 19:03
@Software: PyCharm 
@File : id_mapping.py
"""
import pandas as pd
import os

table1 = pd.read_csv('./data/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt', sep='\s+')
table2 = pd.read_csv('./data/identity_CelebA.txt', names=['orig_file', 'id'], sep=' ')
table = pd.merge(table1, table2, on='orig_file', how='left', suffixes=('_left', '_right'))

# 如果文件不存在，则保存
if not os.path.exists('./data/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping-identity.txt'):
    table.to_csv('./data/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping-identity.txt', sep='\t', index=False)

print(table.head())

