"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/4/17 20:34
@Software: PyCharm 
@File : utils.py
"""
import os

def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))
