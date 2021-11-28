# coding: utf-8

import numpy as np
import cv2
import random
import os
from scipy.io import loadmat
from tqdm import tqdm

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""


train_txt_path = os.path.join("ecg_competition", "data/train")

CNum = 500     # 挑选多少图片进行计算

imgs = np.zeros([12, 5000])
means, stdevs = [], []

file_list = os.listdir(train_txt_path)

for file in tqdm(file_list):

    data = loadmat(os.path.join(train_txt_path, file))
    ecg_data = data["ecgdata"]

    imgs = np.concatenate((imgs, ecg_data), axis=1)


print(imgs.shape)


for i in range(12):
    pixels = imgs[i,:].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# means.reverse() # BGR --> RGB
# stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))


ormMean = [0.02102842710458631, 0.01870473789104995, -0.002323689225004715, -0.016818765162802455, 0.012285585395475473, 0.008799268047423623, -0.010402479687361263, -0.021703821823799017, -0.005407059704317367, 0.023126367419523692, 0.03341531410865756, 0.016268959850275762]
normStd = [0.3053013037015497, 0.24986147063669015, 0.29824570044148807, 0.23575938566213706, 0.27472264903231897, 0.2288858965974916, 0.3108398675616677, 0.38992954676508795, 0.46233869773182745, 0.40961406719874305, 0.3935515632033764, 0.4898131524605047]
