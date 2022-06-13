# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:41:09 2022

@author: mjl
"""

import cv2
import math

from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

#PSNR(峰值性噪比)，SSIM（结构相似性评价）
img1 = cv2.imread('ori.png')
img2 = cv2.imread('ps.png')
img3= cv2.imread('gan.png')
img1 = cv2.resize(img1,(int(img1.shape[1]*img2.shape[0]/img1.shape[0]),img2.shape[0]),interpolation=cv2.INTER_AREA)
RMSE1 = round(math.sqrt(mean_squared_error(img1, img2)),3)
PSNR1 = round(peak_signal_noise_ratio(img1, img2),3)
SSIM1 = round(structural_similarity(img1, img2, multichannel=True),3)
print('MSE1: ', RMSE1)
print('PSNR1: ', PSNR1)
print('SSIM1: ', SSIM1)
RMSE2 = round(math.sqrt(mean_squared_error(img1, img3)),3)
PSNR2 = round(peak_signal_noise_ratio(img1, img3),3)
SSIM2 = round(structural_similarity(img1, img3, multichannel=True),3)
print('MSE2: ', RMSE2)
print('PSNR2: ', PSNR2)
print('SSIM2: ', SSIM2)


import pandas as pd
import matplotlib.pyplot as plt

fig, ax =plt.subplots(1,1)
data=[[RMSE1,PSNR1,SSIM1],
      [RMSE2,PSNR2,SSIM2]]
column_labels=["RMSE", "PSNR", "SSIM"]
df=pd.DataFrame(data,columns=column_labels)
ax.axis('off')
the_table = ax.table(cellText=df.values,
        colLabels=df.columns,
        colWidths = [0.15, 0.15, 0.15],
        rowLabels=["Baseline","Stylegan2"],
        rowColours =["yellow"] * 2,  
        colColours =["green"] * 3,
        loc="center")
the_table.auto_set_font_size(False)
the_table.set_fontsize(15)
the_table.scale(2, 2)

plt.show()