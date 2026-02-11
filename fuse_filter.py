import math
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os
import time

from heatmap_MI import img_heatmap_mi
from heatmap_CD import img_heatmap_cd


ratio_mi = 0.5 # ratio_cd = 1-ratio_mi
kernel_param = 80
thresh_param = 80 # percentile, from small to big

def fuse_heatmap(img_path, orig_H, orig_W):
    '''
    Объединение тепловых карт MI и CD
    
    :param img_path: путь к картинке
    :param orig_H: исходная высота изображения
    :param orig_W: исходная ширина изображения
    '''

    heatmap_MI = img_heatmap_mi(img_path)
    print('heatmap MI shape', heatmap_MI.shape)

    heatmap_CD, _ = img_heatmap_cd(img_path)
    middle_idx = len(heatmap_CD) // 2
    heatmap_CD = heatmap_CD[middle_idx]
    print('heatmap CD shape', heatmap_CD.shape)

    heatmap_MI = cv2.resize(heatmap_MI, (orig_W, orig_H))
    print('heatmap MI resize to original size')
    heatmap_CD = cv2.resize(heatmap_CD, (orig_W, orig_H))
    print('heatmap CD resize to original size')

    # этот кусок можно было бы перенести в отдельную функцию
    heatmap_MI_max = np.max(heatmap_MI)
    heatmap_MI_min = np.min(heatmap_MI)
    print('heatmap_MI_max:', heatmap_MI_max)
    print('heatmap_MI_min:', heatmap_MI_min)
    heatmap_CD_max = np.max(heatmap_CD)
    heatmap_CD_min = np.min(heatmap_CD)
    print('heatmap_CD_max:', heatmap_CD_max)
    print('heatmap_CD_min:', heatmap_CD_min)

    H_prime_mi = ((heatmap_MI - heatmap_MI_min) * 255.0 / 
                (heatmap_MI_max - heatmap_MI_min + 1e-8))  # +1e-8 чтобы избежать деления на 0

    H_prime_cd = ((heatmap_CD - heatmap_CD_min) * 255.0 / 
                (heatmap_CD_max - heatmap_CD_min + 1e-8))

    H_fuse_float = H_prime_mi * ratio_mi + H_prime_cd * (1 - ratio_mi)

    heatmap_MI = np.clip(H_prime_mi, 0, 255).astype(np.uint8)
    heatmap_CD = np.clip(H_prime_cd, 0, 255).astype(np.uint8)  
    heatmap_fusion = np.clip(H_fuse_float, 0, 255).astype(np.uint8)

    heatmap_fusion_flatnparray = np.array(heatmap_fusion, dtype=np.uint8)
    heatmap_fusion_grayscale = heatmap_fusion_flatnparray.reshape(orig_H, orig_W)

    heatmap_MI_flatnparray = np.array(heatmap_MI, dtype=np.uint8)
    heatmap_MI_grayscale = heatmap_MI_flatnparray.reshape(orig_H, orig_W)

    heatmap_CD_flatnparray = np.array(heatmap_CD, dtype=np.uint8)
    heatmap_CD_grayscale = heatmap_CD_flatnparray.reshape(orig_H, orig_W)

    return heatmap_MI_grayscale, heatmap_CD_grayscale, heatmap_fusion_grayscale

def heatmap_filter(heatmap, threshold, height, width):
    '''
    Адаптивный парог и морфологические операции
    
    :param heatmap: объединенная fusion heatmap
    :param threshold: порог
    :param height: высота изображения
    :param width: ширина изображения
    '''
    _, map_thresh = cv2.threshold(heatmap, threshold, maxval=255, type=cv2.THRESH_TOZERO)

    # расчет базового размера ядра для морф операций
    base_kernel_size = int(min(height, width) / kernel_param)
    print(base_kernel_size)

    # OPEN операция
    kernel = np.ones((base_kernel_size * 2,base_kernel_size * 2), np.uint8)
    # kernel=np.ones((base_kernel_size,base_kernel_size),np.uint8)
    map_thresh_OP = cv2.morphologyEx(map_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imwrite(savefig_path+name+"_t_open.png", crosion)

    # CLOSE операция
    kernel = np.ones((base_kernel_size, base_kernel_size), np.uint8)
    # kernel=np.ones((base_kernel_size*2,base_kernel_size*2),np.uint8)
    map_thresh_OP_CL = cv2.morphologyEx(map_thresh_OP, cv2.MORPH_CLOSE, kernel, iterations=2)

    # снова OPEN операция
    kernel = np.ones((base_kernel_size * 3,base_kernel_size * 3), np.uint8)
    map_thresh_OP_CL_OP = cv2.morphologyEx(map_thresh_OP_CL, cv2.MORPH_OPEN, kernel, iterations=2)

    return map_thresh, map_thresh_OP, map_thresh_OP_CL, map_thresh_OP_CL_OP