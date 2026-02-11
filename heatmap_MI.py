import math
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os

from sklearn.metrics.cluster import  mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score


def get_heatmap_mi(img_gray, img_shape_x, img_shape_y, window_size, window_stride):
    '''
    Улучшенная версия MI heatmap с минимальными изменениями
    '''
    
    heatmap_MI = np.zeros_like(img_gray, dtype=np.float32)
    
    stride_x, stride_y = int(window_stride), int(window_stride)
    window_shape_x, window_shape_y = int(window_size), int(window_size)
    
    pad_size = window_shape_x
    img_gray = cv2.copyMakeBorder(img_gray, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    
    mi_cache = {}
    
    for x in range(0, img_shape_x - window_shape_x, stride_x):
        for y in range(0, img_shape_y - window_shape_y, stride_y):
            # Текущее окно
            wx = x + pad_size
            wy = y + pad_size
            W_cur = img_gray[wx:wx + window_shape_x, wy:wy + window_shape_y].flatten()
            
            MI_sum = 0
            neighbor_count = 0
            
            # Соседние окна
            neighbours = [
                (wx - window_shape_x, wy, "left"),
                (wx + window_shape_x, wy, "right"),
                (wx, wy - window_shape_y, "up"),
                (wx, wy + window_shape_y, "down"),
                # (wx - window_shape_x, wy - window_shape_y, "up-left"),
                # (wx + window_shape_x, wy - window_shape_y, "up-right"),
                # (wx - window_shape_x, wy + window_shape_y, "down-left"),
                # (wx + window_shape_x, wy + window_shape_y, "down-right")
            ]
            
            for nx, ny, direction in neighbours:
                # проверка границ
                if (0 <= nx < img_shape_x + 2*pad_size - window_shape_x and 
                    0 <= ny < img_shape_y + 2*pad_size - window_shape_y):
                    
                    # Ключ для кэша
                    cache_key = f"{x},{y},{direction}"
                    
                    if cache_key in mi_cache:
                        mi_value = mi_cache[cache_key]
                    else:
                        W_neighbor = img_gray[nx:nx + window_shape_x, ny:ny + window_shape_y].flatten()
                        

                        if len(W_cur) > 10 and len(W_neighbor) > 10:
                            W_cur_quant = np.digitize(W_cur, np.linspace(0, 255, 17))
                            W_neighbor_quant = np.digitize(W_neighbor, np.linspace(0, 255, 17))
                            
                            mi_value = mutual_info_score(W_cur_quant, W_neighbor_quant)
                            
                            mi_value = mi_value / np.log(16)
                        else:
                            mi_value = 0
                        
                        mi_cache[cache_key] = mi_value
                    
                    neighbor_count += 1
                    MI_sum += mi_value
            
            if neighbor_count > 0:
                window_MI = MI_sum / neighbor_count

                end_x = min(x + window_shape_x, img_shape_x)
                end_y = min(y + window_shape_y, img_shape_y)
                heatmap_MI[x:end_x, y:end_y] = window_MI
    
    heatmap_MI = heatmap_MI[:img_shape_x, :img_shape_y]
    
    # сглаживание гаусса
    kernel_size = max(3, min(img_shape_x, img_shape_y) // 100)
    kernel_size = kernel_size + (kernel_size % 2 == 0)
    if kernel_size >= 3:
        heatmap_MI = cv2.GaussianBlur(heatmap_MI, (kernel_size, kernel_size), 0.5)
    
    mi_values = heatmap_MI.flatten()
    if len(mi_values) > 0:
        p5 = np.percentile(mi_values, 5)
        p95 = np.percentile(mi_values, 95)
        
        if p95 > p5:
            heatmap_MI = np.clip(heatmap_MI, p5, p95)
            heatmap_MI = (heatmap_MI - p5) / (p95 - p5)
        else:
            heatmap_MI = np.zeros_like(heatmap_MI)
    
    gamma = 0.7
    heatmap_MI = np.power(heatmap_MI, gamma)
    
    return heatmap_MI


def img_heatmap_mi(img_path): #img_path - путь к картинке
    '''
    ввод MI карты (переведено с английского)
    '''
    colorful_img = cv2.imread(img_path)
    grey_img = cv2.cvtColor(colorful_img,cv2.COLOR_BGR2GRAY) # перевод в чб (оттенки серого)
    grey_img = np.array(grey_img) 
    
    img_shape = grey_img.shape
    E = np.array(grey_img) # зачем-то здесь копия

    img_shape_x = img_shape[0]
    img_shape_y = img_shape[1]
        
    sx = np.ceil(img_shape_x / 100) + np.mod(np.ceil(img_shape_x / 100), 2)
    # img_shape / 100 - размер окна
    # плюс размер окна четный
    sy = np.ceil(img_shape_y / 100) + np.mod(np.ceil(img_shape_y / 100), 2)
        
    # window_size = [s2, s2 * 1.5 + np.mod(s2 * 1.5, 2), s2 * 2]  странные преобразования, надо оставить только s2

    window_size = max(sx, sy, 8) # размер окна не менее 8
    
    # window_stride = []
    # for a in window_size:
    #     window_stride.append(a/2)
    window_stride = window_size / 2 # шаг скользящего окна = 50% от размера окна
        
    # area = img_shape_x * img_shape_y
    heatmap_MI = get_heatmap_mi(grey_img, img_shape_x, img_shape_y, window_size, window_stride)
    # E = get_heatmap_mi(grey_img,img_shape_x,img_shape_y,40,window_stride[0],0)

    return heatmap_MI


if __name__ == "__main__":
    # чтение картинки (переведено с китайского)
    data_dir = ''
    data_files = os.listdir(data_dir)
    for data_file in data_files:
        img_name = data_file.split(".")[0]
        img_path = data_dir + data_file

        MI = img_heatmap_mi(img_path)
        # print(MI.shape)

        '''надо бы понять, зач это надо'''
        MI_max = np.max(MI)
        MI_min = np.min(MI)
        # print('max:', E_max)
        # print('min:', E_min)
        MI = (MI - MI_min) * 25 / (MI_max-MI_min)  # expand pixel to [0, 255]
        MI = MI.astype(int) # float --> int

        plt.subplot(1, 1, 1)
        plt.imshow(MI, cmap=plt.cm.jet)
        plt.xlabel('MI')
        plt.savefig("" + img_name + "mi.png")
        # plt.show()


















