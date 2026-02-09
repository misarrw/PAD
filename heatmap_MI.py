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
    Создание MI тепловой карты
    '''

    heatmap_MI = np.zeros_like(img_gray, dtype=np.float32)

    stride_x, stride_y = int(window_stride), int(window_stride)
    window_shape_x, window_shape_y = int(window_size), int(window_size)
    # ents = []  сохранение значений для отладки????????

    img_gray = cv2.copyMakeBorder(img_gray, int(window_shape_y / 2), int(window_shape_y / 2), int(window_shape_x / 2), int(window_shape_x / 2), cv2.BORDER_REFLECT)

    '''проход по картинке с шагом stride_x'''
    for x in range(0, img_shape_x - window_shape_x, stride_x):
        '''проход по картинке с шагом stride_y'''
        for y in range(0, img_shape_y - window_shape_y,stride_y):
            W_cur = img_gray[x:x + window_shape_x, y:y + window_shape_y].flatten() # вырезается W_cur и преобразуется в вектор

            MI_sum = 0
            neighbor_count = 0
            if (x - window_shape_x) >= 0: # W_left
                W_left = img_gray[x-window_shape_x:x,y:y+window_shape_y].flatten()
                MI_left = mutual_info_score(W_cur, W_left)
                # MI_left = normalized_mutual_info_score(W_cur, W_left)
                # MI_left = adjusted_mutual_info_score(W_cur, W_left)
                neighbor_count += 1
                MI_sum += MI_left

            if (y - window_shape_y) >= 0: # W_up
                W_up = img_gray[x:x+window_shape_x,y-window_shape_y:y].flatten()
                MI_up = mutual_info_score(W_cur, W_up)
                # MI_up = normalized_mutual_info_score(W_cur, W_up)
                # MI_up = adjusted_mutual_info_score(W_cur, W_up)
                neighbor_count += 1
                MI_sum += MI_up

            if (x + window_shape_x * 2) < img_shape_x: # W_right
                W_right = img_gray[x + window_shape_x:x + window_shape_x * 2,y:y + window_shape_y].flatten()
                MI_right = mutual_info_score(W_cur, W_right)
                # MI_right = normalized_mutual_info_score(W_cur, W_right)
                # MI_right = adjusted_mutual_info_score(W_cur, W_right)
                neighbor_count += 1
                MI_sum += MI_right

            if (y + window_shape_y * 2) < img_shape_y: # W_down
                W_cur_down = img_gray[x:x + window_shape_x, y + window_shape_y:y + window_shape_y * 2].flatten()
                mi_down = mutual_info_score(W_cur, W_cur_down)
                # mi_down = normalized_mutual_info_score(W_cur, W_cur_down)
                # mi_down = adjusted_mutual_info_score(W_cur, W_cur_down)
                neighbor_count += 1
                MI_sum += mi_down

            window_MI = MI_sum / neighbor_count

            heatmap_MI[x:x + window_shape_x, y:y + window_shape_y] = window_MI
            # ents.append(window_MI)
            # print(window_MI)

    x_exclude = heatmap_MI.shape[0] - img_shape_x
    y_exclude = heatmap_MI.shape[1] - img_shape_y
    heatmap_MI = heatmap_MI[round(x_exclude / 2):img_shape_x + round(x_exclude / 2) - 1, round(y_exclude / 2):img_shape_y + round(y_exclude / 2) - 1] # почему -1?
    # print(heatmap_MI.shape)

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


















