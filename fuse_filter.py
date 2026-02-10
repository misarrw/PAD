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
input_path = ''
savefig_path = ''

def fuse_heatmap(img_path, orig_H, orig_W):
    '''
    Объединение тепловых карт MI и CD
    
    :param img_path: путь к картинке
    :param orig_H: исходная высота изображения
    :param orig_W: исходная ширина изображения
    '''

    # time_start = time.time()
    heatmap_MI = img_heatmap_mi(img_path)
    print('heatmap MI shape', heatmap_MI.shape)
    # time_mi_end = time.time()
    # print('--------------mi cost %f s' %(time_mi_end-time_start))

    heatmap_CD, _ = img_heatmap_cd(img_path)
    # heatmap_CD = np.mean(heatmap_CD, axis=0)
    middle_idx = len(heatmap_CD) // 2
    heatmap_CD = heatmap_CD[middle_idx]
    print('heatmap CD shape', heatmap_CD.shape)
    # time_cd_end = time.time()
    # print('--------------cd cost %f s' %(time_cd_end-time_mi_end))

    heatmap_MI = cv2.resize(heatmap_MI, (orig_W, orig_H))
    print('heatmap MI resize to original size')
    # plt.imshow(heatmap_MI)
    # plt.title('heatmap_MI_resize')
    # plt.show()
    heatmap_CD = cv2.resize(heatmap_CD, (orig_W, orig_H))
    print('heatmap CD resize to original size')
    # plt.imshow(heatmap_CD)
    # plt.title('heatmap_CD_resize')
    # plt.show()

    # этот кусок можно было бы перенести в отдельную функцию
    heatmap_MI_max = np.max(heatmap_MI)
    heatmap_MI_min = np.min(heatmap_MI)
    print('heatmap_MI_max:', heatmap_MI_max)
    print('heatmap_MI_min:', heatmap_MI_min)
    heatmap_CD_max = np.max(heatmap_CD)
    heatmap_CD_min = np.min(heatmap_CD)
    print('heatmap_CD_max:', heatmap_CD_max)
    print('heatmap_CD_min:', heatmap_CD_min)
    '''heatmap_MI = [int((heatmap_MI[i][j] - heatmap_MI_min) * 255 /(heatmap_MI_max - heatmap_MI_min)) for i in range(len(heatmap_MI)) for j in range(len(heatmap_MI[0]))] # неэффективно

    heatmap_CD = [int((heatmap_CD[i][j] - heatmap_CD_min) * 255 /(heatmap_CD_max - heatmap_CD_min)) for i in range(len(heatmap_CD)) for j in range(len(heatmap_CD[0]))] # неэффективно

    heatmap_fusion = [int(heatmap_MI[i] * ratio_mi + heatmap_CD[i] *(1 - ratio_mi)) for i in range(len(heatmap_MI))]
    print('length of the fusion heatmap:', len(heatmap_fusion))'''

    # 1. Нормализация MI (формула 8) - БЕЗ int()
    H_prime_mi = ((heatmap_MI - heatmap_MI_min) * 255.0 / 
                (heatmap_MI_max - heatmap_MI_min + 1e-8))  # +1e-8 чтобы избежать деления на 0

    # 2. Нормализация CD (формула 9) - БЕЗ int()  
    H_prime_cd = ((heatmap_CD - heatmap_CD_min) * 255.0 / 
                (heatmap_CD_max - heatmap_CD_min + 1e-8))

    # 3. Fusion (формула 10) - сначала float, потом uint8
    H_fuse_float = H_prime_mi * ratio_mi + H_prime_cd * (1 - ratio_mi)

    # 4. Только теперь преобразуем в uint8 (для совместимости с остальным кодом)
    heatmap_MI = np.clip(H_prime_mi, 0, 255).astype(np.uint8)
    heatmap_CD = np.clip(H_prime_cd, 0, 255).astype(np.uint8)  
    heatmap_fusion = np.clip(H_fuse_float, 0, 255).astype(np.uint8)

    # time_fuse_end = time.time()
    # print('--------------fuse cost %f s' %(time_fuse_end-time_cd_end))

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
    # НЕ АДАПТИВНЫЙ ПОРОГ
    _, map_thresh = cv2.threshold(heatmap, threshold, maxval=255, type=cv2.THRESH_TOZERO)
    # cv2.imshow('thresh',img)
    # cv2.waitKey(0) #0为任意键位终止
    # cv2.destroyAllWindows()
    # cv2.imwrite(savefig_path+name+"_t.png", img)

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
    # cv2.imwrite(savefig_path+name+"_t_open_close.png", crosion2)

    # снова OPEN операция
    kernel = np.ones((base_kernel_size * 3,base_kernel_size * 3), np.uint8)
    map_thresh_OP_CL_OP = cv2.morphologyEx(map_thresh_OP_CL, cv2.MORPH_OPEN, kernel, iterations=2)
    # cv2.imwrite(savefig_path+name+"_t_open_close_open.png", crosion3)

    return map_thresh, map_thresh_OP, map_thresh_OP_CL, map_thresh_OP_CL_OP


if __name__ == "__main__":
    # 读图
    data_dir = input_path
    #data_dir = "proper_patched"
    data_files = os.listdir(data_dir)
    for data_file in data_files:
        print(data_file)
        name = data_file.split(".")[0]
        img_path = data_dir + data_file
        
        ori_img = Image.open(img_path).convert('RGB')
        orig_W, orig_H = ori_img.size
        print("orig_H , orig_W", orig_H, orig_W)

        mi_img, cd_img, fuse_img = fuse_heatmap(img_path, orig_H, orig_W)
        # cv2.imshow("fuse_img", fuse_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if not os.path.exists(savefig_path):
            os.makedirs(savefig_path)

        threshold = np.percentile(fuse_img, thresh_param)
        map_thresh, map_thresh_OP, map_thresh_OP_CL, map_thresh_OP_CL_OP = heatmap_filter(fuse_img, threshold, orig_H, orig_W)

        # plt.figure()
        # plt.subplot(241)
        # plt.imshow(ori_img)
        # plt.title('ori_img')
        # plt.subplot(242)
        # plt.imshow(mi_img, cmap=plt.cm.jet)
        # plt.title('mi_heatmap')
        # plt.subplot(243)
        # plt.imshow(cd_img, cmap=plt.cm.jet)
        # plt.title('cd_heatmap')
        # plt.subplot(244)
        # plt.imshow(fuse_img, cmap=plt.cm.jet)
        # plt.title('fuse_heatmap')
        # plt.subplot(245)
        # plt.imshow(map_thresh)
        # plt.title('map_thresh')
        # plt.subplot(246)
        # plt.imshow(map_thresh_OP)
        # plt.title('map_thresh_OP')
        # plt.subplot(247)
        # plt.imshow(map_thresh_OP_CL)
        # plt.title('map_thresh_OP_CL')
        # plt.subplot(248)
        # plt.imshow(map_thresh_OP_CL_OP)
        # plt.title('map_thresh_OP_CL_OP')
        # # plt.show()
        # plt.savefig(savefig_path+name+".png")

        plt.imshow(ori_img)
        plt.title('ori_img')
        plt.savefig(savefig_path+name+"ori_img.png")
        plt.imshow(mi_img, cmap=plt.cm.jet)
        # plt.imshow(mi_img)
        plt.title('mi_heatmap')
        plt.savefig(savefig_path+name+"mi_heatmap.png")
        plt.imshow(cd_img, cmap=plt.cm.jet)
        plt.title('cd_heatmap')
        plt.savefig(savefig_path+name+"cd_heatmap.png")
        plt.imshow(fuse_img, cmap=plt.cm.jet)
        plt.title('fuse_heatmap')
        plt.savefig(savefig_path+name+"fuse_heatmap.png")
        plt.imshow(map_thresh, cmap=plt.cm.jet)
        plt.title('map_thresh')
        plt.savefig(savefig_path+name+"map_thresh.png")
        plt.imshow(map_thresh_OP, cmap=plt.cm.jet)
        plt.title('map_thresh_OP')
        plt.savefig(savefig_path+name+"map_thresh_OP.png")
        plt.imshow(map_thresh_OP_CL, cmap=plt.cm.jet)
        plt.title('map_thresh_OP_CL')
        plt.savefig(savefig_path+name+"map_thresh_OP_CL.png")
        plt.imshow(map_thresh_OP_CL_OP, cmap=plt.cm.jet)
        plt.title('map_thresh_OP_CL_OP')
        plt.savefig(savefig_path+name+"map_thresh_OP_CL_OP.png")

