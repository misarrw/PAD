import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def recompress_diff(orig_img, checkDisplacements):
    '''
    Функция для сжатия изображения и нахождения разницы
    
    :param orig_img: изначальное изображение
    :param checkDisplacements: флаг какой-то
    '''
    minQ = 51
    maxQ = 100
    stepQ = 1

    if checkDisplacements == 1:
        max_disp = 7 # смещение от 0 до 7 пикселей
    else:
        max_disp = 0

    mins = []
    output = []
    height, width, _ = orig_img.shape
    smoothing_b = max(3, min(height, width) // 50) # размер ядра для сглаживания?
    offset = (smoothing_b - 1) // 2
    disp_imgs = []
    raw_deltas = []

    for i in range(minQ, maxQ + 1, stepQ):
        cv2.imwrite('temp_img.jpg', orig_img, [int(cv2.IMWRITE_JPEG_QUALITY), i]) # сохраняем изображение с качеством i
        temp_img = cv2.imread('temp_img.jpg').astype(float) # читаем его
        
        deltas = []
        overall_delta = []

        for displacement_x in range(max_disp + 1):
            for displacement_y in range(max_disp + 1):
                
                # disp_idx = displacement_x * (max_disp + 1) + displacement_y
                
                temp_img_disp = temp_img[displacement_x:, displacement_y:, :]
                
                orig_img_disp = orig_img[: height - displacement_x, : width - displacement_y, :].astype(float)

                squared_diff = np.square(orig_img_disp - temp_img_disp)
                squared_diff = np.mean(squared_diff, axis=2)

                h = np.ones((smoothing_b, smoothing_b)) / smoothing_b**2
                squared_diff = cv2.filter2D(squared_diff, -1, h) # сглаживающий фильтр

                squared_diff = squared_diff[offset : -offset, offset : -offset]
                
                # deltas.append(np.mean(squared_diff, axis=2))
                deltas.append(squared_diff)
                raw_deltas.append(squared_diff)
                
                overall_delta.append(squared_diff.mean())

        min_overall_delta, min_idx = min(overall_delta), np.argmin(overall_delta)
        mins.append(min_idx)
        output.append(min_overall_delta)
        delta = deltas[min_idx]
        raw_deltas.append(delta)
        # delta = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))

        resized_raw = cv2.resize(delta.astype(np.float32), 
                                (delta.shape[1] // 4, delta.shape[0] // 4),
                                interpolation=cv2.INTER_LINEAR)
        disp_imgs.append(resized_raw)
    
    if raw_deltas:
      normalized_imgs = []
      for raw_delta in raw_deltas:
        delta_nonzero = raw_delta[raw_delta > 0]
        if len(delta_nonzero) > 0:
            p99 = np.percentile(delta_nonzero, 99)
            p1 = np.percentile(delta_nonzero, 1)
            
            if p99 > p1:
                normalized = (raw_delta - p1) / (p99 - p1 + 1e-8)
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized = raw_delta
        else:
            normalized = raw_delta
        
        normalized = normalized ** 0.5
        normalized_imgs.append(normalized)
      disp_imgs = normalized_imgs

    output_Y = output
    output_X = list(range(minQ, maxQ + 1, stepQ))
    _, _, _, i_min = cv2.minMaxLoc(np.array(output_Y))
    i_min = sorted(i_min)
    qualities = [i * stepQ + minQ - 1 for i in i_min]

    return output_X, output_Y, disp_imgs, i_min, qualities, mins

def clean_up_image(filename):
    '''
    Функция для предобработки изображений перед PAD
    
    :param filename: путь к файлу
    '''
    
    img = cv2.imread(filename)

    if len(img.shape) > 3:
        # img = img[:, :, :, 0, 0, 0, 0]
        img = img[:, :, :, 0]

    dots = filename.rfind('.')
    extension = filename[dots:] # определяем расширение файла
    
    # обработка GIF
    if extension.lower() == '.gif' and img.shape[2] < 3:
        im_gif, _ = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        im_gif = im_gif[:, :, 0]
        img = np.uint8(cv2.cvtColor(im_gif, cv2.COLOR_GRAY2RGB) * 255)

    if img.shape[2] < 3:
        img[:, :, 1] = img[:, :, 0]
        img[:, :, 2] = img[:, :, 0]

    if img.shape[2] > 3:
        img = img[:, :, 0:3]

    if img.dtype == np.uint16:
        img = np.uint8(np.floor(img / 256))

    # im_out = img
    return img #im_out

def img_heatmap_cd(img_path):
    '''
    Построение тепловой карты CD
    
    :param img_path: путь к картинке
    '''

    img = clean_up_image(img_path)
    checkDisplacements = 0
    output_X, output_Y, disp_imgs, i_min, qualities, Mins = recompress_diff(img, checkDisplacements)
    heatmap_CD = disp_imgs

    return heatmap_CD, output_X