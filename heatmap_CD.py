import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


'''пока не поняла, зачем это, но это зачем-то надо'''
savefig_path = ''

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
    smoothing_b = 17 # размер ядра для сглаживания?
    offset = (smoothing_b - 1) // 2
    height, width, _ = orig_img.shape
    disp_imgs = []

    for i in range(minQ, maxQ + 1, stepQ):
        cv2.imwrite('temp_img.jpg', orig_img, [int(cv2.IMWRITE_JPEG_QUALITY), i]) # сохраняем изображение с качеством i
        temp_img = cv2.imread('temp_img.jpg').astype(float) # читаем его
        
        deltas = []
        overall_delta = []

        for displacement_x in range(max_disp + 1):
            for displacement_y in range(max_disp + 1):
                
                disp_idx = displacement_x * 8 + displacement_y + 1
                
                temp_img_disp = temp_img[displacement_x:, displacement_y:, :]
                
                orig_img_disp = orig_img[: height - displacement_x, : width - displacement_y, :].astype(float)

                comparison = np.square(orig_img_disp - temp_img_disp)

                h = np.ones((smoothing_b, smoothing_b)) / smoothing_b**2
                comparison = cv2.filter2D(comparison, -1, h) # сглаживающий фильтр

                comparison = comparison[offset : -offset, offset : -offset, :]
                
                deltas.append(np.mean(comparison, axis=2))
                
                overall_delta.append(np.mean(deltas[disp_idx - 1]))

        min_overall_delta, min_idx = min(overall_delta), np.argmin(overall_delta)
        mins.append(min_idx)
        output.append(min_overall_delta)
        delta = deltas[min_idx]
        delta = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))

        disp_imgs.append(cv2.resize(delta.astype(np.float32), (delta.shape[1] // 4, delta.shape[0] // 4), interpolation=cv2.INTER_LINEAR))

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

if __name__ == "__main__":
    data_dir = '' # путь к директории с картинками
    data_files = os.listdir(data_dir)

    for data_file in data_files:
        filename = data_file.split(".")[0]
        img_path = data_dir + data_file

        heatmap_CD, output_X = img_heatmap_cd(img_path)

        # for i in range(len(heatmap_CD)):
        #     print(heatmap_CD[i].shape)
        #     print(output_X[i])
        #     plt.imshow(heatmap_CD[i])
        #     plt.title(output_X[i])
        #     plt.savefig(savefig_path+name+str(output_X[i])+".png")

        average_heatmap_CD = np.mean(heatmap_CD, axis=0)
        # plt.imshow(average_heatmap_CD)
        # plt.title('average')
        # plt.savefig(savefig_path+name+"_average.png")

        heatmap_CD_max = np.max(average_heatmap_CD)
        heatmap_CD_min = np.min(average_heatmap_CD)
        print('max:', heatmap_CD_max)
        print('min:', heatmap_CD_min)

        out_height = len(average_heatmap_CD)
        out_width = len(average_heatmap_CD[0])
        print("out_height , out_width", out_height, out_width)

        average_heatmap_CD = [int((average_heatmap_CD[i][j] - heatmap_CD_min) * 255 /(heatmap_CD_max-heatmap_CD_min)) for i in range(out_height) for j in range(out_width)]
        print(heatmap_CD)

        # translate into numpy array
        flatNumpyArray = np.array(average_heatmap_CD,dtype=np.uint8)
        # Convert the array to make a grayscale image
        grayImage = flatNumpyArray.reshape(out_height, out_width)
        # show gray image
        print(grayImage)

        #resize to original size
        img = cv2.imread(img_path)
        ori_height, ori_width, _ = img.shape
        print("ori_height , ori_width", ori_height, ori_width)
        grayImage = cv2.resize(grayImage, (ori_height, ori_width))

        cv2.imshow("GrayImage", grayImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()