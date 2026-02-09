import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import glob
import os
from segment_anything.segment_anything.build_sam import sam_model_registry
# from segment_anything.segment_anything.predictor import SamPredictor
from segment_anything.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

from PIL import Image

from heatmap_MI import img_heatmap_mi
from heatmap_CD import img_heatmap_cd
from fuse_filter import fuse_heatmap, heatmap_filter

IOU_thresh = 0.5
ratio_MI = 0.5 # ratio_cd = 1-ratio_MI
kernel_param = 80
thresh_param = 80 # percentile, from small to big
input_path = ''
save_path = ''

def get_mask(image, mask_generator):
    masks = mask_generator.generate(image.astype(np.uint8))
    return masks

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry["vit_l"](checkpoint='') # указать путь к модели сегментации
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    print(f'Defended pictures will be saved here: {save_path}')
    folder = os.path.exists(save_path)
    if not folder:
        os.makedirs(save_path)

    with torch.no_grad():
        data_dir = input_path
        print(f'Your patched pictures directory: {data_dir}')
        data_files = os.listdir(data_dir)
        print('Pictures defending process has just started.')
        for data_file in data_files:
            print(f'Currently working with picture {data_file}')
            img_name = data_file.split(".")[0]
            img_path = data_dir + data_file
            
            original_img = Image.open(img_path).convert('RGB')
            orig_W, orig_H = original_img.size
            print(f'Original picture width: {orig_W}')
            print(f'Original picture height: {orig_H}')

            MI_img, CD_img, fuse_img = fuse_heatmap(img_path, orig_H, orig_W) # генерация MI, CD, fuse

            threshold = np.percentile(fuse_img, thresh_param) # какое-то определение порога
            h_t, h_t_o, h_t_o_c, h_t_o_c_o = heatmap_filter(fuse_img, threshold, orig_H, orig_W)

            gray = np.where(h_t_o_c_o > 0,1,0)

            rgb_color = cv2.imread(img_path)

            image = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2RGB)
            
            #just for Dpatch
            #image = cv2.resize(image,(416,416))

            h = image.shape[0]
            w = image.shape[1]
            mask = get_mask(image, mask_generator)
            print(len(mask))

            result_mask = np.zeros((h,w))
            for k in range(len(mask)):

                mask_k = mask[k].get('segmentation')
                n = mask_k&gray
                u = mask_k #|gray
                iou = np.sum(n)/(np.sum(u))
                print("iou",iou)

                n_1 = mask_k&result_mask.astype(np.uint8)
                u_1 = mask_k
                iou1 =  np.sum(n_1)/(np.sum(u_1))
                print("iou1",iou1)

                if(iou>IOU_thresh and iou1<0.1):
                    mask_k_save = np.expand_dims(mask_k,axis=2)
                    mask_k_save = np.tile(mask_k_save,3)
                    rgb_color = rgb_color*(~mask_k_save) 
                    result_mask = result_mask.astype(np.uint8) | mask_k
                    '''mask_k_save = np.expand_dims(mask_k,axis=2)
                    mask_k_save = np.tile(mask_k_save,3)
                    mask_gray = np.expand_dims(mask_k*128,axis=2)
                    mask_gray = np.tile(mask_gray,3)
                    rgb_color = rgb_color*(~mask_k_save) + mask_gray
                    result_mask = result_mask.astype(np.uint8) | mask_k'''
                    '''result_mask = result_mask.astype(np.uint8) | mask_k
                    rgb_color = cv2.inpaint(rgb_color, mask_k.astype(np.uint8), 3, cv2.INPAINT_NS)'''

            cv2.imwrite(save_path+img_name+".png",rgb_color)
            #cv2.imwrite("./mask_hxx_0.05_gray/"+name+".png",cv2.inpaint(rgb_color, cv2.blur(result_mask.astype(np.uint8),(5,5)), 3, cv2.INPAINT_NS))
