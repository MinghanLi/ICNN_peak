#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:09:41 2018

@author: vivi
"""

import cv2
import os
import csv
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from PIL import Image
import argparse
import torch.distributed as dist


save_txt_path = 'txt'
lung_img_path = '/data/volume1/'
lung_annot_path = os.path.join('/data/volume2', 'annotations.csv')
lung_jpg_path = '/data/oss_bucket/'
JPG = True


def locate_lung_lesion():
    parser = argparse.ArgumentParser(description='CapsNet')
    parser.add_argument('--dist_backend', default='mpi', type=str, help='distributed backend')
    args = parser.parse_args()
    # dist.init_process_group(backend=args.dist_backend)

    print("locating lung lesions...\n")
    cnt = 0
    subfold = [file for file in glob(lung_img_path+'/*') if os.path.isdir(file)]
    print('subfold:', subfold)
    annot = pd.read_csv(lung_annot_path)
    pos_file, neg_file = [], []
    
    for i in range(len(annot)):
        img_name = annot.seriesuid[i]
        print(i)
        find = False
        for fold in subfold:
            img_path = fold+'/'+img_name+'.mhd'
            # print('img_path', img_path)
            if os.path.exists(img_path):
                center = np.array([annot.coordX[i], annot.coordY[i], annot.coordZ[i]], dtype='float32')
                imgs = sitk.ReadImage(img_path)
                imgs_array = sitk.GetArrayFromImage(imgs)
                frame_num, width, height = imgs_array.shape
                origin = np.array(imgs.GetOrigin(), dtype='float32')
                spacing = np.array(imgs.GetSpacing(), dtype='float32')
                voxel_coord = np.rint((center-origin)/spacing)
                voxel_coord = abs(voxel_coord)
                voxel_coord = voxel_coord.astype('int32')
                centroid = voxel_coord[0:2]
                diameter = annot.diameter_mm[i]
                # diameter = math.ceil(annot.diameter_mm[i])
                radius = int(diameter//2.4)
                idx = voxel_coord[-1]  # '+1' for adjustment
                for j in range(radius+1):
                    # pos image
                    img_left_pos, img_right_pos = normlize(imgs_array[idx-j]), normlize(imgs_array[idx+j])
                    cv2.imwrite("%s/%s-%d-%s.jpg" % (lung_jpg_path, 'pos', cnt, img_name), img_left_pos)
                    cv2.imwrite("%s/%s-%d-%s.jpg" % (lung_jpg_path, 'pos', cnt+1, img_name), img_right_pos)
                    pos_file += ['pos'+'-'+str(cnt)+'-'+img_name, 1, centroid, diameter]
                    pos_file += ['pos'+'-'+str(cnt+1)+'-'+img_name, 1, centroid, diameter]

                    # neg image
                    if idx-radius-3-j < 0:
                        img_left_neg = normlize(imgs_array[0])
                    else:
                        img_left_neg = normlize(imgs_array[idx - radius - 3 - j])
                    if idx+radius+3+j >= frame_num-1:
                        img_right_neg = normlize(imgs_array[frame_num-1])
                    else:
                        img_right_neg = normlize(imgs_array[idx + radius + 3 + j])
                    cv2.imwrite("%s/%s-%d-%s.jpg" % (lung_jpg_path, 'neg', cnt, img_name), img_left_neg)
                    cv2.imwrite("%s/%s-%d-%s.jpg" % (lung_jpg_path, 'neg', cnt + 1, img_name), img_right_neg)
                    neg_file += ['neg'+'-'+str(cnt)+'-'+img_name, 0, [1000, 1000], 0]
                    neg_file += ['neg'+'-'+str(cnt + 1)+'-'+img_name, 0, [1000, 1000], 0]
                    cnt += 2
                find = True
                break
            else:
                continue
        if not find:
            with open('/data/oss_bucket/Luna16_error_JPG.csv', 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(img_name)

    with open('/data/oss_bucket/Luna16_anno_JPG.csv', 'wb') as f:
        writer = csv.writer(f)
        print(pos_file[0])
        # writer.writerow(['seriesuid', 'label', 'coord', 'diameter_mm'])
        for row in pos_file:
            writer.writerow([row])
        for row in neg_file:
            writer.writerow([row])


def normlize(img):
    img = (img-np.min(img))/float((np.max(img)-np.min(img)))
    return np.uint8(img * 255.)


def main():
    locate_lung_lesion()
    
    
if __name__ == '__main__':
    main()