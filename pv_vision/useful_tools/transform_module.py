import numpy as np
import cv2 as cv
import zlib, base64
from scipy import signal
import os
import shutil
import json
from pathlib import Path
from imutils import paths
from tqdm import tqdm
import argparse
import seg_cnn as seg

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--image', type=str,
                    help='image or image folder, path separated with /')
parser.add_argument('-a', '--ann', type=str,
                    help='ann or ann folder, path separated with /')
parser.add_argument('-o', '--output', type=str,
                    help='the name or dir of output')
parser.add_argument('-n', '--mask_name', type=str, default='module_unet',
                    help='the name of the mask')
parser.add_argument('-m', '--method', type=int, choices=[0, 1], default=0,
                    help='transform method. 0-corner_detection,\
                    1-line_detection')
args = parser.parse_args()

arg_im = Path(args.image)
arg_ann = Path(args.ann)
arg_mask_name = args.mask_name

if os.path.isfile(arg_im):
    file_name = os.path.split(arg_ann)[-1]
    name = file_name.split('.')[0]
    image = cv.imread(str(arg_im))
    mask, mask_center = seg.load_mask(arg_ann, image, arg_mask_name)
    corners = seg.find_module_corner(mask, mask_center, method=args.method,
                                     displace=3, corner_center=True,
                                     center_displace=70)
    wrap = seg.perspective_transform(image, corners, 600, 300)
    if args.output:
        cv.imwrite(str(args.output), wrap)
    else:
        cv.imwrite(str(name+'transformed.png'), wrap)
elif os.path.isdir(arg_im):
    im_files = os.listdir(arg_im)
    if '.DS_Store' in im_files:
        im_files.remove('.DS_Store')
    im_type = os.path.splitext(im_files[0])[-1]
    ann_files = os.listdir(arg_ann)
    if '.DS_Store' in ann_files:
        ann_files.remove('.DS_Store')
    #im_paths = list(paths.list_images(str(arg_im)))
    #ann_paths = list(paths.list_files(str(arg_ann)))

    if args.output:
        store_dir = Path(args.output)
    else:
        store_dir = Path('transformed_images')
    err_dir = Path('failed_images')
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)

    N = 0
    N_err = 0
    for ann_file in tqdm(ann_files):
        N += 1
        try:
            name = ann_file.split('.')[0]
            ann_path = arg_ann/ann_file
            image = cv.imread(str(arg_im/(name+im_type)))

            mask, mask_center = seg.load_mask(ann_path, image, arg_mask_name)
            corners = seg.find_module_corner(mask, mask_center, method=0, displace=3)
            wrap = seg.perspective_transform(image, corners, 600, 300)
            peak_x, peak_y = seg.find_cell_corner(wrap)
            if len(peak_x) > 12 and len(peak_y) > 5:
                cv.imwrite(str(store_dir/(name+'.png')), wrap)
            else:
                corners = seg.find_module_corner(mask, mask_center, method=1, displace=3)
                wrap = seg.perspective_transform(image, corners, 600, 300)
                peak_x, peak_y = seg.find_cell_corner(wrap)
                if len(peak_x) > 12 and len(peak_y) > 5:
                    cv.imwrite(str(store_dir/(name+'.png')), wrap)
                else:
                    corners = seg.find_module_corner(mask, mask_center, method=0,
                                                     displace=3, corner_center=True, center_displace=50)
                    wrap = seg.perspective_transform(image, corners, 600, 300)
                    peak_x, peak_y = seg.find_cell_corner(wrap)
                    if len(peak_x) > 12 and len(peak_y) > 5:
                        cv.imwrite(str(store_dir/(name+'.png')), wrap)
                    else:
                        corners = seg.find_module_corner(mask, mask_center, method=1,
                                                         displace=3, corner_center=True, center_displace=50)
                        wrap = seg.perspective_transform(image, corners, 600, 300)
                        peak_x, peak_y = seg.find_cell_corner(wrap)
                        if len(peak_x) > 12 and len(peak_y) > 5:
                            cv.imwrite(str(store_dir/(name+'.png')), wrap)
                        else:
                            N_err += 1
                            with open("error.csv", 'a') as f:
                                f.write(name+'.png\n')
                            shutil.copyfile(arg_im/(name+im_type), err_dir/(name+'.png'))
        except:
            N_err += 1
            with open("error.csv", 'a') as f:
                f.write(name+'.png\n')
            shutil.copyfile(arg_im/(name+im_type), err_dir/(name+'.png'))
    
    print("total images: " + str(N))
    print("failed images:" + str(N_err))
    print("accuracy: " + str(1-N_err/N))
