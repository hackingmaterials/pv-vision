import cv2 as cv
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse
import pv_vision.transform_crop.perspective_transform as transform

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--image', type=str,
                    help='image or image folder, path separated with /')
parser.add_argument('-a', '--ann', type=str,
                    help='ann or ann folder, path separated with /')
parser.add_argument('-o', '--output', type=str,
                    help='the name or dir of output')
parser.add_argument('--width', type=int, default=16,
                    help='the number of cells on long edge')
parser.add_argument('--height', type=int, default=8,
                    help='the number of cells on short edge')
parser.add_argument('--mask_name', type=str, default='module_unet',
                    help='the name of the mask')
parser.add_argument('--method', type=int, choices=[0, 1], default=0,
                    help='0-find corners on the convex/contour, 1-find corners on original mask')
parser.add_argument('-m', '--mode', type=int, choices=[0, 1, 2, 3], default=0,
                    help='transform mode(default method is 0). 0-convex detection or corner_detection if method=1,\
                    1-convex approximate detection or line detection if method=1, 2-contour approximate detection, 3-blur')
args = parser.parse_args()

arg_im = Path(args.image)
arg_ann = Path(args.ann)
arg_mask_name = args.mask_name

if os.path.isfile(arg_im):
    file_name = os.path.split(arg_ann)[-1]
    name = file_name.split('.')[0]
    image = cv.imread(str(arg_im), cv.IMREAD_UNCHANGED)
    mask, mask_center = transform.load_mask(arg_ann, image, arg_mask_name)
    if args.method == 0:
        corners = transform.find_module_corner2(mask, mode=args.method)
    elif args.method == 1:
        corners = transform.find_module_corner(mask, mask_center, method=args.method,
                                     displace=3, corner_center=True,
                                     center_displace=70)
    wrap = transform.perspective_transform(image, corners, int(37.5*args.width), int(37.5*args.height))
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

    if args.output:
        out_dir = Path(args.output)

    else:
        out_dir = Path('.')
    store_dir = out_dir / 'transformed_images'
    err_dir = out_dir / 'failed_images'
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)

    N = 0
    N_err = 0
    for ann_file in tqdm(ann_files):
        N += 1
        try:
            name = ann_file.split('.')[0]
            ann_path = arg_ann/ann_file
            image = cv.imread(str(arg_im/(name+im_type)), cv.IMREAD_UNCHANGED)

            mask, mask_center = transform.load_mask(ann_path, image, arg_mask_name)

            flag = True
            try:
                for i in [0, 1]:
                    corners = transform.find_module_corner(mask, mask_center, method=i, displace=3)
                    wrap = transform.perspective_transform(image, corners, int(37.5*args.width), int(37.5*args.height))
                    peak_x, peak_y = transform.find_inner_edge(wrap)
                    if len(peak_x) > (args.width-4) and len(peak_y) > (args.height-3):
                        cv.imwrite(str(store_dir/(name+'.png')), wrap)
                        flag = False
                        break
            except:
                pass

            if flag:
                try:
                    for i in [0, 1]:
                        corners = transform.find_module_corner(mask, mask_center, method=i,
                                                            displace=3, corner_center=True, center_displace=50)
                        wrap = transform.perspective_transform(image, corners, int(37.5*args.width), int(37.5*args.height))
                        peak_x, peak_y = transform.find_inner_edge(wrap)
                        if len(peak_x) > (args.width-4) and len(peak_y) > (args.height-3):
                            cv.imwrite(str(store_dir/(name+'.png')), wrap)
                            flag = False
                            break
                except:
                    pass
            
            if flag:
                try:
                    for i in [0, 1, 2, 3]:
                        corners = transform.find_module_corner2(mask, mode=i)
                        wrap = transform.perspective_transform(image, corners, int(37.5*args.width), int(37.5*args.height))
                        peak_x, peak_y = transform.find_inner_edge(wrap)
                        if len(peak_x) > (args.width-4) and len(peak_y) > (args.height-3):
                            cv.imwrite(str(store_dir/(name+'.png')), wrap)
                            flag = False
                            break
                except:
                    pass                
                
            
            if flag:
                N_err += 1
                with open(out_dir/"error.csv", 'a') as f:
                    f.write(name+","+"wrong_peaks\n")
                shutil.copyfile(arg_im/(name+im_type), err_dir/(name+'.png'))
        except:
            N_err += 1
            with open(out_dir/"error.csv", 'a') as f:
                f.write(name+","+"other_errs\n")
            shutil.copyfile(arg_im/(name+im_type), err_dir/(name+'.png'))
    
    print("total images: " + str(N))
    print("failed images:" + str(N_err))
    print("accuracy: " + str(1 - N_err / N))
