import numpy as np 
import cv2 as cv 
import json
import os
from tqdm import tqdm
from pathlib import Path
import argparse


def draw_rec(ann_path, image, color, thickness=3):
    with open(ann_path, 'r') as file:
        data = json.load(file)
    if len(data['objects']) > 0:
        for obj in data['objects']:
            classTitle = obj['classTitle']
            points = obj['points']['exterior']

            cv.rectangle(image, tuple(points[0]), tuple(points[1]),
                         color[classTitle], thickness)


parser = argparse.ArgumentParser()

parser.add_argument('-i', '--image', type=str,
                    help='image or image folder, path separated with /')
parser.add_argument('-a', '--ann', type=str,
                    help='ann or ann folder, path separated with /')
parser.add_argument('-o', '--output', type=str,
                    help='the name or dir of output')
parser.add_argument('-c', '--color', type=str,
                    help='json file which contains the color of defects')
args = parser.parse_args()

arg_im = Path(args.image)
arg_ann = Path(args.ann)
with open(args.color, 'r') as f:
    arg_color = json.load(f)

if os.path.isfile(arg_im):
    file_name = os.path.split(arg_ann)[-1]
    name = file_name.split('.')[0]
    image = cv.imread(str(arg_im))

    draw_rec(arg_ann, image, arg_color, 3)

    if args.output:
        cv.imwrite(str(args.output), image)
    else:
        cv.imwrite(str(name+'_visualized.png'), image)
elif os.path.isdir(arg_im):
    im_files = os.listdir(arg_im)
    if '.DS_Store' in im_files:
        im_files.remove('.DS_Store')
    im_type = os.path.splitext(im_files[0])[-1]
    ann_files = os.listdir(arg_ann)
    if '.DS_Store' in ann_files:
        ann_files.remove('.DS_Store')

    if args.output:
        store_dir = Path(args.output)
    else:
        store_dir = Path('visualized_images')
    os.makedirs(store_dir, exist_ok=True)
    
    for ann_file in tqdm(ann_files):
        name = ann_file.split('.')[0]
        ann_path = arg_ann/ann_file
        image = cv.imread(str(arg_im/(name+im_type)))

        draw_rec(ann_path, image, arg_color, 3)

        cv.imwrite(str(store_dir/(name+'.png')), image)
