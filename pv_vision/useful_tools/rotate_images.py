import numpy as np 
import cv2 as cv 
from imutils import paths 
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--image', type=str,
                    help='image, path separated with /')
parser.add_argument('-f', '--folder', type=str,
                    help='image folder, path separated with /')
parser.add_argument('-r', '--rotation', type=int,
                    help='rotation angles: 90-clockwise = 0, 180 = 1, 90-counter-clockwise = 2')
args = parser.parse_args()

arg_angle = args.rotation

if args.image:
    arg_image = Path(args.image)
    image = cv.imread(str(arg_image))
    image_r = cv.rotate(image, arg_angle)
    cv.imwrite(str(arg_image), image_r)
elif args.folder:
    arg_folder = Path(args.folder)
    im_paths = list(paths.list_images(str(arg_folder)))
    for im_path in tqdm(im_paths):
        image = cv.imread(im_path)
        image_r = cv.rotate(image, arg_angle)
        cv.imwrite(im_path, image_r)