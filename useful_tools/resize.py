import cv2 as cv
from imutils import paths
from pathlib import Path
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--image', type=str,
                    help='image path, path separated with /')

parser.add_argument('-s', '--size', nargs="+", type=int,
                    help='size list, width x height')
args = parser.parse_args()

arg_size = tuple(args.size)

if os.path.isfile(args.image):
    arg_image = Path(args.image)
    image = cv.imread(str(arg_image))
    image_r = cv.resize(image, arg_size)
    cv.imwrite(str(arg_image), image_r)
elif os.path.isdir(args.image):
    arg_folder = Path(args.image)
    im_paths = list(paths.list_images(str(arg_folder)))
    for im_path in tqdm(im_paths):
        image = cv.imread(im_path)
        image_r = cv.resize(image, arg_size)
        cv.imwrite(im_path, image_r)
