import csv
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--csv', type=str,
                    help='input csv file, path separated with /')
parser.add_argument('-i', '--image', type=str,
                    help='image folder, path separated with /')
parser.add_argument('-o', '--output', type=str, default='.',
                    help='the name or dir of output')
args = parser.parse_args()

args_csv = Path(args.csv)
arg_image = Path(args.image)

folder = Path(args.output) / 'classified_images'
for subfolder in ['category1', 'category2', 'category3']:
    os.makedirs(folder/subfolder, exist_ok=True)


with open(args_csv, 'r') as file:
    data = [line.rstrip() for line in file]

for cell in tqdm(data):
    name, label = cell.split(',')[0], cell.split(',')[1]
    shutil.copyfile(arg_image/(name+'.png'), folder/label/(name + '.png'))
