import csv
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('ann', type=str,
                    help='input ann folder, path separated with /')
args = parser.parse_args()

ann_dir = Path(args.ann)
ann_files = os.listdir(ann_dir)

for file in tqdm(ann_files):
    name = file.split('.')[0]
    
    with open(ann_dir/file, 'r') as f1:
        data = json.load(f1)
        
    crack = 0
    oxygen = 0
    intra = 0
    solder = 0

    for defect in data["objects"]:
        if defect["classTitle"] == "crack_bbox_yolo":
            crack += 1
        elif defect["classTitle"] == "oxygen_bbox_yolo":
            oxygen += 1
        elif defect["classTitle"] == "intra_bbox_yolo":
            intra += 1
        elif defect["classTitle"] == "solder_bbox_yolo":
            solder += 1
    

    if crack <= 1 and oxygen == 0 and solder == 0 and intra == 0:
        content = [name, 'category1', 'crack', str(crack),
                   'oxygen', str(oxygen), 'solder', str(solder),
                   'intra', str(intra), '\n']
        with open("cell_issues.csv", 'a') as f2:
            f2.write(','.join(content))
        
    elif (crack > 1 or oxygen > 0 or solder > 0) and intra == 0:
        content = [name, 'category2', 'crack', str(crack),
                   'oxygen', str(oxygen), 'solder', str(solder),
                   'intra', str(intra), '\n']
        with open("cell_issues.csv",'a') as f2:
            f2.write(','.join(content))

    elif intra > 0:
        content = [name, 'category3', 'crack', str(crack),
                   'oxygen', str(oxygen), 'solder', str(solder),
                   'intra', str(intra), '\n']
        with open("cell_issues.csv",'a') as f2:
            f2.write(','.join(content))