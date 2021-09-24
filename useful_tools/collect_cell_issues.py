import csv
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--ann', type=str,
                    help='input ann folder, path separated with /')
parser.add_argument('-n', '--name', type=str, default=None,
                    help='json file which contains the name of defects')
parser.add_argument('-o', '--output', type=str, default='.',
                    help='the name or dir of output')
args = parser.parse_args()

ann_dir = Path(args.ann)
ann_files = os.listdir(ann_dir)
store_dir = Path(args.output)
os.makedirs(store_dir, exist_ok=True)

if args.name:
    with open(args.name, 'r') as f:
        defect_name = json.load(f)
else:
    defect_name = {
        "crack": "crack_bbox_yolo",
        "oxygen": "oxygen_bbox_yolo",
        "intra": "intra_bbox_yolo",
        "solder": "solder_bbox_yolo"
    }

for file in tqdm(ann_files):
    name = file.split('.')[0]
    
    with open(ann_dir/file, 'r') as f1:
        data = json.load(f1)
        
    crack = 0
    oxygen = 0
    intra = 0
    solder = 0

    for defect in data["objects"]:
        if defect["classTitle"] == defect_name['crack']:
            crack += 1
        elif defect["classTitle"] == defect_name['oxygen']:
            oxygen += 1
        elif defect["classTitle"] == defect_name['intra']:
            intra += 1
        elif defect["classTitle"] == defect_name['solder']:
            solder += 1
    

    if crack <= 1 and oxygen == 0 and solder == 0 and intra == 0:
        content = [name, 'category1', 'crack', str(crack),
                   'oxygen', str(oxygen), 'solder', str(solder),
                   'intra', str(intra), '\n']
        with open(store_dir/"cell_issues.csv", 'a') as f2:
            f2.write(','.join(content))
        
    elif (crack > 1 or oxygen > 0 or solder > 0 or intra > 0) and intra < 2:
        content = [name, 'category2', 'crack', str(crack),
                   'oxygen', str(oxygen), 'solder', str(solder),
                   'intra', str(intra), '\n']
        with open(store_dir/"cell_issues.csv",'a') as f2:
            f2.write(','.join(content))

    elif intra > 1:
        content = [name, 'category3', 'crack', str(crack),
                   'oxygen', str(oxygen), 'solder', str(solder),
                   'intra', str(intra), '\n']
        with open(store_dir/"cell_issues.csv",'a') as f2:
            f2.write(','.join(content))