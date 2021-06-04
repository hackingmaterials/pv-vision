from pathlib import Path
from imutils import paths
from tqdm import tqdm
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str,
                    help='subfodlers of which to be merged')
parser.add_argument('-o', '--output', type=str,
                    help='the name of output folder', default='merged')

args = parser.parse_args()

arg_in = Path(args.folder)
arg_out = Path(args.output)

os.makedirs(arg_out, exist_ok=True)
file_paths = list(paths.list_files(str(arg_in)))

for file_path in tqdm(file_paths):
    shutil.copy(file_path, arg_out)

os.remove(arg_out/'.DS_Store')