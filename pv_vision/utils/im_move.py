import csv
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def im_move(im_folder_path, csv_path, subfolders, parentfolder='classified_images', im_extension=None):
    """Move the source images into subfolders based on their categories.

    Parameters
    ----------
    im_folder_path: str or pathlib.PosixPath
        The folder path of the original images.

    csv_path: str or pathlib.PosixPath
        The path of the csv file that indicates the category of each solar module.
        The first column of the csv file should be the name of a module without filename extension
        The second column should be its category which is the same in the subfolders.

    subfolders: list of strings
        The categories of the solar modules.
        E.g. ['category1', 'category2', 'category3']

    parentfolder: str
        The parent folder name of subfolders

    im_extension: str
        The filename extension of the src images, e.g. '.png', '.jpg', etc.
    """
    if not im_extension:
        image = os.listdir(im_folder_path)[0]
        im_extension = os.path.splitext(image)[-1]

    folder = Path(parentfolder)
    for subfolder in subfolders:
        os.makedirs(folder/ subfolder, exist_ok=True)

    with open(csv_path, 'r') as file:
        data = [line.rstrip() for line in file]

    for cell in tqdm(data):
        name, label = cell.split(',')[0], cell.split(',')[1]
        shutil.copy(im_folder_path/ (name + im_extension), folder/label)
