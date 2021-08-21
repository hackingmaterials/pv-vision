from pathlib import Path
from imutils import paths
from tqdm import tqdm
import os
import shutil


def subfolder_merge(parent_folder, output_folder):
    """Merge the subfolders into one folder.

    Parameters
    ----------
    parent_folder: str or pathlib.PosixPath
    The folder that contains the subfolders

    output_folder: str or pathlib.PosixPath
    The folder that collects files from subfolders
    """
    os.makedirs(output_folder, exist_ok=True)
    file_paths = list(paths.list_files(str(parent_folder)))

    for file in tqdm(file_paths):
        shutil.copy(file, output_folder)
    
    os.remove(Path(output_folder)/'.DS_Store')
