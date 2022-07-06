import numpy as np
import pv_vision.transform_crop.perspective_transform as transform
from skimage.morphology import skeletonize
from pathlib import Path
import os
import cv2 as cv
import json
from scipy import stats


def skeleton_crack(mask_crack):
    """Skeletonize crack masks
    Parameteres
    -----------
    mask_crack: array
    Mask of cracks

    Returns
    -------
    ske_crack: array
    Skeletonized crack mask
    """
    return skeletonize(mask_crack).astype(np.uint8)


def extend_busbar(mask_busbar, kernel_size=(10, 100)):
    """Connet and extend broken busbars. Return the skeleton of the busbar masks
    Parameters
    ----------
    mask_busbar: array
    Masks of busbars

    kernel_size: list or tuple
    Kernel used to do morphological operation. Details available on
    https://docs.opencv.org/4.5.4/d9/d61/tutorial_py_morphological_ops.html

    Returns
    -------
    ske_busbar: array
    Skeletonized busbar masks
    """
    kernel = np.ones(kernel_size, np.uint8)
    closing = cv.morphologyEx(mask_busbar, cv.MORPH_CLOSE, kernel)
    ske_busbar = skeletonize(closing).astype(np.uint8)

    return ske_busbar


def locate_busbar(ske_busbar):
    """Get position of busbars
    Parameters
    ----------
    ske_busbar: array
    Skeletonized busbar masks

    Returns
    -------
    pos_busbar: list
    Positions of busbars
    """
    numlist_busbar = []
    for i in np.linspace(10, ske_busbar.shape[-1] - 10, 10, dtype=int):
        numlist_busbar.append(len(np.argwhere(ske_busbar[:, i] == 1)))

    num_busbar = stats.mode(numlist_busbar).mode[0]

    pos_busbar = np.zeros((num_busbar, 1))
    for i in np.linspace(10, ske_busbar.shape[-1] - 10, 100, dtype=int):
        tem_pos = np.argwhere(ske_busbar[:, i] == 1)

        if len(tem_pos) == num_busbar:
            pos_busbar = np.hstack((pos_busbar, tem_pos))

    pos_busbar = np.delete(pos_busbar, 0, axis=1)
    pos_busbar = pos_busbar.mean(axis=1, dtype=int).tolist()

    return pos_busbar


def skeleton_cell(ske_crack, pos_busbar):
    """Get the skeleton of a cell. Crack has the value of -1, busbar is 1, other area is 0.
    Parameters
    ----------
    ske_crack: array
    Skeleton of crack masks

    pos_busbar: list
    Position of busbars

    Returns
    -------
    ske_cell: array
    Skeleton of cell
    """
    ske_cell = ske_crack * -1
    for i in pos_busbar:
        ske_cell[i, :] = 1

    return ske_cell


def stop_diff(val):
    """Check whether diffusion should stop. If meet busbar(val=1) or crack(val=-1), stop diffusion
    Parameters
    ----------
    val: int
    Grayscale value of a pixel

    Returns
    -------
    bool
    """
    return val == 1 or val == -1


def diff_up(image, row, col):
    """Diffuse the electrons up. The busbars are horizontally aligned
    Parameters
    ----------
    image: array
    Skeleton of cell

    row, col: int
    Position of current pixel
    """
    current = row - 1
    while not (current < 0 or stop_diff(image[current, col])):
        image[current, col] = 1
        current -= 1


def diff_down(image, row, col):
    """Diffuse the electrons down. The busbars are horizontally aligned
    Parameters
    ----------
    image: array
    Skeleton of cell

    row, col: int
    Position of current pixel
    """
    end = image.shape[0]
    current = row + 1
    while not (current > end - 1 or stop_diff(image[current, col])):
        image[current, col] = 1
        current += 1


def diffuse_line(image, row):
    """Diffuse the electrons from one busbar. The busbars are horizontally aligned
    Parameters
    ----------
    image: array
    Skeleton of cell

    row: int
    Position of current busbar
    """
    for j in range(image.shape[-1]):
        diff_up(image, row, j)
        diff_down(image, row, j)


def diffuse(image, pos_busbar):
    """Diffuse the electrons from all busbars. The busbars are horizontally aligned
    Parameters
    ----------
    image: array
    Skeleton of cell

    pos_busbar: list
    Position of busbars

    Returns
    -------
    image_c: array
    Diffused cell
    """
    image_c = np.copy(image)
    for i in pos_busbar:
        diffuse_line(image_c, i)
    return image_c


def count_area(cell_diff):
    """Count worst-case percentage of inactive area
    Parameters
    ----------
    cell_diff: array
    Diffused cell image

    Returns
    -------
    percentage of inactive area: float
    """
    inactive_area = np.zeros(cell_diff.shape).astype(np.uint8)
    inactive_area[cell_diff == 0] = 1
    return inactive_area.sum() / (inactive_area.shape[0] * inactive_area.shape[1])


def detect_inactive(mask_crack, mask_busbar, extend_kernel=(10, 100)):
    """Detect the worst-case isolated area and calculate its proportion
    Parameters
    ----------
    mask_crack: array
    Mask of cracks

    mask_busbar: array
    Masks of busbars

    extend_kernel: list or tuple
    Kernel used to do morphological operation to extend busbar. Details available on
    https://docs.opencv.org/4.5.4/d9/d61/tutorial_py_morphological_ops.html

    Returns
    -------
    inactive_area: array
    Binary isolated area

    inactive_prop: float
    percentage of inactive area. Not in the form of %
    """
    ske_crack = skeleton_crack(mask_crack)
    ske_busbar = extend_busbar(mask_busbar, kernel_size=extend_kernel)
    pos_busbar = locate_busbar(ske_busbar)
    ske_cell = skeleton_cell(ske_crack, pos_busbar)
    cell_diff = diffuse(ske_cell, pos_busbar)

    inactive_area = np.zeros(cell_diff.shape).astype(np.uint8)
    inactive_area[cell_diff == 0] = 1
    inactive_prop = inactive_area.sum() / (inactive_area.shape[0] * inactive_area.shape[1])

    return inactive_area, inactive_prop

