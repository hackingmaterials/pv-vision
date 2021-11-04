import numpy as np
import cv2 as cv
import json
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit



def linear(x, a, b):
    """Linear function
    Parameters
    ----------
    x, a, b: float
    linear function f = ax + b

    Returns
    -------
    f: float
    linear function f = ax + b
    """
    return a * x + b


def linear_regression(inxes, lines):
    """Linear regression: f = ax + b. Return multiple fitted lines. After detect_edge, do linear fitting to
    return the final parameters(a and b) of edges.

    Parameters
    ----------
    inxes: list, or array
    Values of x

    lines: list, or array
    Each line in lines represents values of f.

    Returns
    -------
    ab_s: list
    List of a and b of multiple lines
    """
    ab_s = []

    for line in lines:
        ab, _ = curve_fit(linear, inxes, line)
        # line_fit = ab[0]*inxes+ab[1]
        ab_s.append(ab)

    ab_s = np.array(ab_s)

    return ab_s


def filter_margin(edges, im_length, margin=20):
    """Filter the wrongly detected edges on the margin of the image

    Parameters
    ----------
    edges: list or array
    Detected positions of edges

    im_length: int
    Length of width or height of the image. If detect vertical edges, use width.
    If detect horizontal edges, use height

    margin: int
    Margin in which there shouldn't be edges but may be wrongly detected. Default is 20 pixels

    Returns
    -------
    edges: list or array
    Filtered edges
    """
    if edges[0] < margin:
        edges = np.delete(edges, 0)
    if edges[-1] > (im_length - margin):
        edges = np.delete(edges, -1)

    return edges


def detect_edge(image, peaks_on=0, cell_size=30, split_size=10, im_size=[300, 600], row_col=[8, 16], margin=20):
    """Detect the inner edges of a solar module. Split the solar module first, then detect position of
    edges in each split.

    Parameters
    ----------
    image: array
    Perspective transformed image of solar module. 
    Input should be grayscale or BGR read from Opencv

    peaks_on: int
    Detect horizontal edges or vertical edges
    If 0: detect vertical edges
    If 1: detect horizontal edges

    cell_size: int
    Estimated size of single cells in pixels. Default is 30.

    split_size: int
    The size of each split

    im_size: list
    Shape of image. [height, width]

    row_col: list
    Number of rows/columns of solar module

    margin: int
    Margin in which there shouldn't be edges but may be wrongly detected. Default is 20 pixels

    Returns
    -------
    split_inx: array
    Index of each split

    edge_list: array
    Position of edges of each split.
    Form is [[Positions of first edge in each split], [Positions of second edges in each split], ...]
    """
    if len(image.shape) == 2:
        image_g = image
    elif len(image.shape) == 3:
        image_g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if peaks_on == 0:
        splits = np.vsplit(image_g, image_g.shape[0] / split_size)

    elif peaks_on == 1:
        splits = np.hsplit(image_g, image_g.shape[1] / split_size)

    else:
        print('peaks_on must be 0 or 1')
        return

    peaklist = []
    splits_inx = []

    # peaks when suming the whole image
    sum_whole = np.sum(image_g, axis=peaks_on)
    sum_whole = sum_whole / sum_whole.max()
    peaks_whole, _ = find_peaks(-sum_whole, distance=cell_size)
    peaks_whole = filter_margin(peaks_whole, im_length=im_size[peaks_on - 1], margin=margin)

    flag = False
    for inx, split in enumerate(splits):

        sum_split = np.sum(split, axis=peaks_on)
        sum_split = sum_split / sum_split.max()
        peaks_split, _ = find_peaks(-sum_split, distance=cell_size)
        peaks_split = filter_margin(peaks_split, im_length=im_size[peaks_on - 1], margin=margin)

        splits_inx.append(int(split_size * (inx + 1 / 2)))
        if len(peaks_split) == row_col[peaks_on - 1] - 1:
            peaklist.append(peaks_split)
        elif (len(peaks_split) != row_col[peaks_on - 1] - 1) and inx != 0:
            peaklist.append(peaklist[inx - 1])
        elif len(peaks_split) == row_col[peaks_on - 1] - 1:
            peaklist.append(peaks_whole)
        else:
            peaklist.append('nan')
            flag = True

    # use any other splits to represent split[0, 1, 2...] if peaklist[0] fails
    if flag:
        peaklist = np.array(peaklist)
        nan = np.argwhere(peaklist == 'nan')
        n_nan = np.argwhere(peaklist != 'nan')
        for i in nan:
            peaklist[int(i)] = peaklist[int(n_nan[0])]
        peaklist = list(peaklist)

    edgelist = np.array(list(zip(*peaklist)))
    edgelist_c = np.copy(edgelist)

    if len(peaks_whole) == row_col[peaks_on - 1] - 1:
        for i, edge in enumerate(edgelist_c):
            for j, sub_edge in enumerate(edge):
                if np.abs(sub_edge - peaks_whole[i]) > 10:
                    edgelist_c[i][j] = peaks_whole[i]

    return np.array(splits_inx), edgelist_c


def displace(line_ab, displacement):
    """Displace fitted edges to increase tolerance

    Parameters
    ----------
    line_ab: list
    [a, b] of fitted edges

    displacement: int
    Perpendicular distance of moving the lines

    Returns
    (a, b): tuple
    a, b of displaced line
    """
    c = displacement * np.sqrt((1 + line_ab[0] ** 2))

    return (line_ab[0], line_ab[1] + c)


def couple_edges(lines_ab, length, displacement=0, add_edge=True):
    """Couple two neighbour edges

    Parameters
    ----------
    lines_ab: list or array
    [a, b] of fitted edges

    length: int
    Width or height of the image.
    If couple vertical edges, use width.
    If couple horizontal edges, use height

    displacement: int
    Displacement of edges to increase tolerance
    The first line in the couple moves to negative direction, the second to positive direction

    add_edge: bool
    If true, add the initial and tail edges, which are the two ends of images

    Returns
    -------
    lines_couple: array
    Coupled edges
    """
    #
    lines_copy = np.copy(lines_ab)
    if add_edge:
        lines_copy = np.insert(lines_copy, 0, [0, 0], axis=0)
        lines_copy = np.insert(lines_copy, len(lines_copy), [0, length - 1], axis=0)
    lines_l = np.delete(lines_copy, -1, 0)
    lines_r = np.delete(lines_copy, 0, 0)
    lines_couple = np.array(list(zip(lines_l, lines_r)))

    lines_couple_new = []
    for couple in lines_couple:
        minus = displace(couple[0], -displacement)
        plus = displace(couple[1], displacement)
        lines_couple_new.append((minus, plus))

    return np.array(lines_couple_new)


def cross(vline, hline):
    """Intersction of edges

    Parameters
    ----------
    vline, hline: list or tuple
    (a, b) of vertical edge, horizontal edge
    """
    x = (vline[0] * hline[1] + vline[1]) / (1 - vline[0] * hline[0])
    y = (hline[0] * vline[1] + hline[1]) / (1 - vline[0] * hline[0])

    return (x, y)


def perspective_transform(image, src, size):
    """Crop out a solar cell

    Parameters
    ----------
    image: array
    Perspective transformed image of solar module

    src: array
    Sorted coordinates of corners. [top-left, top-right, bottom-left, bottom-right]

    size: int
    Size of cropped solar cell, e.g., to output 32x32 cells, input 32.

    Returns
    -------
    warped: array
    Transformed solar cell image
    """
    src = np.float32(src)
    dst = np.float32([(0, 0), (size, 0), (0, size), (size, size)])
    M = cv.getPerspectiveTransform(src, dst)

    warped = cv.warpPerspective(image, M, (size, size))

    return warped


def segment_cell(image, abs_x_couple, abs_y_couple, cellsize=32, save_path=None):
    """Crop out solar cells from a module

    Parameters
    ----------
    image: array
    Perspective transformed image of solar module

    abs_x_couple, abs_y_couple: array
    Couple of horizontal edges, vertical edges

    cellsize: int
    Size of cropped solar cell, e.g., to output 32x32 cells, input 32.

    save_path: str or pathlib.PosixPath
    Path to store the output

    Returns
    -------
    cells: array
    Images of cropped cells
    """
    cells = []
    for i, hline_ab in enumerate(abs_y_couple):
        for j, vline_ab in enumerate(abs_x_couple):
            xy = []
            for hab in hline_ab:
                for vab in vline_ab:
                    xy.append(cross(vab, hab))

            warped = perspective_transform(image, xy, cellsize)
            cells.append(warped)

            # counter += 1

            if save_path:
                cv.imwrite(str(save_path / (str(i) + str(j) + '.png')), warped)

    return cells


def coordinate2inx(coordinate, row=8, col=16, im_shape=[300, 600]):
    """Convert coordinate of coordinate of top-left corner into index. Index order is:
    [[0 1 2]
     [3 4 5]
     [6 7 8]]

    row, col: int
    Number of rows/columns of solar module

    im_shape: list
    [height, width] of perspective transformed solar module

    Returns
    -------
    inx: int
    Index
    """
    inx = col * round(coordinate[1] / (im_shape[0] / row)) + round(coordinate[0] / (im_shape[1] / col))

    return inx


def classify_cells(ann_path, defects_inx_dic, row_col=[8, 16], shape=[300, 600]):
    """Classify solar cells based on the class of the annotation of bounding box on the solar module

    Parameters
    ----------
    ann_path: str or pathlib.PosixPath
    Path of annotation file

    defects_inx_dic: dict
    Dict of defects with empty value. Used to store the index of the defective cells
    e.g.
    defects_dic = {
        'crack_bbox': [],
        'solder_bbox': [],
        'intra_bbox': [],
        'oxygen_bbox': []
    }

    row_col: list
    [row, col] of solar module

    shape: list
    [height, width] of solar module image

    Returns
    -------
    defects_inx_dic:
    Dict of defects with the index of the defective cells
    """
    with open(ann_path, 'r') as file:
        data = json.load(file)
    if len(data['objects']) > 0:
        for obj in data['objects']:
            classTitle = obj['classTitle']
            points = obj['points']['exterior'][0]
            inx = coordinate2inx(points, row=row_col[0], col=row_col[1], im_shape=shape)
            defects_inx_dic[classTitle].append(inx)

    return defects_inx_dic


def write_cells(single_cells, defects_inx, defect2folder, name, save_path):
    """Output cells into folders of corresponding class

    Parameters
    ----------
    single_cells: array
    Images of cropped cells

    defects_inx: dict
    Dict of defects with the index of the defective cells

    defect2folder: dict
    Convert keys in defects_inx into names of folders
    e.g.
    {
        "crack_bbox": "crack",
        "oxygen_bbox": "oxygen",
        "solder_bbox": "solder",
        "intra_bbox": "intra"
    }

    name: str
    Name of the solar module

    save_path: str or pathlib.PosixPath
    Path to store the output
    """
    for inx, cell in enumerate(single_cells):
        flag = True
        for key, value in defects_inx.items():
            if inx in value:
                cv.imwrite(str(Path(save_path) / f"{defect2folder[key]}/{name}__{str(inx)}.png"), cell)
                flag = False
        """
        if inx in defects_inx['crack_bbox']:
            cv.imwrite(path + 'crack/' + name + '__' + str(inx) + '.png', cell)
            flag = False
        if inx in defects_inx['solder_bbox']:
            cv.imwrite(path + 'solder/' + name + '__' + str(inx) + '.png', cell)
            flag = False
        if inx in defects_inx['oxygen_bbox']:
            cv.imwrite(path + 'oxygen/' + name + '__' + str(inx) + '.png', cell)
            flag = False
        if inx in defects_inx['intra_bbox']:
            cv.imwrite(path + 'intra/' + name + '__' + str(inx) + '.png', cell)
            flag = False
        """
        if flag:
            cv.imwrite(str(Path(save_path) / f"intact/{name}__{str(inx)}.png"), cell)
