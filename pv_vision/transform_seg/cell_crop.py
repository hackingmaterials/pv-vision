import numpy as np
import cv2 as cv
import json
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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


def linear_regression(inxes, lines, outlier_filter=False, threshold=0.3):
    """Linear regression: f = ax + b. Return multiple fitted lines. After detect_edge, do linear fitting to
    return the final parameters(a and b) of edges.

    Parameters
    ----------
    inxes: list, or array
    Values of x

    lines: list, or array
    Each line in lines represents values of f.

    outlier_filter: bool
    If true, outliers will be filtered

    threshold: float
    Filter threshold

    Returns
    -------
    ab_s: list
    List of a and b of multiple lines
    """
    ab_s = []

    for line in lines:
        ab, _ = curve_fit(linear, inxes, line)
        if outlier_filter:
            line_fit = ab[0] * inxes + ab[1]
            mse = np.mean((line - line_fit) ** 2)
            cook_distances = []
            outlier_inx = []
            for inx, element in enumerate(line):
                inx_i = np.delete(inxes, inx)
                line_i = np.delete(line, inx)
                ab_i, _ = curve_fit(linear, inx_i, line_i)
                line_fit_i = ab_i[0] * inxes + ab_i[1]
                cook_distance = (np.sum((line_fit - line_fit_i) ** 2)) / mse
                cook_distances.append(cook_distance)
                if cook_distance > threshold:
                    outlier_inx.append(inx)
            inxes_new = np.delete(inxes, outlier_inx)
            line_new = np.delete(line, outlier_inx)
            ab, _ = curve_fit(linear, inxes_new, line_new)

        ab_s.append(ab)

    ab_s = np.array(ab_s)

    return ab_s


def detectoutliers(data, rate=1.5, option=1):
    # may be replaced by a better algorithm
    """ Detect outliers among peaks(internal edges) of a split.
    If option == 0, will replace outliers.
    If option == 1, will remove outliers. (e.g. some extra wrong peaks)

    Parameters
    ----------
    data: array
    Peaks of internal edges

    rate: float
    Tolerance of good data

    option: int
    """

    bottom = np.percentile(data, 25)
    up = np.percentile(data, 75)
    IQR = up - bottom
    outlier_step = rate * IQR

    outlier_list = (data < bottom - outlier_step) | (data > up + outlier_step)
    if option == 0:
        for inx, flag in enumerate(outlier_list):
            if flag:
                left = inx
                right = inx
                while outlier_list[left]:
                    left -= 1
                    if left <= 0:
                        left = inx
                        while outlier_list[left]:
                            left += 1

                while outlier_list[right]:
                    right += 1
                    if right >= len(outlier_list) - 1:
                        right = inx
                        while outlier_list[right]:
                            right -= 1
                outlier_list[inx] = False
                data[inx] = (data[left] + data[right]) / 2

        return data

    if option == 1:
        outlier_list = np.insert(outlier_list, 4, False)
        for i in range(5, len(outlier_list) - 5):
            if outlier_list[i]:
                outlier_list[i] = False

        return (1 - outlier_list).astype(np.bool)


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


def split_img(image, split=None, split_size=None, direction=0):
    """Split the image along x/y axis.

    Parameters
    ----------
    image: array
    Image to be split

    split: int
    Number of splits.

    split_size: int
    Size of each split. Ignored if SPLIT is provided.

    direction: int
    If 0: along y axis. Generate horizontal splits. Detect vertical lines.
    If 1: along x axis. Generate vertical splits. Detect horizontal lines.

    Returns
    -------
    splits
    """
    if (direction != 1) and (direction != 0):
        raise ValueError("Direction should be either 0 or 1.")

    dimension = image.shape[direction]
    if split_size and (split is None):
        split = int(dimension / split_size)
    elif (split_size is None) and (split is None):
        raise ValueError("Either split or split_size should be provided.")

    end = int(dimension / split)
    if direction == 0:
        splits = np.vsplit(image[: end * split, :], split)
        if end * split < dimension:
            splits.append(image[end * split:, :])
    elif direction == 1:
        splits = np.hsplit(image[:, :end * split], split)
        if end * split < dimension:
            splits.append(image[:, end * split:])

    return splits


def detect_peaks(split, direction, cell_size, busbar, thre=0.9, interval=None, margin=None):
    """Detect peaks which correspond to internal edges

    Parameters
    ----------
    split: array
    Split of the image

    direction: int
    If 0: along y axis. Process horizontal splits. Detect vertical lines.
    If 1: along x axis. Process vertical splits. Detect horizontal lines.

    cell_size: int
    Size (pixel) of a cell in the raw image.

    busbar: int
    Number of busbars of a cell.

    thre: float
    Peak intensity above THRE will be set as 1.
    Note that the edge's peak intensity should be lowest because edges are black

    interval: int
    Distance between each peak.
    If direction == 0, default is int(cell_size * 0.95)
    If direction == 1, Default is int(cell_size / (busbar + 1) * 0.5)

    margin: int
    Margin in which there shouldn't be edges but may be wrongly detected.
    """
    if (direction != 1) and (direction != 0):
        raise ValueError("Direction should be either 0 or 1.")

    if interval is None and direction == 0:
        interval = int(cell_size * 0.95)
    elif interval is None and direction == 1:
        interval = int(cell_size / (busbar + 1) * 0.5)

    sum_split = np.sum(split, axis=direction)
    sum_split = sum_split / np.max(sum_split)
    sum_split[sum_split > thre] = 1

    peaks, _ = find_peaks(-1 * sum_split, distance=interval)
    if margin:
        peaks = filter_margin(peaks, im_length=split.shape[direction - 1], margin=margin)

    return peaks


def plot_peaks(n, image, cell_size, busbar, split=None, split_size=None, direction=0, thre=0.9, interval=None, margin=None):
    splits = split_img(image, split, split_size, direction)
    split = splits[n]
    sum_split = np.sum(split, axis=direction)
    sum_split = sum_split / np.max(sum_split)
    sum_split[sum_split > thre] = 1
    peaks = detect_peaks(split, direction, cell_size, busbar, thre, interval, margin)
    plt.plot(list(range(len(sum_split))), sum_split)
    plt.scatter(peaks, sum_split[peaks])


def detect_vertical_lines(image_thre, column, cell_size, thre=0.8, split=100, peak_interval=None, margin=None):
    """ Detect vertical edges by segmenting image into horizontal splits

    Parameters
    ----------
    image_thre: array
    Adaptive threshold of raw images

    column: int
    Number of columns of solar module

    cell_size: int
    Output cell size in pixel

    thre: float
    Peak intensity above THRE will be set as 1.
    Note that the edge's peak intensity should be lowest because edges are black

    split: int
    Number of splits

    peak_interval: int
    Distance between each peak. Default is int(cell_size * 0.95)

    Returns
    -------
    vline_abs_couple: array
    Suppose a line is y=a*x+b.
    Return a and b of a couple edges (left and right of a cell).
    """
    #height = image_thre.shape[0]
    #end = int(height / split)
    #image_hsplits = np.vsplit(image_thre[: end * split, :], split)  # horizontal splits
    #image_hsplits.append(image_thre[end * split:, :])
    image_hsplits = split_img(image_thre, split=split, direction=0)

    edge_x = []
    inx_y = []
    for inx, im_split in enumerate(image_hsplits):
        #sum_split = np.sum(im_split, axis=0)
        #sum_split = sum_split / np.max(sum_split)
        #sum_split[sum_split > thre] = 1

        #if peak_interval is None:
        #    peak_interval = int(cell_size * 0.95)
        #peak, _ = find_peaks(-1 * sum_split, distance=peak_interval)
        peak = detect_peaks(im_split, direction=0, cell_size=cell_size,
                            busbar=None, thre=thre, interval=peak_interval, margin=margin)
        if len(peak) > column - 2:
            peak_new = [peak[0]]
            for i in range(1, len(peak) - 1):
                if np.abs(peak[i] - peak[i + 1]) < 15:
                    peak_mean = (peak[i] + peak[i + 1]) / 2
                    peak_new.append(peak_mean)
                elif np.abs(peak[i] - peak[i - 1]) > 15:
                    peak_new.append(peak[i])

            peak_new.append(peak[-1])
            peak_new = np.array(peak_new)
            peak_new_a = np.delete(peak_new, 0)
            peak_new_b = np.delete(peak_new, -1)
            peak_new_detect = peak_new[detectoutliers(np.abs(peak_new_a - peak_new_b), option=1)]

            if len(peak_new_detect) == 1 + column:
                edge_x.append(peak_new_detect)
                inx_mean = ((2 * inx + 1) * (image_thre.shape[0] / split) - 1) / 2
                inx_y.append(inx_mean)
    edge_x = np.array(edge_x)

    vlines = list(zip(*edge_x))  # line parallel to y axis
    vlines = np.array(vlines)
    inx_y = np.array(inx_y)
    # for lines in vlines:
    #    lines_new = self.detectoutliers(lines, option=0)
    #    while np.std(lines_new) > 15:
    #        lines_new = self.detectoutliers(lines, rate=1, option=0)

    # v_abs = []
    # for verticaline in vlines:
    #    ab, _ = curve_fit(self.linear, inx_y, verticaline) # x = ay + b
    #    v_abs.append(ab)
    v_abs = linear_regression(inx_y, vlines, outlier_filter=True)
    # temp1 = v_abs.copy()
    temp1 = np.delete(v_abs, -1, 0)
    # temp2 = v_abs.copy()
    temp2 = np.delete(v_abs, 0, 0)

    vline_abs_couple = np.array(list(zip(temp1, temp2)))
    # vline_abs = [(v_abs[i],v_abs[i+1]) for i in range(0, len(v_abs), 2)] # put edges of each cell into a tuple

    return vline_abs_couple


def detect_horizon_lines(image_thre, row, busbar, cell_size, thre=0.6, split=50, peak_interval=None, margin=None):
    """ Detect horizontal edges by segmenting image into vertical splits

    Parameters
    ---------
    image_thre: array
    Adaptive threshold of raw images

    row: int
    Number of rows of solar module

    busbar: int
    Number of busbars of a solar cell

    cell_size: int
    Output cell size in pixel

    thre: float
    Peak intensity above THRE will be set as 1.
    Note that the edge's peak intensity should be lowest because edges are black

    split: int
    Number of splits

    peak_interval: int
    Distance between each peak.

    Returns
    -------
    hline_abs_couple: array
    Suppose a line is y=a*x+b.
    Return 'a' and 'b' of a couple edges (top and bottom of a cell).
        """
    #width = image_thre.shape[1]
    #end = int(width / split)
    #image_vsplits = np.hsplit(image_thre[:, :end * split], split)  # vertical splits
    #image_vsplits.append(image_thre[:, end * split:])
    image_vsplits = split_img(image_thre, split=split, direction=1)

    edge_y = []
    inx_x = []
    for inx, im_split in enumerate(image_vsplits):
        #sum_split = np.sum(im_split, axis=1)
        #sum_split = sum_split / np.max(sum_split)
        #sum_split[sum_split > thre] = 1

        #if peak_interval is None:
        #    peak_interval = int(cell_size / (busbar + 1) * 0.5)
        #peak, _ = find_peaks(-1 * sum_split, distance=peak_interval)
        peak = detect_peaks(im_split, 1, cell_size, busbar, thre, peak_interval, margin=margin)
        if len(peak) >= row * (busbar + 1) - 1:
            peak_new = [peak[0]]
            for i in range(1, len(peak) - 1):
                if np.abs(peak[i] - peak[i + 1]) < 15:
                    peak_mean = (peak[i] + peak[i + 1]) / 2
                    peak_new.append(peak_mean)
                elif np.abs(peak[i] - peak[i - 1]) > 15:
                    peak_new.append(peak[i])

            peak_new.append(peak[-1])
            peak_new = np.array(peak_new)
            peak_new_a = np.delete(peak_new, 0)
            peak_new_b = np.delete(peak_new, -1)
            peak_new_detect = peak_new[detectoutliers(np.abs(peak_new_a - peak_new_b), rate=0.5, option=1)]

            if len(peak_new_detect) == (busbar + 1) * row + 1:
                edge_y.append(peak_new_detect)
                inx_mean = ((2 * inx + 1) * (image_thre.shape[1] / split) - 1) / 2
                inx_x.append(inx_mean)

    edge_y = np.array(edge_y)

    hlines = list(zip(*edge_y))
    hlines = np.array(hlines)
    inx_x = np.array(inx_x)
    # for lines in hlines:
    #    lines_new = self.detectoutliers(lines, option=0)
    #    while np.std(lines_new) > 10:
    #        lines_new = self.detectoutliers(lines, rate=1, option=0)

    # hb_abs = [] # all lines including busbar
    hb_abs = linear_regression(inx_x, hlines, outlier_filter=True)
    hline_abs_couple = []  # all lines excluding busbar

    # for horizonline in hlines:
    #    ab, _ = curve_fit(self.linear, inx_x, horizonline) # y = ax + b
    #    hb_abs.append(ab)

    hline_abs_couple = [(hb_abs[(busbar + 1) * i], hb_abs[(busbar + 1) * (i + 1)]) for i in range(row)]
    # hline_abs = [(hb_abs[(4+1)*i],hb_abs[(4+1)*(i+1)]) for i in range(6)]
    # hline_abs = [(hb_abs[(self.busbar+2)*i],hb_abs[(self.busbar+2)*(i+1)-1]) for i in range(self.row)]

    return hline_abs_couple


def detect_edge(image, row_col, cell_size, busbar, peaks_on=0, split_size=10, peak_interval=None, margin=20):
    """Detect the inner edges of a solar module. Split the solar module first, then detect position of
    edges in each split.
    Note that the solar module is already perspectively transformed.

    Parameters
    ----------
    image: array
    Perspective transformed image of solar module. 
    Input should be grayscale or BGR read from Opencv

    row_col: list
    Number of rows/columns of solar module

    cell_size: int
    Estimated size of single cells in pixels. Default is 30.

    busbar: int
    number of busbar

    peaks_on: int
    Detect horizontal edges or vertical edges
    If 0: detect vertical edges
    If 1: detect horizontal edges

    split_size: int
    The size of each split

    peak_interval: int
    Distance between each peak.

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

    if peaks_on != 0 and peaks_on != 1:
        raise ValueError('peaks_on must be 0 or 1')

    #if peaks_on == 0:
    #    splits = np.vsplit(image_g, image_g.shape[0] / split_size)

    #elif peaks_on == 1:
    #    splits = np.hsplit(image_g, image_g.shape[1] / split_size)

    #else:
    #    raise ValueError('peaks_on must be 0 or 1')

    splits = split_img(image_g, split_size=split_size, direction=peaks_on)

    peaklist = []
    splits_inx = []

    # peaks when suming the whole image
    #sum_whole = np.sum(image_g, axis=peaks_on)
    #sum_whole = sum_whole / sum_whole.max()
    #peaks_whole, _ = find_peaks(-sum_whole, distance=cell_size)
    #peaks_whole = filter_margin(peaks_whole, im_length=im_size[peaks_on - 1], margin=margin)
    peaks_whole = detect_peaks(image_g, peaks_on, cell_size, busbar, interval=peak_interval, margin=margin)

    flag = False
    for inx, split in enumerate(splits):

        #sum_split = np.sum(split, axis=peaks_on)
        #sum_split = sum_split / sum_split.max()
        #peaks_split, _ = find_peaks(-sum_split, distance=cell_size)
        #peaks_split = filter_margin(peaks_split, im_length=im_size[peaks_on - 1], margin=margin)
        peaks_split = detect_peaks(split, peaks_on, cell_size, busbar, interval=peak_interval, margin=margin)

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


def correction(lines, displacement):
    """The first line in the tuple moves to negative direction,
    the second to positive direction.

    Parameters
    ----------
    lines: list or array

    """
    lines_new = []
    for couple in lines:
        minus = displace(couple[0], -displacement)
        plus = displace(couple[1], displacement)
        lines_new.append((minus, plus))

    return lines_new


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


def segment_cell(image, hline_abs_couple, vline_abs_couple, cellsize,
                 save_path=None, displacement=None):
    """Crop out solar cells from a module

    Parameters
    ----------
    image: array
    Perspective transformed image of solar module

    hline_abs_couple, vline_abs_couple,: array
    Couple of horizontal edges, vertical edges

    cellsize: int
    Size of cropped solar cell, e.g., to output 32x32 cells, input 32.

    save_path: str or pathlib.PosixPath
    Path to store the output

    displacement
    If displace, then move edge couples with displacement

    Returns
    -------
    cells: array
    Images of cropped cells
    """
    if displacement:
        hline_abs_couple = correction(hline_abs_couple, displacement)
        vline_abs_couple = correction(vline_abs_couple, displacement)

    cells = []
    for i, hline_ab in enumerate(hline_abs_couple):
        for j, vline_ab in enumerate(vline_abs_couple):
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


def plot_edges(image, hline_abs_couple, vline_abs_couple, displacement=1, linewidth=1):
    """Plot the detected edges

    Parameters
    ----------
    image: array
    Raw image

    hline_abs_couple, vline_abs_couple: list or array
    a and b of edge couples

    displacement: int
    Displace detected edges

    linewidth: int
    Width of plotted lines
    """
    newhline_abs = correction(hline_abs_couple, displacement)
    newvline_abs = correction(vline_abs_couple, displacement)

    plt.imshow(image, cmap='gray')
    x = np.array(range(image.shape[1]))
    y = np.array(range(image.shape[0]))

    for h_ab in newhline_abs:
        plt.plot(x, h_ab[0][0] * x + h_ab[0][1], linewidth=linewidth, linestyle='--', color='r')
        plt.plot(x, h_ab[1][0] * x + h_ab[1][1], linewidth=linewidth, linestyle='--', color='r')

    for v_ab in newvline_abs:
        plt.plot(v_ab[0][0] * y + v_ab[0][1], y, linewidth=linewidth, linestyle='--', color='r')
        plt.plot(v_ab[1][0] * y + v_ab[1][1], y, linewidth=linewidth, linestyle='--', color='r')


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
