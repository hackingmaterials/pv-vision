import numpy as np
import cv2 as cv
import zlib
import base64
from scipy import signal
import json


def image_threshold(image, blur=5, blocksize=11, threshold=None, adaptive=False):
    """Return an binary image. AdaptiveThreshold will keep skeleton of solar modules,
    threshold will only keep contour.

    Parameters
    ----------
    image: array
    Image of solar modules

    blur: int
    Block size in cv.medianBlur

    blocksize: int
    blocksize in cv.adaptiveThreshold

    threshold: int or float
    Threshold value in cv.threshold.
    Pixels above this value will be set 255, and below will be 0.
    If threshold < 1, first threshold (%) of the image grayscale value will be used.
    If not given, first 10% of the image grayscale value will be used.

    adaptive: bool
    if true, adaptiveThreshold will be used # need to change inverse binary into binary

    Returns
    -------
    image_thre: array
    Binary image
    """
    # image_eq = cv.equalizeHist(image_resize)
    image_blur = cv.medianBlur(image, blur)
    if adaptive:
        image_thre = cv.adaptiveThreshold(image_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY, blocksize, 2)
    else:
        if threshold is None:
            threshold = np.quantile(image_blur, 0.1)
        elif threshold < 1:
            threshold = np.quantile(image_blur, threshold)
        _, image_thre = cv.threshold(image_blur, threshold, 255, cv.THRESH_BINARY)

    image_thre = cv.medianBlur(image_thre, 1)

    return image_thre


def base64_2_mask(s):
    """decode the masks from supervisely

    Parameters
    ----------
    s: str
    code of mask from annotated on supervisely

    Returns
    -------
    mask: array of bool
    Image of mask. Only show the convex.
    """
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv.imdecode(n, cv.IMREAD_UNCHANGED)[:, :, 3].astype(bool)

    return mask


def has_mask(mask_name, path=None, data=None):
    """Check if mask exists
    Parameters
    ----------
    mask_name: str
    The annotation name of the mask.

    path: str or pathlib.PosixPath
    The path of annotation json file

    data: dict
    If provided, will not open path

    Returns
    -------
    If exist, return the index in objects list
    If not, return False
    """
    if path is None and data is None:
        raise ValueError("Mask file not provided.")
    if path:
        with open(path, 'r') as file:
            data = json.load(file)
    
    for inx, obj in enumerate(data["objects"]):
        if obj['classTitle'] == mask_name:
            return inx
    
    return False


def load_mask(path, image, mask_name='module_unet', center=True):
    """Load the image of mask

    Parameters
    ----------
    path: str or pathlib.PosixPath
    The path of annotation json file

    image: array
    The original image of solar module

    mask_name: str
    The annotation name of the mask. Default is 'module_unet'.

    center: bool
    If True, return mask center.

    Returns
    -------
    mask: array
    The image of the mask. The shape is the same as the image of module.

    mask_center: array
    The center coordinate of the mask in the form of [x, y]
    """
    with open(path, 'r') as file:
        data = json.load(file)
    # if len(data["objects"]) == 0:
    #    return None
        # code = data["objects"][0]["bitmap"]["data"]
        # origin = data["objects"][0]["bitmap"]["origin"]
    # else:
    #    flag = True
    #    for obj in data["objects"]:
    #        if obj['classTitle'] == mask_name:
    inx = has_mask(mask_name, data=data)
    if inx is not False:
        obj = data["objects"][inx]
        code = obj["bitmap"]["data"]
        origin = obj["bitmap"]["origin"]
    else:
        mask = np.zeros((image.shape[0], image.shape[1]))
        mask = mask.astype('uint8')
        mask_center = np.array([mask.shape[1]/2, mask.shape[0]/2])
        if center:
            return mask, mask_center
        else:
            return mask
    mask = base64_2_mask(code)
    mask_center = np.array([mask.shape[1]/2, mask.shape[0]/2])
    mask_center += origin

    up = np.zeros((origin[1], mask.shape[1]))
    mask2 = np.vstack((up, mask))
    left = np.zeros((mask2.shape[0], origin[0]))
    mask3 = np.hstack((left, mask2))
    down = np.zeros((image.shape[0] - mask3.shape[0], mask3.shape[1]))
    mask4 = np.vstack((mask3, down))
    right = np.zeros((mask4.shape[0], image.shape[1] - mask4.shape[1]))
    mask5 = np.hstack((mask4, right))

    if center:
        return mask5.astype('uint8'), mask_center.astype(int)
    else:
        return mask5.astype('uint8')


def find_intersection(mask_part, houghlinePara=50):
    """Find the intersection of two edges. Edges detected by houghline method.
    Here we use corner part of a mask to make better detection. (Only one corner need
    to be detected in a corner part).

    Parameters
    ----------
    mask_part: array
    Image of the corner part of a mask

    houghlinePara: int
    Threshold parameter in cv.HoughLines()

    Returns
    -------
    x_cross, y_cross: float
    Coordinate of the intersection
    """
    edge = cv.Canny(mask_part, 0, 1)
    lines = cv.HoughLines(edge, 1, np.pi / 180, houghlinePara)

    rhos = []
    thetas = []
    for line in lines:
        rho, theta = line[0]
        rhos.append(rho)
        thetas.append(theta)

    thetas = np.array(thetas)
    rhos = np.array(rhos)
    mean = np.mean(thetas)
    inx = thetas > mean

    thetas1 = thetas[inx]
    rhos1 = rhos[inx]
    thetas2 = thetas[1 - inx != 0]
    rhos2 = rhos[1 - inx != 0]
    # detect outliers
    inx2 = np.abs(rhos1-np.mean(rhos1)) <= np.std(rhos1)
    rhos1 = rhos1[inx2]
    thetas1 = thetas1[inx2]
    inx3 = np.abs(rhos2-np.mean(rhos2)) <= np.std(rhos2)
    rhos2 = rhos2[inx3]
    thetas2 = thetas2[inx3]

    theta1 = np.mean(thetas1)
    rho1 = np.mean(rhos1)
    theta2 = np.mean(thetas2)
    rho2 = np.mean(rhos2)

    k1 = -1 / np.tan(theta1)
    k2 = -1 / np.tan(theta2)
    b1 = rho1 * np.sin(theta1) - k1 * rho1 * np.cos(theta1)
    b2 = rho2 * np.sin(theta2) - k2 * rho2 * np.cos(theta2)

    x_cross = (b2-b1) / (k1-k2)
    y_cross = (k1 * b2 - k2 * b1) / (k1 - k2)
    # return thetas1, thetas2
    return x_cross, y_cross


def find_module_corner(mask, mask_center, dist=200, displace=0, method=0, corner_center=False, center_displace=10):
    """Detect the corner of solar module. Intersection of edges or corner_detection method is used.

    Parameters
    ----------
    mask: array
    Image of mask

    mask_center: array
    Coordinate of the mask center

    dist: int
    Distance threshold between two corners. Default is 200.

    displace: int
    Displacement of the detected corners to increase tolerance.

    method: int
    0 = use cv.goodFeaturesToTrack() to detect corners
    1 = use cv.HoughLines() to detect edges first and then find the intersections

    corner_center: Bool
    If True, use auto-detected nask center. Otherwise use 'mask_center' parameter. Default is False.

    center_displace: int
    Displacement of the mask center when dividing the mask into four corner parts

    Returns
    -------
    Corners: array
    Sorted coordinates of module corners. The order is top-left, top-right, bottom-left and bottom-right
    """

    x_m = mask_center[0]
    y_m = mask_center[1]

    if corner_center:
        corners = cv.goodFeaturesToTrack(mask, 4, 0.01, 200, blockSize=9)
        corners = np.int0(corners)
        xs1 = []
        ys1 = []
        for i in corners:
            x, y = i.ravel()
            xs1.append(x)
            ys1.append(y)
        x_m = int(np.mean(xs1))
        y_m = int(np.mean(ys1))
    # divide the mask into four corner parts to increase the accuracy
    A = mask[0:y_m+center_displace, 0:x_m+center_displace]
    B = mask[0:y_m+center_displace, x_m-center_displace:]
    C = mask[y_m-center_displace:, 0:x_m+center_displace]
    D = mask[y_m-center_displace:, x_m-center_displace:]

    xs = []
    ys = []
    if method == 0:
        corners_A = cv.goodFeaturesToTrack(A, 1, 0.01, dist, blockSize=9)
        corners_A = np.int0(corners_A)
        corners_B = cv.goodFeaturesToTrack(B, 1, 0.01, dist, blockSize=9)
        corners_B = np.int0(corners_B)
        corners_C = cv.goodFeaturesToTrack(C, 1, 0.01, dist, blockSize=9)
        corners_C = np.int0(corners_C)
        corners_D = cv.goodFeaturesToTrack(D, 1, 0.01, dist, blockSize=9)
        corners_D = np.int0(corners_D)

        for corners in [corners_A, corners_B, corners_C, corners_D]:
            for i in corners:
                x, y = i.ravel()
                xs.append(x)
                ys.append(y)

    if method == 1:
        for part in [A, B, C, D]:
            x_cross, y_cross = find_intersection(part)
            xs.append(x_cross)
            ys.append(y_cross)

    # sort out the corners
    xs[1] += x_m - center_displace
    ys[2] += y_m - center_displace
    xs[3] += x_m - center_displace
    ys[3] += y_m - center_displace

    xs[0] -= displace
    ys[0] -= displace

    xs[1] += displace
    ys[1] -= displace

    xs[2] -= displace
    ys[2] += displace

    xs[3] += displace
    ys[3] += displace

    corners_order = list(zip(xs, ys))

    return np.array(corners_order)


def perspective_transform(image, src, sizex, sizey, rotate=True):
    """Do perspective transform on the solar modules. Orientation of the input module is auto-detected. The output
    module has short side vertically arranged and long side horizontally arranged.

    Parameters
    ----------
    image: array
    Image of solar modules

    src: array
    Sorted coordinates of corners

    sizex, sizey: int
    size of the output image. x is the long side and y is the short side.

    rotate: bool
    If true, auto-detection of orientation is on.

    Returns
    -------
    warped: array
    Transformed image of solar module
    """
    src = np.float32(src)
    
    if rotate and np.sum((src[0] - src[2])**2) > np.sum((src[0] - src[1])**2):
        dst = np.float32([(0, sizey), (0, 0), (sizex, sizey), (sizex, 0)])
    else:
        dst = np.float32([(0, 0), (sizex, 0), (0, sizey), (sizex, sizey)])
    #if np.sum((src[0] - src[2])**2) <= np.sum((src[0] - src[1])**2):
    #    dst = np.float32([(0, 0), (sizex, 0), (0, sizey), (sizex, sizey)])
    #else:
        
    M = cv.getPerspectiveTransform(src, dst)

    warped = cv.warpPerspective(image, M, (sizex, sizey))

    return warped


def find_inner_edge(wrap, dist=25, prom=0.08):  # used to be named as find_cell_corner
    """Detect the inner edges of the transformed solar module. This can be used to verify whether the module
    is successfully transformed.

    Parameters:
    -----------
    wrap: array
    Transformed image of solar module. 
    It should be grayscale or BGR read from Opencv.

    dist: int
    Distance threshold between two edge signals. If the size of solar is large, the threshold should be changed.
    Default is 25 for a 300x600 pixel image of solar module.

    prom: int
    Prominence parameter in scipy.signal.find_peaks(). If signal is weak, the prominence should be decreased.

    Returns
    -------
    peak_x, peak_y: array
    Indices of peaks along x and y directions
    """
    if len(wrap.shape) == 2:
        wrap_g = wrap
    elif len(wrap.shape) == 3:
        wrap_g = cv.cvtColor(wrap, cv.COLOR_BGR2GRAY)

    sum_x = np.sum(wrap_g, axis=0)
    sum_x = sum_x / np.max(sum_x)
    peak_x, _ = signal.find_peaks(-sum_x, distance=dist, prominence=prom)

    sum_y = np.sum(wrap_g, axis=1)
    sum_y = sum_y / np.max(sum_y)
    peak_y, _ = signal.find_peaks(-sum_y, distance=dist, prominence=prom)

    return peak_x, peak_y


# a new version of finding the module corners #
def fill_polygon(points, im_shape):
    """Fill the polygon defined by convex or contour points

    Parameters
    ----------
    points: array
    Coordinates of the points that define the convex or contour of the mask

    im_shape: array
    Array shape of the mask

    Returns
    -------
    im_cnt: array
    Filled contour or convex
    """
    im_cnt = np.zeros((im_shape[0],im_shape[1],1), np.uint8)
    cv.fillPoly(im_cnt, [points], (255,255))

    return im_cnt


def sort_corners(corners):
    """Sort corners for perspective transform

    Parameters
    ----------
    corners: array
    Corners detected by cv.goodFeaturesToTrack()

    Returns
    -------
    sorted_corners: array
    Sorted coordinates of module corners. The order is top-left, top-right, bottom-left and bottom-right.
    """
    col_sorted = corners[np.argsort(corners[:, 1])] # sort on the value in column

    # sort on the value in rows. a, b are the indexes
    a = np.argsort(col_sorted[:2, 0])
    b = np.argsort(col_sorted[2:, 0]) + 2

    return col_sorted[np.hstack((a, b))]


def find_polygon_corners(im, n_corners=4, dist_corners=100, displace=3):
    """Use cv.goodFeaturesToTrack to find the four corners of a filled polygon

    Parameters
    ----------
    im: array
    image of the mask

    n_corners: int
    Number of the corners. Default is 4.

    dist_corners: int
    Distance threshold between two neighbour corners. Default is 100 pixels.

    displace: int
    Displace detected corners to increase tolerance. Default is 3.

    Returns
    -------
    corners_displaced: array
    Detected corners that are sorted and displaced.
    """
    corners_unravel = cv.goodFeaturesToTrack(im, n_corners, 0.01, dist_corners, blockSize=9),
    corners = []
    for corner in corners_unravel[0]:
        corners.append(list(corner.ravel()))
    corners_sorted = sort_corners(np.array(corners))
    corners_displaced = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]) * displace + corners_sorted

    return corners_displaced


def find_module_corner2(mask, mode=0):
    """ Detect the corners of a solar module

    Parameters
    ----------
    mask: array
    Image of a mask

    mode: int
    mode == 0: detect corners of the convex of module
    mode == 1: detect corners of the approximated convex of module
    mode == 2: detect corners of the approximated contour of the module
    mode == 3: detect corners of the blurred mask of the module
    mode == 4: detect corners using boudingRect

    Returns
    -------
    Corners: array
    Corners of the solar module
    """
    blur = cv.blur(mask, (6, 6))
    contours, _ = cv.findContours(blur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    inx = 0
    length = 0
    for i, cnt in enumerate(contours):
        if length < len(cnt):
            length = len(cnt)
            inx = i

    if mode == 4:
        rect = cv.minAreaRect(contours[inx])
        corners = cv.boxPoints(rect)
        corners_sorted = sort_corners(np.array(corners))
        corners_displaced = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]) * 3 + corners_sorted
        return corners_displaced

    cnt_approx = cv.approxPolyDP(contours[inx], 8, True)
    convex = cv.convexHull(contours[inx])
    conv_approx = cv.approxPolyDP(convex, 8, True)

    if mode == 0:
        im_conv = fill_polygon(convex, blur.shape)
        corners = find_polygon_corners(im_conv)
    elif mode == 1:
        im_conv_app = fill_polygon(conv_approx, blur.shape)
        corners = find_polygon_corners(im_conv_app)
    elif mode == 2:
        im_cnt_app = fill_polygon(cnt_approx, blur.shape)
        corners = find_polygon_corners(im_cnt_app)
    elif mode == 3:
        corners = find_polygon_corners(blur)
    else:
        raise Exception("mode must be one of 0, 1, 2, 3, 4")
    
    return corners

