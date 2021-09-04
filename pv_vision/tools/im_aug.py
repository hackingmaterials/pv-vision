import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def lightness_balance(src, threshold=[1, 99], extend=[0.1, 0.9]):
    # compute the lightness of source image
    hsv_image = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    # set a threshold of min_grayscale and max_grayscale,
    # outside which are removed
    max_percentile_pixel = np.percentile(src, threshold[1])
    min_percentile_pixel = np.percentile(src, threshold[0])

    # remove the pixels of grayscale outside the threshold
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # balance brightness
    out = np.zeros(src.shape, src.dtype)
    cv.normalize(src, out, 255*extend[0], 255*extend[1], cv.NORM_MINMAX)

    return out


def edge_remove(im_src, threshold=0.2878, displace=3, size=400):
    # find the edge, here only find horizontal edge
    image = im_src / np.max(im_src)
    image_mask = image > threshold
    sum_1 = np.sum(image_mask, axis=1)
    sum_1 = sum_1 / np.max(sum_1)
    
    inx = np.argwhere(sum_1 > 0)
    top = inx[5]-5
    bottom = inx[-5]+5
    if top > displace:
        top -= displace
    else:
        top = 0
    if bottom < (im_src.shape[0]-displace):
        bottom += displace
    else:
        bottom = im_src.shape[-1]
    top = int(top)
    bottom = int(bottom)

    # transform
    src = [(0, top), (im_src.shape[-1], top), (0, bottom),
           (im_src.shape[-1], bottom)]
    src = np.float32(src)
    dst = [(0, 0), (size, 0), (0, size), (size, size)]
    dst = np.float32(dst)
    M = cv.getPerspectiveTransform(src, dst)
    
    warped = cv.warpPerspective(im_src, M, (size, size))
    
    return warped

