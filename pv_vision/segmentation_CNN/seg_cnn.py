import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import zlib, base64
from scipy import signal
import os
import json


def base64_2_mask(s):
   z = zlib.decompress(base64.b64decode(s))
   n = np.fromstring(z, np.uint8)
   mask = cv.imdecode(n, cv.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
   return mask


def load_mask(path, image, mask_name='module_unet'):
    with open(path, 'r') as file:
        data = json.load(file)
    if len(data["objects"]) == 1: 
        code = data["objects"][0]["bitmap"]["data"]
        origin = data["objects"][0]["bitmap"]["origin"]
    else:
        for obj in data["objects"]:
            if obj['classTitle'] == mask_name:
                code = obj["bitmap"]["data"]
                origin = obj["bitmap"]["origin"]
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

    return mask5.astype('uint8'), mask_center.astype(int)


def find_cross(part, houghlinePara=50):
    edge = cv.Canny(part, 0, 1)
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

    k1 = -1/np.tan(theta1)
    k2 = -1/np.tan(theta2)
    b1 = rho1*np.sin(theta1)-k1*rho1*np.cos(theta1)
    b2 = rho2*np.sin(theta2)-k2*rho2*np.cos(theta2)

    x_cross = (b2-b1) / (k1-k2)
    y_cross = (k1 * b2 - k2 * b1) / (k1 - k2)
    # return thetas1, thetas2
    return x_cross, y_cross


def find_module_corner(mask, mask_center, dist=200, displace=0, method=0,
                       corner_center=False, center_displace=10):

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
            x_cross, y_cross = find_cross(part)
            xs.append(x_cross)
            ys.append(y_cross)

    # sort out the corners    
    xs[1] += x_m-center_displace
    ys[2] += y_m-center_displace
    xs[3] += x_m-center_displace
    ys[3] += y_m-center_displace

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


def perspective_transform(image, src, sizex, sizey):
    src = np.float32(src)
    if np.sum((src[0] - src[2])**2) <= np.sum((src[0] - src[1])**2):
        dst = np.float32([(0, 0), (sizex, 0), (0, sizey), (sizex, sizey)])
    else:
        dst = np.float32([(0, sizey), (0, 0), (sizex, sizey), (sizex, 0)])
    M = cv.getPerspectiveTransform(src, dst)

    warped = cv.warpPerspective(image, M, (sizex, sizey))

    return warped


def find_cell_corner(wrap, dist=25):
    wrap_g = cv.cvtColor(wrap, cv.COLOR_BGR2GRAY)

    sum_x = np.sum(wrap_g, axis=0)
    sum_x = sum_x / np.max(sum_x)
    peak_x, _ = signal.find_peaks(-sum_x, distance=dist, prominence=0.08)

    sum_y = np.sum(wrap_g, axis=1)
    sum_y = sum_y / np.max(sum_y)
    peak_y, _ = signal.find_peaks(-sum_y, distance=dist, prominence=0.08)

    return peak_x, peak_y


def find_corner_mean(wrap, x, y, displace=7):
    try:
        x_l = x - displace
        x_r = x + displace
        y_u = y - displace
        y_d = y + displace

        xs = [x_l, x_r, x_l, x_r]
        ys = [y_u, y_u, y_d, y_d]
        crop = perspective_transform(wrap, list(zip(xs, ys)), 
                                     displace*2, displace*2)
        crop_g = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)

        corners = cv.goodFeaturesToTrack(crop_g, 4, 0.01, 1)
        corners = np.int0(corners)
        points_x = []
        points_y = []

        for i in corners:
            x_c, y_c = i.ravel()
            points_x.append(x_c)
            points_y.append(y_c)

        x_mean = np.mean(points_x)
        y_mean = np.mean(points_y)
        x_mean += x_l
        y_mean += y_u
    except:
        return x, y
    else:
        return x_mean, y_mean


def crop_cell(wrap, peak_x, peak_y, saveplace, plot_bool=0, data_bool=0):
    if plot_bool:
        plt.imshow(cv.cvtColor(wrap, cv.COLOR_BGR2RGB))

    data = []
    # n = 0
    for i in range(len(peak_x)+1):
        if i == 0:
            x_l = 0
            x_r = peak_x[i]
        elif i == len(peak_x):
            x_l = peak_x[i-1]
            x_r = wrap.shape[1]
        else:
            x_l = peak_x[i-1]
            x_r = peak_x[i]

        for j in range(len(peak_y)+1):
            if j == 0:
                y_u = 0
                y_d = peak_y[j]
            elif j == len(peak_y):
                y_u = peak_y[j-1]
                y_d = wrap.shape[0]
            else: 
                y_u = peak_y[j-1]
                y_d = peak_y[j]
            xs = []
            ys = []
            for y_c in [y_u, y_d]:
                for x_c in [x_l, x_r]:
                    x_mean, y_mean = find_corner_mean(wrap, x_c, y_c)

                    xs.append(x_mean)
                    ys.append(y_mean)
            if plot_bool:
                plt.scatter(xs, ys)
            if data_bool:
                data.append(list(zip(xs, ys)))

            crop = perspective_transform(wrap, list(zip(xs, ys)), 64, 64)
            cv.imwrite(saveplace+str(i) + "_" + str(j) + ".png", crop)
            #n += 1
    if data_bool:
        return np.array(data)


# a new version of finding the module corners
def fill_polygon(points, im_shape):
    im_cnt = np.zeros((im_shape[0],im_shape[1],1), np.uint8)
    cv.fillPoly(im_cnt, [points], (255,255))

    return im_cnt


def sort_corners(corners):
    col_sorted = corners[np.argsort(corners[:,1])]
    a = np.argsort(col_sorted[:2, 0])
    b = np.argsort(col_sorted[2:, 0]) + 2

    return col_sorted[np.hstack((a, b))]


def find_corners(im, n_corners=4, dist_corners=100, displace=3):
    corners_unravel = cv.goodFeaturesToTrack(im, n_corners, 0.01, dist_corners, blockSize=9),
    corners = []
    for corner in corners_unravel[0]:
        corners.append(list(corner.ravel()))
    corners_sorted = sort_corners(np.array(corners))
    corners_displaced = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]) * displace + corners_sorted

    return corners_displaced

def find_module_corner2(mask, mode=0):
    """
    mode == 0: detect corners of the convex of module
    mode == 1: detect corners of the approximated convex of module
    mode == 2: detect corners of the approximated contour of the module
    mode == 3: detect corners of the blurred mask of the module
    """
    blur = cv.blur(mask, (6,6))
    contours, _ = cv.findContours(blur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt_approx = cv.approxPolyDP(contours[0], 8, True)
    convex = cv.convexHull(contours[0])
    conv_approx = cv.approxPolyDP(convex, 8, True)

    if mode == 0:
        im_conv = fill_polygon(convex, blur.shape)
        corners = find_corners(im_conv)
    elif mode == 1:
        im_conv_app = fill_polygon(conv_approx, blur.shape)
        corners = find_corners(im_conv_app)
    elif mode == 2:
        im_cnt_app = fill_polygon(cnt_approx, blur.shape)
        corners = find_corners(im_cnt_app)
    elif mode == 3:
        corners = find_corners(blur)
    else:
        print("mode must be one of 0, 1, 2")
        return
    return corners
