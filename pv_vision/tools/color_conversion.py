import numpy as np
import cv2 as cv
import pandas as pd


def load_colormap(map_path, gray_col, rgb_col, convert):
    """Output the colormap between grayscale and RGB

    Parameters
    ----------
    map_path: str or pathlib.PosixPath
    Path of colormap file

    gray_col: str
    Name of grayscale value column

    rgb_col: list
    List of names of RGB value column

    convert: int [0 or 1]
    If 0: rgb2gray
    If 1: gray2rgb

    Returns
    -------
    convert_dict: dict
    Mapping dictionary
    """
    colormap = pd.read_csv(map_path)
    grayscale = colormap[gray_col]
    rgb = colormap[rgb_col]

    convert_dict = {}
    if convert == 0:
        # rgb2gray
        for i in range(len(rgb)):
            convert_dict[str(list(rgb.iloc[i].to_numpy()))] = grayscale.iloc[i]
    elif convert == 1:
        # gray2rgb
        for i in range(len(grayscale)):
            convert_dict[grayscale.iloc[i]] = rgb.iloc[i].to_numpy()

    return convert_dict


def convert_rgb2gray(image, convert_dic):
    """convert rgb image to grayscale

    Parameters
    ----------
    image: array
    RGB image. Channel order should be RGB.

    convert_dic: dict
    dictionary key is str(rgb list), value is grayscale value

    Returns
    -------
    image_gray: array
    Grayscale image
    """
    image_r = image[:, :, 0]
    image_g = image[:, :, 1]
    image_b = image[:, :, 2]
    im_shape = image_r.shape

    image_gray = np.zeros(im_shape)

    for i in range(im_shape[0]):
        for j in range(im_shape[1]):
            image_gray[i, j] = convert_dic[str([image_r[i, j], image_g[i, j], image_b[i, j]])]

    return image_gray


def convert_gray2rgb(image, convert_dic):
    """convert rgb image to grayscale

        Parameters
        ----------
        image: array
        Grayscale image. Only 1 channel.

        convert_dic: dict
        dictionary key is int(grayscale value), value is rgb array

        Returns
        -------
        image_rgb: array
        RGB image
        """
    im_shape_0 = image.shape[0]
    im_shape_1 = image.shape[1]
    image_rgb = np.zeros((im_shape_0, im_shape_1, 3))

    for i in range(im_shape_0):
        for j in range(im_shape_1):
            image_rgb[i, j, :] = convert_dic[image[i, j]]

    return image_rgb


def remove_repeated_channel(image):
    """Remove repeated channel in grayscale images.
       Opencv read grayscale images into 3-channel if not specified, so sometimes people may output a 3-channel
       grayscale image.

    Parameters
    ----------
    image: array
    Grayscale images with 3 repeated channels

    Returns
    -------
    image: array
    Grayscale images with 1 channel
    """
    return image[:, :, 0]

