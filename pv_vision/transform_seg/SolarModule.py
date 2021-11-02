import pv_vision.transform_seg.perspective_transform as seg
import numpy as np
import cv2 as cv


class SolarModule:
    def __init__(self, image, row, col):
        self._image = image
        self._size = image.shape
        self._row = row
        self._col = col

    @property
    def image(self):
        """The EL image of the module"""
        return self._image

    @image.setter
    def image(self, new_image):
        self._image = new_image
        self._size = new_image.shape

    @property
    def size(self):
        """The size of the image"""
        return self._size

    @property
    def row_col(self):
        """The number of row and col of the module"""
        return [self._row, self._col]

    @row_col.setter
    def row_col(self, row, col):
        self._col = col
        self._row = row

    def _reset(self, new_image, new_row, new_col):
        """Reset the image in this instance"""
        self._image = new_image
        self._size = new_image.shape
        self._row = new_row
        self._col = new_col

    def resize(self, new_size, in_place=False):
        """Resize the image

        Parameters
        ----------
        new_size: list or tuple
        [height, width] of the resized image

        in_place: bool
        If true, change will be applied to current instance

        Returns
        -------
        image_r: array
        Resized image
        """
        image_r = cv.resize(self._image, (new_size[1], new_size[0]))
        if in_place:
            self._reset(image_r, self._row, self._col)
        else:
            return image_r

    def rotate(self, rotate_angle, in_place=False):
        """ rotate input images through 90 or 180 degrees

        Parameters
        ----------
        rotate_angle: int, one of [0, 1, 2]
            The angle of rotation.
            0 = 90-clockwise,
            1 = 180
            2 = 90-counter-clockwise

        in_place: bool
        If true, change will be applied to current instance

        Returns
        -------
        image_r: array
        The rotated image.
        """
        image_r = cv.rotate(self._image, rotate_angle)
        if in_place:
            if rotate_angle == 1:
                self._reset(image_r, self._row, self._col)
            else:
                self._reset(image_r, self._col, self._row)
        else:
            return image_r

    def remove_channel(self, in_place=False):
        """Remove repeated channel in grayscale images.
           Opencv read grayscale images into 3-channel if not specified, so sometimes people may output a 3-channel
           grayscale image.

        Parameters
        ----------
        in_place: bool
        If true, change will be applied to current instance

        Returns
        -------
        image: array
        Grayscale images with 1 channel
        """
        if len(self._size) == 2:
            print("It is already 1-channel")
        else:
            if in_place:
                self._reset(self._image[:, :, 0], self._row, self._col)
            else:
                return self._image[:, :, 0]

    def copy_channel(self, in_place=False):
        """Duplicate the grayscale channel and expand to 3 channels

        Parameters
        ----------
        in_place: bool
        If true, change will be applied to current instance

        Returns
        -------
        image: array
        Grayscale images with 3 channels
        """
        if len(self._size) == 3:
            print("It is already 3-channel")
        else:
            image_r = cv.merge((self._image, self._image, self._image))
            if in_place:
                self._reset(image_r, self._row, self._col)
            else:
                return image_r

    def save_fig(self, save_path):
        """Save the image file

        Parameters
        ----------
        save_path: str or pathlib.PosixPath
        The folder path of the original images.
        """
        cv.imwrite(str(save_path), self._image)



