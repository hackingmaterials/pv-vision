import pv_vision.transform_seg.perspective_transform as transform
import pv_vision.transform_seg.cell_crop as seg
import numpy as np
import cv2 as cv
from pathlib import Path


class SolarModule:
    """Parent class. This class provide basic methods of processing a module image"""
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


class MaskModule(SolarModule):
    """Raw module image class that has a mask of target modules"""
    def __init__(self, image, row, col):
        super().__init__(image, row, col)
        self._mask = None
        self._mask_center = None
        self._corners = None

    @property
    def mask(self):
        if self._mask is None:
            print("Mask is not loaded")

        return self._mask

    @property
    def mask_center(self):
        if self._mask_center is None:
            print("Mask is not loaded")

        return self._mask_center

    @property
    def corners(self):
        if self._corners is None:
            print("Corners are not detected")

        return self._corners

    def load_mask(self, mask_path, output=False):
        self._mask, self._mask_center = \
            transform.load_mask(mask_path, self._image, 'module_unet')

        if output:
            return self._mask, self._mask_center

    def corner_detection_line(self, dist=200, displace=0, method=0,
                              corner_center=False, center_displace=10, output=False):
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

        output: bool
        If true, return the corners

        Returns
        -------
        Corners: array
        Sorted coordinates of module corners. The order is top-left, top-right, bottom-left and bottom-right
        """
        if self._mask is None:
            print("Mask is not loaded")
            return None
        self._corners = transform.find_module_corner(self._mask, self._mask_center,
                                               dist, displace, method, corner_center, center_displace)
        if output:
            return self._corners

    def corner_detection_cont(self, mode=0, output=False):
        """ Detect the corners of a solar module

        Parameters
        ----------
        mode: int
        mode == 0: detect corners of the convex of module
        mode == 1: detect corners of the approximated convex of module
        mode == 2: detect corners of the approximated contour of the module
        mode == 3: detect corners of the blurred mask of the module

        output: bool
        If true, return the corners

        Returns
        -------
        Corners: array
        Corners of the solar module
        """
        if self._mask is None:
            print("Mask is not loaded")
            return None
        self._corners = transform.find_module_corner2(self._mask, mode)
        if output:
            return self._corners

    def transform(self, width=600, height=300, img_only=False):
        """Do perspective transform on the solar module

        Parameters
        ----------
        width, height: int
        Width/height of transformed image

        img_only: bool
        If true, only return the image of transformed module. Otherwise return a transformed module instance

        Returns
        -------
        wrap: array or instance
        Transformed solar module
        """
        if self._corners is None:
            print("Corners are not detected")
        else:
            wrap = transform.perspective_transform(self._image, self._corners, width, height)
            if img_only:
                return wrap
            else:
                return TransformedModule(wrap, self._row, self._col)


class TransformedModule(SolarModule):
    """Solar module class after perspective transformation"""
    def __init__(self, image, row, col):
        super().__init__(image, row, col)
        if len(self._size) != 2:
            super().remove_channel(in_place=True)
    
    @staticmethod
    def is_transformed(image, x_min, y_min):
        """Determine whether the module is properly transformed by checking the number of internal edges.

        Parameters
        ----------
        image: array
        The image that needs to be checked

        x_min, y_min: int
        The threshold of the number of detected internal edges. X means col and y means row

        Returns
        -------
        bool
        """
        peak_x, peak_y = transform.find_inner_edge(image)
        return (len(peak_x) >= x_min) and (len(peak_y) >= y_min)

    def crop_cell(self):
        splits_inx_x, edgelist_x = seg.detect_edge(self._image, peaks_on=0)
        abs_x = seg.linear_regression(splits_inx_x, edgelist_x)
        splits_inx_y, edgelist_y = seg.detect_edge(self._image, peaks_on=1)
        abs_y = seg.linear_regression(splits_inx_y, edgelist_y)

        abs_x_couple = seg.couple_edges(abs_x, length=self.size[1])
        abs_y_couple = seg.couple_edges(abs_y, length=self.size[0])

        return np.array(seg.segment_cell(self.image, abs_x_couple, abs_y_couple))

    def classify_cells(self, ann_path, defects_inx_dic):
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
        return seg.classify_cells(ann_path, defects_inx_dic, row_col=self.row_col, shape=self.size)

    @staticmethod
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
        seg.write_cells(single_cells, defects_inx, defect2folder, name, save_path)






