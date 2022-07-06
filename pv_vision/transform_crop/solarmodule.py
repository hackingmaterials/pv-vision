import pv_vision.transform_crop.perspective_transform as transform
import pv_vision.transform_crop.cell_crop as seg
import numpy as np
import cv2 as cv
from pathlib import Path


class AbstractModule:
    """Parent class. This class provide basic methods of processing a module image"""
    def __init__(self, image, row, col, busbar):
        self._image = image
        self._size = image.shape
        self._row = row
        self._col = col
        self._busbar = busbar

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
    def row(self):
        """The number of row and col of the module"""
        return self._row

    @row.setter
    def row(self, row):
        self._row = row

    @property
    def col(self):
        """The number of row and col of the module"""
        return self._col

    @col.setter
    def col(self, col):
        self._col = col

    @property
    def busbar(self):
        """The number of busbars"""
        return self._busbar

    @busbar.setter
    def busbar(self, busbar):
        self._busbar = busbar

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


class SplitModule(AbstractModule):
    """Raw module image class that use splitting to crop cells"""
    def __init__(self, image, row, col, busbar):
        super().__init__(image, row, col, busbar)
        self._cells = None
        self._vline_abs = None
        self._hline_abs = None

    @property
    def cells(self):
        if self._cells is None:
            print("Cells are not cropped")

        return self._cells

    def crop_cell(self, cellsize, hsplit=100, hthre=0.8, hinterval=100, hmargin=None, vsplit=50, vthre=0.6, vinterval=250, vmargin=None, savepath=None, displace=None):
        """Crop cells

        Parameters
        ----------
        cellsize: int
        Edge size of a cell. This can be an estimated value from raw module image.

        hsplit, vplit: int
        Number of horizontal/vertical splits

        hthre, vthre: float
        Peaks will be set as 1 above this threshold

        savepath: str
        Save path

        displace: int
        Displace the detected lines

        Returns
        -------
        cells: array
        """

        image_thre = transform.image_threshold(self.image, adaptive=True)
        image_thre = cv.resize(image_thre,(4000,2500))
        self._vline_abs = seg.detect_vertical_lines(image_thre, cell_size=cellsize,
                                              column=self.col, thre=hthre, split=hsplit, peak_interval=vinterval, margin=vmargin)
        self._hline_abs = seg.detect_horizon_lines(image_thre, row=self.row, busbar=self.busbar,
                                             cell_size=cellsize, thre=vthre, split=vsplit, peak_interval=hinterval, margin=hmargin)

        self._cells = seg.segment_cell(self.image, self._hline_abs, self._vline_abs, cellsize, savepath, displace)

        return self._cells

    def plot_edges(self, linewidth=1):
        seg.plot_edges(self.image, self._hline_abs, self._vline_abs, linewidth)


class MaskModule(AbstractModule):
    """Raw module image class that has a mask of target modules"""
    def __init__(self, image, row, col, busbar):
        super().__init__(image, row, col, busbar)
        self._mask = None
        self._mask_center = None
        self._corners = None
        self._transformed = None
        self._cells = None

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

    @property
    def transformed_image(self):
        if self._transformed is None:
            print("Module is not transformed")

        return self._transformed

    @property
    def cells(self):
        if self._cells is None:
            print("Cells are not cropped")

        return self._cells

    def load_mask(self, mask_path=None, output=False, blur=5, thre=None):
        """Load the mask of a module.
        If the background of the image looks good, you don't need to provide masks explicitly.
        Otherwise, you should provide masks predicted from neural networks

        Parameters
        ----------
        mask_path: str
        If provided, then mask is loaded explicitly. Otherwise, this method will detect masks automatically.

        output: bool
        If true, return mask and mask center

        blur: int
        Blur filter size in cv.medianBlur

        thre: int or float
        Threshold value in cv.threshold.
        Pixels above this value will be set 255, and below will be 0.
        If threshold < 1, first threshold (%) of the image grayscale value will be used.
        If not given, first 10% of the image grayscale value will be used.
        """
        if mask_path is None:
            self._mask = transform.image_threshold(image=self.image, blur=blur, threshold=thre)
        else:
            self._mask, self._mask_center = \
                transform.load_mask(mask_path, self._image, 'module_unet')

        if output:
            return self._mask

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

    def transform(self, width=None, height=None, cellsize=None, auto_rotate=True, img_only=True):
        """Do perspective transform on the solar module

        Parameters
        ----------
        width, height: int
        Width/height of transformed image

        cellsize: int
        Edge length of a cell

        auto_rotate: bool
        If true, automatically adjust module orientation such that shorter side is vertically aligned.

        img_only: bool
        If true, only return the image of transformed module.
        Otherwise return a transformed module instance

        Returns
        -------
        wrap: array or instance
        Transformed solar module
        """
        if self._corners is None:
            print("Corners are not detected. Start automatic detection")
            return None
            #self._corners = self.corner_detection_cont(mode=1)

        if cellsize:
            width = self.col * cellsize
            height = self.row * cellsize
        wrap = transform.perspective_transform(self._image, self._corners, width, height, rotate=auto_rotate)
        self._transformed = TransformedModule(wrap, self._row, self._col, self._busbar)
        if img_only:
            return wrap
        else:
            return TransformedModule(wrap, self._row, self._col, self._busbar)

    def is_transformed(self, x_min, y_min):
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
        res = self._transformed.is_transformed(x_min, y_min)
        return res

    def crop_cell(self, cellsize, simple=False, vl_interval=None, vl_split_size=None,
                  hl_interval=None, hl_split_size=None, margin=None):
        cells = self._transformed.crop_cell(cellsize, simple, vl_interval, vl_split_size,
                  hl_interval, hl_split_size, margin)

        return cells


class TransformedModule(AbstractModule):
    """Solar module class after perspective transformation"""

    def __init__(self, image, row, col, busbar):
        super().__init__(image, row, col, busbar)
        if len(self._size) != 2:
            super().remove_channel(in_place=True)

    # @staticmethod
    def is_transformed(self, x_min, y_min):
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
        peak_x, peak_y = transform.find_inner_edge(self._image)
        return (len(peak_x) >= x_min) and (len(peak_y) >= y_min)

    def crop_cell(self, cellsize, simple=False, vl_interval=None, vl_split_size=None,
                  hl_interval=None, hl_split_size=None, margin=None):
        if simple:
            vline_abs = list(zip(np.zeros(self.col - 1), 
                np.linspace(0, self.size[1], self.col + 1)[1: -1].astype(int)))
            hline_abs = list(zip(np.zeros(self.row - 1), 
                np.linspace(0, self.size[0], self.row + 1)[1: -1].astype(int)))
        else:
            vinx_split, vline_split = seg.detect_edge(self._image, row_col=[self.row, self.col], cell_size=cellsize,
                                                    busbar=self.busbar, peaks_on=0, split_size=vl_split_size,
                                                    peak_interval=vl_interval, margin=margin)
            vline_abs = seg.linear_regression(vinx_split, vline_split)
            hinx_split, hline_split = seg.detect_edge(self._image, row_col=[self.row, self.col], cell_size=cellsize,
                                                    busbar=self.busbar, peaks_on=1, split_size=hl_split_size,
                                                    peak_interval=hl_interval, margin=margin)
            hline_abs = seg.linear_regression(hinx_split, hline_split)

        hline_abs_couple = seg.couple_edges(hline_abs, length=self.size[0])
        vline_abs_couple = seg.couple_edges(vline_abs, length=self.size[1])

        return np.array(seg.segment_cell(self.image, hline_abs_couple, vline_abs_couple, cellsize=cellsize))

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