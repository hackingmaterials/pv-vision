import warnings

from pv_vision.crack_analysis import _feature_extraction as isolate
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.stats import mode


class CrackCell:
    def __init__(self, img_path, ann_path, crack_inx, busbar_inx, busbar_num):
        """extract crack features from cell images.

        Args:
        ------
        img_path: str
            path to the raw image

        ann_path: str
            path to the annotation image
        
        crack_inx: int
            index value of crack in the annotation image

        busbar_inx: int
            index value of busbar in the annotation image

        busbar_num: int
            number of busbars in the cell
        """
        self.img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        self.masks = cv.imread(ann_path, cv.IMREAD_GRAYSCALE)
        self.crack = (self.masks == crack_inx).astype(np.uint8)
        self.busbar = (self.masks == busbar_inx).astype(np.uint8)
        self._merge = self.crack + self.busbar
        self._merge[self._merge > 1] = 1

        self.busbar_num = busbar_num

        self.ske_crack = self._extract_crack_skeleton()
        self.ske_busbar = self._extract_busbar_skeleton()
        self.ske_merge = self._merge_crack_busbar()

        self.inactive_area = None
        self.features = {}

    def _extract_crack_skeleton(self):
        """Extract crack skeleton"""
        ske_crack = isolate.skeleton_crack(self.crack)
        return ske_crack

    def _extract_busbar_skeleton(self):
        """Extract busbar skeleton"""
        ske_busbar = isolate.extend_busbar(self.busbar)
        busbar_num = mode(np.sum(ske_busbar, axis=0), keepdims=False).mode
        if busbar_num != self.busbar_num:
            raise ValueError("Busbar number is not correct")

        return ske_busbar

    def _merge_crack_busbar(self):
        """Merge crack and busbar skeletons"""
        ske_merge = self.ske_crack + self.ske_busbar
        ske_merge[ske_merge > 1] = 1
        return ske_merge

    def extract_inactive_area(self):
        """Extract inactive area"""
        self.inactive_area, self.features["inactive_prop"] = isolate.detect_inactive(self.crack, self.busbar)
        return self.inactive_area, self.features["inactive_prop"]

    def extract_crack_length(self):
        """Extract crack length (din pixel)"""
        self.features["crack_length"] = self.ske_crack.sum()
        return self.features["crack_length"]

    def extract_brightness(self, mode, norm=255):
        """Extract brightness
        mode: str
        if mode == "avg_all": average brightness of the raw image
        if mode == "avg_inactive": average brightness of the inactive area and treat the active area as 1
        if mode == "avg_inactive_only": average brightness of the inactive area only and return 1 for intact cells

        norm: int
        normalization factor
        """
        if mode == "avg_all":
            self.features["brightness_cell"] = isolate.avg_grayscale(self.img, norm=norm)
            return self.features["brightness_cell"]
        elif mode == "avg_inactive":
            if self.inactive_area is None:
                warnings.warn("Inactive area is not extracted. Extracting now...")
                self.extract_inactive_area()
            self.features["brightness_inactive"] = isolate.avg_grayscale2(self.img, self.inactive_area, norm=norm)
            return self.features["brightness_inactive"]
        elif mode == "avg_inactive_only":
            if self.inactive_area is None:
                warnings.warn("Inactive area is not extracted. Extracting now...")
                self.extract_inactive_area()
            self.features["brightness_inactive_only"] = isolate.avg_grayscale3(self.img, self.inactive_area, norm=norm)
            return self.features["brightness_inactive_only"]

    def extract_features(self, mode="avg_inactive_only", norm=255):
        """Extract all features"""
        self.extract_inactive_area()
        self.extract_crack_length()
        self.extract_brightness(mode=mode, norm=norm)
        return self.features

    def plot(self):
        """plot raw image, masks, skeletons and inactive area"""
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes[0, 0].imshow(self.img, cmap="gray")
        axes[0, 0].set_title("Raw image")
        axes[0, 1].imshow(self._merge, cmap="gray")
        axes[0, 1].set_title("Masks")
        axes[1, 0].imshow(self.ske_merge, cmap="gray")
        axes[1, 0].set_title("Skeleton")
        axes[1, 1].imshow(self.inactive_area, cmap="gray")
        axes[1, 1].set_title("Inactive area")
        plt.tight_layout()
        plt.show()