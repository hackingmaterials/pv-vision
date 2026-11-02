"""End-to-end tests of the module-to-cell cropping pipelines, on the example
images shipped in examples/transform_seg/. Same calls and parameters as
tutorials/transform_contour.ipynb and tutorials/transform_split.ipynb.

Checks are on stable properties (cell count, sizes, corner order), not on
exact pixel values, so they survive OpenCV version changes.
"""
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from pv_vision.transform_crop.solarmodule import MaskModule, SplitModule

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "transform_seg" / "module_imgs"
CELLSIZE = 250

# Inner-edge parameters from tutorials/transform_contour.ipynb
INNER_EDGES = {
    "vl_interval": 150, "vl_thre": 0.9, "vl_split_size": 200, "vl_margin": 200,
    "hl_interval": 200, "hl_thre": 0.9, "hl_split_size": 200, "hl_margin": 100,
}


def _load(name):
    path = EXAMPLES / name
    if not path.exists():
        pytest.skip(f"example image {path} not available")
    image = cv.imread(str(path), 0)
    assert image is not None, f"could not read {path}"
    return image


@pytest.fixture(scope="module")
def mask_module():
    """6 x 12 module, 3 busbars: mask -> corners -> perspective transform."""
    module = MaskModule(_load("example_3.jpg"), 6, 12, 3)
    module.load_mask(thre=0.6)
    module.corner_detection_cont(mode=0)
    module.transform(cellsize=CELLSIZE, img_only=True)
    return module


def test_mask_module_finds_four_ordered_corners(mask_module):
    corners = np.asarray(mask_module.corners)
    assert corners.shape == (4, 2)
    tl, tr, bl, br = corners
    # order is top-left, top-right, bottom-left, bottom-right
    assert tl[0] < tr[0] and bl[0] < br[0]
    assert tl[1] < bl[1] and tr[1] < br[1]


def test_mask_module_transform_has_module_aspect(mask_module):
    # transformed_image returns a TransformedModule; its pixels are .image
    image = mask_module.transformed_image.image
    assert image.shape[:2] == (6 * CELLSIZE, 12 * CELLSIZE)


@pytest.mark.parametrize("simple", [False, True])
def test_mask_module_crops_one_image_per_cell(mask_module, simple):
    cells = mask_module.crop_cell(cellsize=CELLSIZE, inner_edges_para=INNER_EDGES, simple=simple)
    assert cells.shape == (6 * 12, CELLSIZE, CELLSIZE)
    # every cell holds part of the module, not an empty background crop
    assert all(cell.std() > 0 for cell in cells)


def test_split_module_crops_one_image_per_cell():
    """6 x 10 module, cropped by the split method without a mask."""
    module = SplitModule(_load("example_2.jpg"), 6, 10, 3)
    cells = np.asarray(module.crop_cell(cellsize=CELLSIZE, hsplit=100, hthre=0.95, hinterval=50,
                                        vsplit=100, vthre=0.95, vinterval=300))
    assert cells.shape == (6 * 10, CELLSIZE, CELLSIZE)
    assert all(cell.std() > 0 for cell in cells)
