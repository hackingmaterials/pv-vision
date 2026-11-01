"""Tests for module-corner helpers."""
import numpy as np

from pv_vision.transform_crop import _perspective_transform as pt


def test_sort_corners_order():
    # shuffled corners of a 100x50 rectangle, as (x, y)
    corners = np.array([[100, 50], [0, 0], [0, 50], [100, 0]])
    out = pt.sort_corners(corners)
    np.testing.assert_array_equal(out, [[0, 0], [100, 0], [0, 50], [100, 50]])


def test_fill_polygon_fills_inside():
    pts = np.array([[2, 2], [7, 2], [7, 7], [2, 7]], dtype=np.int32)
    mask = pt.fill_polygon(pts, (10, 10))
    assert mask.shape == (10, 10, 1)
    assert mask[5, 5, 0] == 255
    assert mask[0, 0, 0] == 0
