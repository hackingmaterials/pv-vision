"""Tests for the geometry helpers used to crop cells from a module image."""
import numpy as np
import pytest

from pv_vision.transform_crop import _cell_crop as cc


def test_linear():
    assert cc.linear(2.0, 3.0, 1.0) == pytest.approx(7.0)


def test_linear_regression_recovers_slope_and_intercept():
    x = np.arange(10, dtype=float)
    lines = [2.0 * x + 1.0, -0.5 * x + 4.0]
    ab = cc.linear_regression(x, lines)
    assert ab.shape == (2, 2)
    np.testing.assert_allclose(ab[0], [2.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(ab[1], [-0.5, 4.0], atol=1e-6)


def test_filter_margin_drops_edges_near_borders():
    edges = np.array([5, 100, 200, 395])
    out = cc.filter_margin(edges, im_length=400, margin=20)
    np.testing.assert_array_equal(out, [100, 200])


def test_filter_margin_keeps_inner_edges():
    edges = np.array([50, 100, 200, 350])
    np.testing.assert_array_equal(cc.filter_margin(edges, 400, 20), edges)


@pytest.mark.parametrize("direction, expected_shape", [(0, (10, 40)), (1, (30, 10))])
def test_split_img_by_size(direction, expected_shape):
    image = np.zeros((30, 40))
    splits = cc.split_img(image, split_size=10, direction=direction)
    assert all(s.shape == expected_shape for s in splits)


def test_split_img_by_count():
    splits = cc.split_img(np.zeros((30, 40)), n_split=3, direction=0)
    assert len(splits) == 3


def test_split_img_rejects_bad_direction():
    with pytest.raises(ValueError):
        cc.split_img(np.zeros((10, 10)), n_split=2, direction=2)


def test_split_img_requires_size_or_count():
    with pytest.raises(ValueError):
        cc.split_img(np.zeros((10, 10)), direction=0)


def test_displace_horizontal_line():
    # y = 0*x + 5 moved by 2 pixels perpendicular -> y = 7
    assert cc.displace((0.0, 5.0), 2) == pytest.approx((0.0, 7.0))


def test_couple_edges_adds_image_borders():
    lines = np.array([[0.0, 50.0]])
    couples = cc.couple_edges(lines, length=100, displacement=0, add_edge=True)
    # borders at 0 and length-1 plus the inner edge -> two neighbouring pairs
    assert couples.shape == (2, 2, 2)
    np.testing.assert_allclose(couples[0], [[0, 0], [0, 50]])
    np.testing.assert_allclose(couples[1], [[0, 50], [0, 99]])


def test_cross_of_axis_aligned_lines():
    # vertical edge x = 0*y + 10 ; horizontal edge y = 0*x + 20
    assert cc.cross((0.0, 10.0), (0.0, 20.0)) == pytest.approx((10.0, 20.0))


def test_perspective_transform_identity_crop():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    src = [(0, 0), (10, 0), (0, 10), (10, 10)]
    out = cc.perspective_transform(image, src, 10)
    assert out.shape == (10, 10)
    np.testing.assert_array_equal(out, image)


def test_coordinate2inx_row_major_order():
    # 8x16 module on a 300x600 image: each cell is 37.5 tall and 37.5 wide
    assert cc.coordinate2inx([0, 0]) == 0
    assert cc.coordinate2inx([37.5, 0]) == 1
    assert cc.coordinate2inx([0, 37.5]) == 16


@pytest.mark.parametrize("n", [1, 2, 3])
def test_detectoutliers_option1_handles_short_input(n):
    # Regression for issue #16: fewer than 4 peak intervals raised IndexError.
    data = np.arange(1, n + 1, dtype=float) * 100
    mask = cc.detectoutliers(data, option=1)
    assert mask.dtype == bool
    assert len(mask) == n + 1


def test_detectoutliers_option1_unchanged_for_long_input():
    data = np.array([100.0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 300])
    mask = cc.detectoutliers(data, option=1)
    assert len(mask) == len(data) + 1
    assert not mask[-1]
    assert mask[:-1].all()
