import numpy as np
import pytest

from box_interceptor.geometry.birdseye import (
    CameraProjector,
    fuse_points,
    inside_bounds,
    match_points,
    points_in_polygon,
)
from box_interceptor.geometry.epfl import TopViewGrid, load_calibration, load_ground_truth
from box_interceptor.geometry.homography import invert, project_points

# Гомография "масштаб x2 + сдвиг (10, 20)"
H_SCALE_SHIFT = np.array([[2, 0, 10], [0, 2, 20], [0, 0, 1]], dtype=float)


def test_project_identity():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(project_points(pts, np.eye(3)), pts)


def test_project_scale_shift_and_inverse_roundtrip():
    pts = np.array([[0.0, 0.0], [5.0, 5.0]])
    mapped = project_points(pts, H_SCALE_SHIFT)
    np.testing.assert_allclose(mapped, [[10, 20], [20, 30]])
    np.testing.assert_allclose(project_points(mapped, invert(H_SCALE_SHIFT)), pts)


def test_project_perspective_division():
    # Третья строка [0, 0, 2] делит результат пополам — так работает "w"
    h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float)
    np.testing.assert_allclose(project_points([[4, 6]], h), [[2, 3]])


def test_project_empty():
    assert project_points(np.empty((0, 2)), np.eye(3)).shape == (0, 2)


CALIB_TEXT = """# header
##########
# Camera 0
##########

# Ground plane homography
1 0 0
0 1 0
0 0 1

# Head plane homography
2 0 0
0 2 0
0 0 1

##########
# Camera 1
##########
# Ground plane homography
2	0	10
0	2	20
0	0	1
# Head plane height in camera view
10.5
"""


def test_load_calibration(tmp_path):
    f = tmp_path / "calib.txt"
    f.write_text(CALIB_TEXT)
    cams = load_calibration(f)
    assert [c.camera for c in cams] == [0, 1]
    np.testing.assert_allclose(cams[0].ground_h, np.eye(3))
    assert cams[0].head_h is not None
    np.testing.assert_allclose(cams[1].ground_h, H_SCALE_SHIFT)
    assert cams[1].head_h is None


def test_grid_cell_center():
    grid = TopViewGrid(grid_w=4, grid_h=2, origin=(100.0, 0.0), size=(40.0, 20.0))  # клетка 10x10
    np.testing.assert_allclose(grid.cell_center(0), [105, 5])
    np.testing.assert_allclose(grid.cell_center(3), [135, 5])  # конец первой строки
    np.testing.assert_allclose(grid.cell_center(4), [105, 15])  # начало второй строки
    assert grid.bounds == (100, 0, 140, 20)


def test_load_ground_truth(tmp_path):
    f = tmp_path / "gt.txt"
    # 3 кадра, 2 человека, сетка 4x2, шаг 2, кадры 0..2
    f.write_text("1\n3 2 4 2 2 0 2\n0 -1\n-1 -1\n5 -2\n")
    grid = TopViewGrid(4, 2, (0.0, 0.0), (40.0, 20.0))
    gt = load_ground_truth(f, grid)
    assert gt.labeled_frames == [0, 2]
    np.testing.assert_allclose(gt.people_at(0), [[5, 5]])
    np.testing.assert_allclose(gt.people_at(2), [[15, 15]])
    assert gt.people_at(1).shape == (0, 2)
    assert gt.people_at(99).shape == (0, 2)


def test_load_ground_truth_grid_mismatch(tmp_path):
    f = tmp_path / "gt.txt"
    f.write_text("1\n1 1 4 2 1 0 0\n0\n")
    with pytest.raises(ValueError, match="сетка"):
        load_ground_truth(f, TopViewGrid(5, 5, (0, 0), (1, 1)))


def test_camera_projector_feet_and_scale():
    proj = CameraProjector(camera=0, ground_h=np.eye(3), scale=2.0)
    boxes = np.array([[10, 0, 30, 50]])  # ступни в (20, 50)
    np.testing.assert_allclose(proj.feet(boxes), [[20, 50]])
    np.testing.assert_allclose(proj.to_floor(boxes), [[40, 100]])  # x2 из-за scale
    np.testing.assert_allclose(proj.floor_to_image([[40, 100]]), [[20, 50]])


def test_fuse_merges_across_cameras():
    cam0 = np.array([[0.0, 0.0]])
    cam1 = np.array([[1.0, 1.0]])
    centers, support = fuse_points([cam0, cam1], radius=5)
    np.testing.assert_allclose(centers, [[0.5, 0.5]])
    assert support.tolist() == [2]


def test_fuse_keeps_two_close_people_from_same_camera():
    # Одна камера видит двух людей рядом — это два человека, а не один
    cam0 = np.array([[0.0, 0.0], [2.0, 0.0]])
    centers, _ = fuse_points([cam0], radius=5)
    assert len(centers) == 2


def test_fuse_far_points_stay_separate_and_empty_input():
    centers, _ = fuse_points([np.array([[0.0, 0.0]]), np.array([[100.0, 0.0]])], radius=5)
    assert len(centers) == 2
    centers, support = fuse_points([np.empty((0, 2))], radius=5)
    assert centers.shape == (0, 2) and support.shape == (0,)


def test_match_points():
    gt = np.array([[0.0, 0.0], [10.0, 0.0], [50.0, 50.0]])
    pred = np.array([[1.0, 0.0], [10.0, 2.0], [100.0, 100.0]])
    r = match_points(pred, gt, radius=3)
    assert (r.tp, r.fp, r.fn) == (2, 1, 1)
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == pytest.approx(2 / 3)
    total = r + match_points(np.empty((0, 2)), gt, radius=3)
    assert (total.tp, total.fp, total.fn) == (2, 1, 4)


def test_inside_bounds_and_polygon():
    pts = np.array([[1, 1], [5, 5], [np.nan, 1]])
    assert inside_bounds(pts, (0, 0, 2, 2)).tolist() == [True, False, False]
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    assert points_in_polygon(np.array([[1, 1], [3, 3]]), square).tolist() == [True, False]
