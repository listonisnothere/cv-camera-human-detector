import numpy as np

from box_interceptor.video.io import FfmpegWriter, FrameSource
from box_interceptor.video.mosaic import mosaic


def test_mosaic_grid_shape():
    frames = [np.full((10, 20, 3), i * 50, dtype=np.uint8) for i in range(3)]
    m = mosaic(frames, cols=2)
    assert m.shape == (20, 40, 3)  # 2 ряда x 2 колонки
    assert m[15, 30].tolist() == [0, 0, 0]  # 4-я ячейка пустая


def test_mosaic_resizes_to_tile():
    frames = [np.zeros((100, 100, 3), np.uint8), np.zeros((50, 80, 3), np.uint8)]
    assert mosaic(frames, cols=2, tile_size=(32, 24)).shape == (24, 64, 3)


def test_writer_and_source_roundtrip(tmp_path):
    path = tmp_path / "out.mp4"
    with FfmpegWriter(path, fps=10, size=(64, 48)) as w:
        for i in range(5):
            w.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    with FrameSource(path) as src:
        assert (src.width, src.height) == (64, 48)
        frames = list(src)
    assert len(frames) == 5
    # H.264 сжимает с потерями, но средняя яркость должна сохраниться
    assert abs(float(frames[-1].mean()) - 160) < 10
