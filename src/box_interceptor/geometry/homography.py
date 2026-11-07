"""Гомография: как перевести точку с картинки камеры на карту пола (вид сверху).

Идея. Пол — это плоскость. Любые две "фотографии" одной плоскости связаны
матрицей 3x3, которая называется гомографией H. Если знать H, то точку (u, v)
на кадре можно пересчитать в точку (x, y) на полу:

    [x', y', w']^T = H @ [u, v, 1]^T      # однородные координаты
    x = x' / w',  y = y' / w'             # делим на w' — "перспективное деление"

Работает только для точек, которые ЛЕЖАТ на полу. Поэтому у человека берём
нижнюю середину рамки — точку между ступнями, а не центр рамки.
"""

from __future__ import annotations

import numpy as np


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Применяет гомографию к массиву точек формы (N, 2). Возвращает (N, 2).

    То же самое делает cv2.perspectiveTransform, но здесь видно, что внутри.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])  # (u, v) -> (u, v, 1)
    mapped = homogeneous @ np.asarray(homography, dtype=np.float64).T
    w = mapped[:, 2:3]
    # Точки "за горизонтом" дают w около нуля — это не пол, такие отбрасываем позже.
    with np.errstate(divide="ignore", invalid="ignore"):
        return mapped[:, :2] / w


def invert(homography: np.ndarray) -> np.ndarray:
    """Обратная гомография: вид сверху -> картинка камеры."""
    return np.linalg.inv(np.asarray(homography, dtype=np.float64))
