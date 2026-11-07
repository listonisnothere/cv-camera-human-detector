"""Чтение калибровки и ground truth датасета EPFL "Multi-camera pedestrians" (POM).

Калибровка (calibration-*.txt): для каждой камеры гомография плоскости пола
H_ground, которая переводит точку картинки в координаты вида сверху:
    H * X_image = X_topview      (так написано на странице датасета)

Ground truth (gt_*.txt): позиции людей на дискретной сетке пола.
    строка 1:  версия формата (1)
    строка 2:  <кадров> <людей> <ширина сетки> <высота сетки> <шаг> <первый> <последний>
    далее:     по строке на кадр, в ней по числу на человека:
               >= 0 — номер клетки сетки, -1 — не размечено, -2 — человек вне сетки.
Размечен каждый <шаг>-й кадр (обычно каждый 25-й, т.е. раз в секунду).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CAMERA_RE = re.compile(r"^#\s*Camera\s+(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class CameraCalibration:
    camera: int
    ground_h: np.ndarray  # 3x3, картинка -> вид сверху
    head_h: np.ndarray | None  # 3x3 для плоскости голов (1.75 м над полом), если есть


def load_calibration(path: str | Path) -> list[CameraCalibration]:
    """Парсит calibration-*.txt. Возвращает калибровки, отсортированные по номеру камеры."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    cameras: dict[int, dict[str, np.ndarray | None]] = {}
    current: int | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if m := _CAMERA_RE.match(line):
            current = int(m.group(1))
            cameras[current] = {"ground": None, "head": None}
        elif current is not None and line.lower().startswith("# ground plane homography"):
            cameras[current]["ground"] = _read_matrix(lines, i + 1)
            i += 3
        elif current is not None and line.lower().startswith("# head plane homography"):
            cameras[current]["head"] = _read_matrix(lines, i + 1)
            i += 3
        i += 1

    result = []
    for cam, mats in sorted(cameras.items()):
        if mats["ground"] is None:
            raise ValueError(f"{path}: у камеры {cam} нет 'Ground plane homography'")
        result.append(CameraCalibration(cam, mats["ground"], mats["head"]))
    return result


def _read_matrix(lines: list[str], start: int) -> np.ndarray:
    rows = [[float(x) for x in lines[start + k].split()] for k in range(3)]
    matrix = np.array(rows, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Ожидалась матрица 3x3 в строке {start + 1}, получено {matrix.shape}")
    return matrix


@dataclass(frozen=True)
class TopViewGrid:
    """Сетка на полу. Параметры берутся из таблицы на странице датасета."""

    grid_w: int
    grid_h: int
    origin: tuple[float, float]  # координаты угла сетки в системе вида сверху
    size: tuple[float, float]  # ширина и высота области в тех же единицах

    def cell_center(self, pos: np.ndarray) -> np.ndarray:
        """Номер клетки -> координаты её центра. Порт функции grid_to_tv со страницы EPFL."""
        pos = np.asarray(pos)
        x = (pos % self.grid_w + 0.5) * (self.size[0] / self.grid_w) + self.origin[0]
        y = (pos // self.grid_w + 0.5) * (self.size[1] / self.grid_h) + self.origin[1]
        return np.stack([x, y], axis=-1).astype(np.float64)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) области сетки."""
        x0, y0 = self.origin
        return x0, y0, x0 + self.size[0], y0 + self.size[1]


@dataclass(frozen=True)
class GroundTruth:
    grid: TopViewGrid
    step: int
    first_frame: int
    last_frame: int
    positions: np.ndarray  # (кадры, люди), int

    @property
    def labeled_frames(self) -> list[int]:
        return list(range(self.first_frame, self.last_frame + 1, self.step))

    def people_at(self, frame: int) -> np.ndarray:
        """Координаты людей (N, 2) в виде сверху на кадре frame. Пусто, если кадр не размечен."""
        if frame < 0 or frame >= len(self.positions):
            return np.empty((0, 2))
        row = self.positions[frame]
        valid = row[row >= 0]
        return self.grid.cell_center(valid) if len(valid) else np.empty((0, 2))


def load_ground_truth(path: str | Path, grid: TopViewGrid) -> GroundTruth:
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    header = [int(x) for x in rows[1]]
    n_frames, n_people, grid_w, grid_h, step, first, last = header
    if (grid_w, grid_h) != (grid.grid_w, grid.grid_h):
        raise ValueError(
            f"{path}: сетка в файле {grid_w}x{grid_h}, в конфиге {grid.grid_w}x{grid.grid_h}"
        )
    positions = np.array([[int(x) for x in r] for r in rows[2 : 2 + n_frames]], dtype=np.int64)
    if positions.shape != (n_frames, n_people):
        raise ValueError(f"{path}: ожидалось {(n_frames, n_people)}, прочитано {positions.shape}")
    return GroundTruth(grid, step, first, last, positions)
