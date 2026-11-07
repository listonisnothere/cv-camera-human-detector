"""Вид сверху: детекции со всех камер на одной карте пола.

Конвейер для одного момента времени:
    1. каждая камера даёт рамки людей (xyxy в пикселях своего кадра);
    2. берём точку ступней = середина нижней стороны рамки;
    3. гомографией камеры переводим её на пол (CameraProjector.to_floor);
    4. один и тот же человек виден с нескольких камер -> на полу несколько
       близких точек. Склеиваем их в одну (fuse_points);
    5. сравниваем с ground truth: сопоставляем точки "венгерским алгоритмом"
       и считаем, сколько людей найдено правильно (match_points).

Почему это ценно: одна камера не видит людей, закрытых другими людьми
(окклюзия). Несколько камер с разных сторон — видят. На карте сверху
можно считать людей в зоне, мерить дистанцию, строить маршруты.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from box_interceptor.geometry.homography import invert, project_points

# BGR-цвета для камер 0..5
CAMERA_COLORS = [(66, 135, 245), (80, 200, 120), (60, 80, 230), (200, 160, 40), (180, 90, 200), (40, 200, 220)]  # fmt: skip


@dataclass(frozen=True)
class CameraProjector:
    camera: int
    ground_h: np.ndarray  # картинка калибровки -> пол
    scale: float = 1.0  # во сколько раз кадр калибровки больше нашего видео

    def feet(self, xyxy: np.ndarray) -> np.ndarray:
        """Точки ступней (N, 2) в пикселях видео."""
        xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)

    def to_floor(self, xyxy: np.ndarray) -> np.ndarray:
        return project_points(self.feet(xyxy) * self.scale, self.ground_h)

    def floor_to_image(self, floor_pts: np.ndarray) -> np.ndarray:
        """Обратное направление: точка пола -> пиксель видео (чтобы рисовать зоны пола на кадре)."""
        return project_points(floor_pts, invert(self.ground_h)) / self.scale


def inside_bounds(points: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    p = np.asarray(points).reshape(-1, 2)
    return np.isfinite(p).all(1) & (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)  # fmt: skip


def fuse_points(per_camera: list[np.ndarray], radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Склеивает точки разных камер, лежащие ближе radius, в одну.

    Жадный алгоритм: начинаем с точек, у которых больше всего соседей
    (их подтверждают несколько камер), забираем соседей в кластер, берём центр.
    Возвращает (центры (K, 2), сколько камер подтверждает каждый центр (K,)).
    """
    pts_list, cam_list = [], []
    for cam, pts in enumerate(per_camera):
        pts = np.asarray(pts).reshape(-1, 2)
        pts_list.append(pts)
        cam_list.append(np.full(len(pts), cam))
    if not pts_list or sum(len(p) for p in pts_list) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    pts = np.concatenate(pts_list)
    cams = np.concatenate(cam_list)

    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    near = dist <= radius
    used = np.zeros(len(pts), dtype=bool)
    centers, support = [], []
    for i in np.argsort(-near.sum(1)):
        if used[i]:
            continue
        group = near[i] & ~used
        # Одна камера не может видеть одного человека дважды: из каждой камеры — ближайшая точка
        members = []
        for cam in np.unique(cams[group]):
            idx = np.where(group & (cams == cam))[0]
            members.append(idx[np.argmin(dist[i, idx])])
        used[members] = True
        centers.append(pts[members].mean(0))
        support.append(len(members))
    return np.array(centers), np.array(support)


@dataclass(frozen=True)
class MatchResult:
    tp: int  # найден и есть в GT
    fp: int  # найден, но в GT его нет (ложная тревога)
    fn: int  # есть в GT, но не найден (пропуск)
    errors: tuple[float, ...]  # расстояния для верных совпадений

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    def __add__(self, other: MatchResult) -> MatchResult:
        return MatchResult(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.errors + other.errors
        )


def match_points(pred: np.ndarray, gt: np.ndarray, radius: float) -> MatchResult:
    """Оптимальное сопоставление "предсказание <-> истина" (венгерский алгоритм).

    Каждой GT-точке — не больше одного предсказания и наоборот. Пара засчитывается,
    если расстояние <= radius.
    """
    pred = np.asarray(pred).reshape(-1, 2)
    gt = np.asarray(gt).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0:
        return MatchResult(0, len(pred), len(gt), ())
    cost = np.linalg.norm(pred[:, None] - gt[None], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    ok = cost[rows, cols] <= radius
    tp = int(ok.sum())
    return MatchResult(tp, len(pred) - tp, len(gt) - tp, tuple(cost[rows, cols][ok].tolist()))


class TopViewRenderer:
    """Рисует карту пола: сетку, зоны, GT, точки камер и итоговые позиции людей."""

    def __init__(self, bounds: tuple[float, float, float, float], height_px: int = 576):
        self.bounds = bounds
        x0, y0, x1, y1 = bounds
        self.pad = 30
        self.scale = (height_px - 2 * self.pad) / (y1 - y0)
        self.size = (int((x1 - x0) * self.scale) + 2 * self.pad, height_px)

    def to_px(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        x0, y0, _, _ = self.bounds
        return ((pts - [x0, y0]) * self.scale + self.pad).round().astype(int)

    def render(
        self,
        fused: np.ndarray,
        per_camera: list[np.ndarray] | None = None,
        gt: np.ndarray | None = None,
        zones: list[tuple[str, np.ndarray, int]] | None = None,
        title: str = "",
    ) -> np.ndarray:
        w, h = self.size
        img = np.full((h, w, 3), 245, dtype=np.uint8)
        x0, y0, x1, y1 = self.bounds
        tl, br = self.to_px(np.array([[x0, y0], [x1, y1]]))
        cv2.rectangle(img, tuple(tl), tuple(br), (200, 200, 200), 1)
        for name, polygon, count in zones or []:
            poly = self.to_px(polygon)
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], (180, 220, 255))
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, dst=img)
            cv2.polylines(img, [poly], True, (0, 120, 220), 2)
            cv2.putText(img, f"{name}: {count}", tuple(poly[0] + [4, 16]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 160), 1, cv2.LINE_AA)  # fmt: skip
        if gt is not None:
            for p in self.to_px(gt):
                cv2.circle(img, tuple(p), 10, (40, 40, 40), 1, cv2.LINE_AA)
        for cam, pts in enumerate(per_camera or []):
            color = CAMERA_COLORS[cam % len(CAMERA_COLORS)]
            for p in self.to_px(pts):
                cv2.circle(img, tuple(p), 3, color, -1, cv2.LINE_AA)
        for p in self.to_px(fused):
            cv2.circle(img, tuple(p), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), 6, (255, 255, 255), 1, cv2.LINE_AA)
        caption = f"{title}  people: {len(fused)}" + (f"  GT: {len(gt)}" if gt is not None else "")
        cv2.putText(
            img, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA
        )
        return img


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Маска: какие точки лежат внутри многоугольника (на полу)."""
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return np.array(
        [cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in np.asarray(points).reshape(-1, 2)],
        dtype=bool,
    )  # fmt: skip
