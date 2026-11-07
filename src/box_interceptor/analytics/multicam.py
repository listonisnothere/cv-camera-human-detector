"""Несколько синхронных камер: мозаика 2x2 + вид сверху + оценка по ground truth.

Детектор (тяжёлая нейросеть) загружается ОДИН раз и используется всеми камерами.
Трекер — СВОЙ у каждой камеры: ID человека в камере 0 и камере 1 никак не связаны.
Связать людей между камерами — это задача multi-camera tracking / re-identification,
а мы делаем более простую и надёжную вещь: переводим всех на общую карту пола.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from box_interceptor.analytics.pipeline import CameraPipeline, FrameResult
from box_interceptor.config import SceneConfig
from box_interceptor.detection.detector import PersonDetector
from box_interceptor.geometry.birdseye import (
    CameraProjector,
    MatchResult,
    TopViewRenderer,
    fuse_points,
    inside_bounds,
    match_points,
    points_in_polygon,
)
from box_interceptor.geometry.epfl import (
    GroundTruth,
    TopViewGrid,
    load_calibration,
    load_ground_truth,
)
from box_interceptor.video.io import FrameSource
from box_interceptor.video.mosaic import mosaic


@dataclass
class TopView:
    """Всё, что нужно для вида сверху одной сцены."""

    projectors: list[CameraProjector]
    grid: TopViewGrid
    gt: GroundTruth
    renderer: TopViewRenderer
    merge_radius: float
    match_radius: float
    min_support: int
    zones: list[tuple[str, np.ndarray]]

    @classmethod
    def from_scene(cls, scene: SceneConfig) -> TopView | None:
        if not (scene.calibration and scene.ground_truth and scene.birdseye):
            return None
        calib = {c.camera: c for c in load_calibration(scene.calibration)}
        projectors = [
            CameraProjector(cam.id, calib[cam.id].ground_h, scene.calibration_scale)
            for cam in scene.cameras
        ]
        g = scene.ground_truth.grid
        grid = TopViewGrid(g.width, g.height, g.origin, g.size)
        return cls(
            projectors=projectors,
            grid=grid,
            gt=load_ground_truth(scene.ground_truth.file, grid),
            renderer=TopViewRenderer(grid.bounds),
            merge_radius=scene.birdseye.merge_radius,
            match_radius=scene.birdseye.match_radius,
            min_support=scene.birdseye.min_support,
            zones=[(z.name, np.array(z.polygon)) for z in scene.birdseye.zones],
        )

    def floor_points(self, xyxy_per_camera: list[np.ndarray]) -> list[np.ndarray]:
        """Ступни каждого человека с каждой камеры -> точки на полу (только внутри сетки)."""
        result = []
        for proj, xyxy in zip(self.projectors, xyxy_per_camera, strict=True):
            pts = proj.to_floor(xyxy)
            result.append(pts[inside_bounds(pts, self.grid.bounds)])
        return result

    def zone_counts(self, fused: np.ndarray) -> dict[str, int]:
        return {name: int(points_in_polygon(fused, poly).sum()) for name, poly in self.zones}

    def render(self, per_camera: list[np.ndarray], fused: np.ndarray, frame_idx: int) -> np.ndarray:
        counts = self.zone_counts(fused)
        zones = [(n, p, counts[n]) for n, p in self.zones]
        labeled = (frame_idx - self.gt.first_frame) % self.gt.step == 0
        gt = self.gt.people_at(frame_idx) if labeled else None
        return self.renderer.render(fused, per_camera, gt, zones, title=f"top view  #{frame_idx}")


@dataclass
class MultiCamResult:
    cameras: list[FrameResult]  # результат каждой камеры (кадр с отрисовкой, детекции, счётчики)
    top: np.ndarray | None = None  # картинка вида сверху, если у сцены есть калибровка
    floor_people: int | None = None  # людей на карте после слияния камер
    floor_zones: dict[str, int] = field(default_factory=dict)  # людей в зонах на полу


class MultiCamRunner:
    def __init__(self, scene: SceneConfig, detector: PersonDetector, birdseye: bool = True):
        self.scene = scene
        self.sources = [FrameSource(cam.video) for cam in scene.cameras]
        fps = self.sources[0].fps
        self.pipelines = [CameraPipeline(detector, cam, fps=fps) for cam in scene.cameras]
        self.top = TopView.from_scene(scene) if birdseye else None

    @property
    def fps(self) -> float:
        return self.sources[0].fps

    @property
    def frame_count(self) -> int | None:
        counts = [s.frame_count for s in self.sources if s.frame_count]
        return min(counts) if counts else None

    def frames(self) -> Iterator[tuple[int, list[np.ndarray]]]:
        """Читает кадры со всех камер синхронно. Останавливается, когда кончится любое видео."""
        idx = 0
        while True:
            frames = [s.read() for s in self.sources]
            if any(f is None for f in frames):
                return
            yield idx, frames  # type: ignore[misc]
            idx += 1

    def process(self, idx: int, frames: list[np.ndarray]) -> MultiCamResult:
        """Один момент времени: все камеры + вид сверху. Кадры камер остаются отдельными."""
        results = [p.process(f) for p, f in zip(self.pipelines, frames, strict=True)]
        if self.top is None:
            return MultiCamResult(results)
        per_cam = self.top.floor_points([r.detections.xyxy for r in results])
        fused, support = fuse_points(per_cam, self.top.merge_radius)
        fused = fused[support >= self.top.min_support]
        top = self.top.render(per_cam, fused, idx)
        return MultiCamResult(results, top, len(fused), self.top.zone_counts(fused))

    def step(self, idx: int, frames: list[np.ndarray]) -> np.ndarray:
        """Всё в одну картинку: мозаика камер + вид сверху справа (для записи в mp4)."""
        res = self.process(idx, frames)
        labels = [f"cam {cam.id}: {r.people} people" for cam, r in zip(self.scene.cameras, res.cameras, strict=True)]  # fmt: skip
        grid = mosaic([r.frame for r in res.cameras], cols=2, labels=labels)
        if res.top is None:
            return grid
        top = cv2.resize(res.top, (int(res.top.shape[1] * grid.shape[0] / res.top.shape[0]), grid.shape[0]))  # fmt: skip
        return np.hstack([grid, top])

    def close(self) -> None:
        for s in self.sources:
            s.close()


def evaluate_birdseye(
    scene: SceneConfig, detector: PersonDetector, max_frames: int | None = None
) -> dict[str, MatchResult]:
    """Сравнивает позиции людей на полу с ground truth на размеченных кадрах.

    Возвращает результат для каждой камеры по отдельности и для слияния всех камер.
    Так видно, сколько даёт многокамерность.
    """
    top = TopView.from_scene(scene)
    if top is None:
        raise ValueError(f"У сцены {scene.name} нет калибровки/GT/birdseye в конфиге")
    samples = collect_birdseye_samples(scene, detector, top, max_frames)
    cam_ids = [cam.id for cam in scene.cameras]
    return score_birdseye(samples, cam_ids, top.merge_radius, top.match_radius, top.min_support)


Sample = tuple[list[np.ndarray], np.ndarray]  # (точки на полу по камерам, GT-точки)


def collect_birdseye_samples(
    scene: SceneConfig, detector: PersonDetector, top: TopView, max_frames: int | None = None
) -> list[Sample]:
    """Прогоняет детектор по размеченным кадрам всех камер. Самая дорогая часть —
    поэтому результат отделён от подсчёта метрик: параметры слияния можно
    перебирать, не запуская нейросеть заново."""
    labeled = [f for f in top.gt.labeled_frames if max_frames is None or f < max_frames]
    wanted = set(labeled)
    sources = [FrameSource(cam.video) for cam in scene.cameras]
    samples: list[Sample] = []
    try:
        for idx in range(max(labeled) + 1):
            if idx not in wanted:
                for s in sources:
                    s.cap.grab()  # пропускаем кадр без полного декодирования в numpy
                continue
            frames = [s.read() for s in sources]
            if any(f is None for f in frames):
                break
            per_cam = top.floor_points([detector(f).xyxy for f in frames])
            samples.append((per_cam, top.gt.people_at(idx)))
    finally:
        for s in sources:
            s.close()
    return samples


def score_birdseye(
    samples: list[Sample],
    camera_ids: list[int],
    merge_radius: float,
    match_radius: float,
    min_support: int = 1,
) -> dict[str, MatchResult]:
    """Метрики для каждой камеры отдельно и для слияния.

    min_support — сколько камер минимум должны подтвердить человека. 2 убирает
    ложные срабатывания одной камеры, но теряет людей, которых видит только одна.
    """
    empty = MatchResult(0, 0, 0, ())
    totals: dict[str, MatchResult] = {}
    for per_cam, gt in samples:
        for cam_id, pts in zip(camera_ids, per_cam, strict=True):
            key = f"camera {cam_id}"
            totals[key] = totals.get(key, empty) + match_points(pts, gt, match_radius)
        fused, support = fuse_points(per_cam, merge_radius)
        fused = fused[support >= min_support]
        totals["fusion"] = totals.get("fusion", empty) + match_points(fused, gt, match_radius)
    return totals
