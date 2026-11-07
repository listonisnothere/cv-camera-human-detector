"""Конвейер одной камеры: кадр -> детекция -> трекинг -> зоны/линии -> отрисовка.

    ┌────────┐   ┌──────────┐   ┌─────────┐   ┌──────────────┐   ┌──────────┐
    │ кадр   │ → │ детектор │ → │ трекер  │ → │ линии, зоны, │ → │ рисуем   │
    │ BGR    │   │ (рамки)  │   │ (ID)    │   │ тепловая к.  │   │ результат│
    └────────┘   └──────────┘   └─────────┘   └──────────────┘   └──────────┘

Где у человека "точка"? Для зон и линий берём НИЖНЮЮ СЕРЕДИНУ рамки (ступни):
человек стоит в зоне, если в зоне стоят его ноги, а не голова, которая на
кадре может быть над соседней зоной.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from box_interceptor.config import CameraConfig
from box_interceptor.detection.detector import PersonDetector
from box_interceptor.tracking.tracker import make_tracker

ANCHOR = sv.Position.BOTTOM_CENTER


@dataclass
class FrameResult:
    frame: np.ndarray  # кадр с отрисовкой
    detections: sv.Detections  # с tracker_id, если трекинг включён
    zone_counts: dict[str, int] = field(default_factory=dict)  # людей в зоне сейчас
    line_counts: dict[str, tuple[int, int]] = field(default_factory=dict)  # (вошли, вышли)

    @property
    def people(self) -> int:
        return len(self.detections)


class CameraPipeline:
    def __init__(
        self,
        detector: PersonDetector,
        camera: CameraConfig | None = None,
        fps: float = 25.0,
        track: bool = True,
        heatmap: bool = False,
        trace: bool = True,
    ):
        self.detector = detector
        self.tracker = make_tracker(fps) if track else None
        cam_lines = camera.lines if camera else []
        cam_zones = camera.zones if camera else []
        if cam_lines and not track:
            raise ValueError("Подсчёт пересечений линии требует трекинга (нужны ID)")

        self.lines = {
            cfg.name: sv.LineZone(
                start=sv.Point(*cfg.start), end=sv.Point(*cfg.end), triggering_anchors=[ANCHOR]
            )
            for cfg in cam_lines
        }
        self.zones = {
            cfg.name: sv.PolygonZone(
                polygon=np.array(cfg.polygon, dtype=np.int32), triggering_anchors=(ANCHOR,)
            )
            for cfg in cam_zones
        }

        # Отрисовка. ColorLookup.TRACK — у каждого ID свой цвет, стабильный во времени.
        self.lookup = sv.ColorLookup.TRACK if track else sv.ColorLookup.CLASS
        self.box = sv.BoxAnnotator(thickness=2, color_lookup=self.lookup)
        self.label = sv.LabelAnnotator(text_scale=0.4, text_padding=3, color_lookup=self.lookup)
        self.trace: sv.TraceAnnotator | None = None
        self.heatmap: sv.HeatMapAnnotator | None = None
        self.set_overlays(heatmap=heatmap, trace=trace)
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=2, text_scale=0.4, text_thickness=1, text_padding=4
        )
        self.zone_annotators = {
            name: sv.PolygonZoneAnnotator(
                zone=z,
                color=sv.Color.from_hex("#ff9f1c"),
                thickness=2,
                text_scale=0.5,
                text_padding=4,
            )  # fmt: skip
            for name, z in self.zones.items()
        }

    def set_overlays(self, heatmap: bool | None = None, trace: bool | None = None) -> None:
        """Включить/выключить тепловую карту и след на лету (None — не менять).

        Тепловая карта копит "тепло" с момента включения, поэтому при повторном
        включении начинается с нуля.
        """
        if heatmap is not None:
            self.heatmap = (
                sv.HeatMapAnnotator(position=ANCHOR, opacity=0.4, radius=20) if heatmap else None
            )
        if trace is not None:
            self.trace = (
                sv.TraceAnnotator(trace_length=40, position=ANCHOR, color_lookup=self.lookup)
                if trace and self.tracker is not None
                else None
            )

    def detect_and_track(self, frame: np.ndarray) -> sv.Detections:
        detections = self.detector(frame)
        if self.tracker is not None:
            detections = self.tracker.update(detections)
        return detections

    def process(self, frame: np.ndarray, draw: bool = True) -> FrameResult:
        detections = self.detect_and_track(frame)

        line_counts = {}
        for name, line in self.lines.items():
            line.trigger(detections)
            line_counts[name] = (line.in_count, line.out_count)
        zone_counts = {}
        for name, zone in self.zones.items():
            zone.trigger(detections)
            zone_counts[name] = int(zone.current_count)

        out = self.annotate(frame, detections) if draw else frame
        return FrameResult(out, detections, zone_counts, line_counts)

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        out = frame.copy()
        if self.heatmap is not None:
            out = self.heatmap.annotate(out, detections)
        for annotator in self.zone_annotators.values():
            out = annotator.annotate(out)
        if self.trace is not None:
            out = self.trace.annotate(out, detections)
        out = self.box.annotate(out, detections)
        out = self.label.annotate(out, detections, labels=self._labels(detections))
        for line in self.lines.values():
            out = self.line_annotator.annotate(out, line_counter=line)
        return out

    @staticmethod
    def _labels(detections: sv.Detections) -> list[str]:
        conf = (
            detections.confidence if detections.confidence is not None else [None] * len(detections)
        )
        ids = (
            detections.tracker_id if detections.tracker_id is not None else [None] * len(detections)
        )
        labels = []
        for c, t in zip(conf, ids, strict=True):
            parts = [f"#{t}"] if t is not None else []
            if c is not None:
                parts.append(f"{c:.2f}")
            labels.append(" ".join(parts) or "person")
        return labels
