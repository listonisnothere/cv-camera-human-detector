"""Метрики трекинга на MOT17 val: детектор + ByteTrack -> файлы в формате MOT -> HOTA/MOTA/IDF1.

Формат MOT (по строке на человека на кадре):
    кадр, id, x, y, ширина, высота, уверенность, -1, -1, -1
Его понимают все инструменты оценки (TrackEval, trackers.eval).

Метрики считает trackers.eval — порт эталонного TrackEval. Он сам применяет
правила MOTChallenge: оцениваются только пешеходы (класс 1), а находки,
совпавшие с "отвлекающими" объектами (стоящие люди, отражения...), выкидываются.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from box_interceptor import paths
from box_interceptor.data import mot17
from box_interceptor.detection.detector import build_detector
from box_interceptor.tracking.tracker import TUNED, TrackerParams, make_tracker

OUT_ROOT = paths.OUTPUTS / "tracking"


@dataclass(frozen=True)
class TrackingRun:
    model: str
    det_conf: float = 0.05  # трекеру нужны и неуверенные детекции (2-й шаг ByteTrack)
    imgsz: int = 640
    params: TrackerParams = field(default=TUNED)

    @property
    def label(self) -> str:
        p = self.params
        return f"{Path(self.model).stem}@{self.imgsz} hi={p.high_conf} act={p.activation}"


def track_sequence(run: TrackingRun, detector, info: mot17.SeqInfo) -> list[str]:  # noqa: ANN001
    tracker = make_tracker(info.fps, run.params)
    lines = []
    for frame in range(1, info.length + 1):
        img = cv2.imread(str(info.frame_path(frame)))
        tracked = tracker.update(detector(img))
        for (x1, y1, x2, y2), tid, conf in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence, strict=True):  # fmt: skip
            if tid is None or tid < 0:
                continue
            lines.append(f"{frame},{int(tid)},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{conf:.3f},-1,-1,-1")  # fmt: skip
    return lines


def evaluate_run(run: TrackingRun, sequences: list[str] = mot17.VAL_SEQUENCES) -> dict:
    from trackers.eval import evaluate_mot_sequences

    detector = build_detector(run.model, conf=run.det_conf, imgsz=run.imgsz)
    out_dir = OUT_ROOT / run.label.replace(" ", "_").replace("=", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sequences:
        info = mot17.read_seqinfo(mot17.seq_dir(name))
        (out_dir / f"{info.path.name}.txt").write_text("\n".join(track_sequence(run, detector, info)) + "\n")  # fmt: skip
    seqmap = out_dir / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(f"{s}-FRCNN" for s in sequences) + "\n")

    res = evaluate_mot_sequences(
        gt_dir=mot17.TRAIN_ROOT, tracker_dir=out_dir, seqmap=seqmap,
        metrics=["CLEAR", "HOTA", "Identity"],
    ).aggregate  # fmt: skip
    return {
        "run": run.label,
        "HOTA": res.HOTA.HOTA,
        "DetA": res.HOTA.DetA,
        "AssA": res.HOTA.AssA,
        "MOTA": res.CLEAR.MOTA,
        "IDF1": res.Identity.IDF1,
        "IDSW": res.CLEAR.IDSW,
        "FP": res.CLEAR.CLR_FP,
        "FN": res.CLEAR.CLR_FN,
    }
