"""Сравнение детекторов на MOT17 (val): точность и скорость на одних и тех же кадрах."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from box_interceptor.data import mot17
from box_interceptor.detection.detector import build_detector
from box_interceptor.eval.det_metrics import DetectionEvaluator

# Для AP нужна вся кривая precision-recall — от очень уверенных находок до
# совсем сомнительных. Поэтому при оценке порог уверенности почти нулевой.
EVAL_CONF = 0.01
WARMUP_FRAMES = 5  # первые кадры на GPU медленные (компиляция ядер, выделение памяти)


def evaluate_detector(
    spec: str,
    sequences: list[str] = mot17.VAL_SEQUENCES,
    step: int = 1,
    imgsz: int = 640,
    max_frames: int | None = None,
) -> dict[str, float | str]:
    detector = build_detector(spec, conf=EVAL_CONF, imgsz=imgsz)
    evaluator = DetectionEvaluator()
    times: list[float] = []
    for name in sequences:
        info = mot17.read_seqinfo(mot17.seq_dir(name))
        gt = mot17.load_gt(info.path)
        frames = list(range(1, info.length + 1, step))[:max_frames]
        for f in frames:
            img = cv2.imread(str(info.frame_path(f)))
            t0 = time.perf_counter()
            det = detector(img)
            times.append(time.perf_counter() - t0)
            evaluator.update(det.xyxy, det.confidence, gt[f].boxes, gt[f].ignore)
    t = np.array(times[WARMUP_FRAMES:] if len(times) > 2 * WARMUP_FRAMES else times)
    return {
        "model": detector.name,
        **evaluator.compute(),
        "ms/frame": 1000 * float(t.mean()),
        "FPS": 1 / float(t.mean()),
        "frames": len(times),
    }


def compare_detectors(specs: list[str], **kwargs) -> pd.DataFrame:
    rows = []
    for spec in specs:
        print(f"→ {spec}")
        rows.append(evaluate_detector(spec, **kwargs))
    df = pd.DataFrame(rows)
    cols = ["model", "AP50", "AP75", "AP50-95", "F1", "precision", "recall", "best_conf", "ms/frame", "FPS", "frames", "n_gt"]  # fmt: skip
    return df[cols]
