"""Замер скорости детекторов на одном и том же наборе кадров.

Меряем только саму модель (предобработка + сеть + постобработка), без чтения
видео и отрисовки. FPS = 1000 / (мс на кадр).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from box_interceptor.data import mot17
from box_interceptor.detection.detector import YoloDetector, build_detector


def load_frames(n: int = 150, sequence: str = "MOT17-09") -> list[np.ndarray]:
    """Full HD кадры со статичной камеры — типичный вход камеры наблюдения."""
    info = mot17.read_seqinfo(mot17.seq_dir(sequence))
    return [cv2.imread(str(info.frame_path(f))) for f in range(1, min(n, info.length) + 1)]


def benchmark(
    spec: str, frames: list[np.ndarray], device: str | None = None, warmup: int = 10
) -> dict:
    if device is not None and not spec.startswith("rfdetr"):
        path = spec if spec.endswith((".pt", ".mlpackage")) else f"{spec}.pt"
        detector = YoloDetector(path, device=device)
        label = f"{detector.name} [{device}]"
    else:
        detector = build_detector(spec)
        dev = getattr(detector, "device", None) or (
            "coreml" if spec.endswith(".mlpackage") else "auto"
        )
        label = f"{detector.name} [{dev}]"
    for f in frames[:warmup]:
        detector(f)
    t0 = time.perf_counter()
    for f in frames:
        detector(f)
    ms = 1000 * (time.perf_counter() - t0) / len(frames)
    return {"model": label, "ms/frame": ms, "FPS": 1000 / ms}


def run(specs: list[str], n_frames: int = 150) -> pd.DataFrame:
    frames = load_frames(n_frames)
    rows = []
    for spec in specs:
        name, _, device = spec.partition("@")  # "yolo26n@cpu" — принудительно на CPU
        print(f"→ {spec}")
        rows.append(benchmark(name, frames, device=device or None))
    return pd.DataFrame(rows)
