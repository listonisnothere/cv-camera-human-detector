"""Мозаика: несколько кадров в одной картинке (как на мониторе охранника)."""

from __future__ import annotations

import math

import cv2
import numpy as np


def mosaic(
    frames: list[np.ndarray],
    cols: int = 2,
    tile_size: tuple[int, int] | None = None,
    labels: list[str] | None = None,
) -> np.ndarray:
    """Склеивает кадры в сетку cols x rows. Пустые ячейки заливает чёрным.

    tile_size=(ширина, высота) — размер одной ячейки; по умолчанию размер первого кадра.
    """
    if not frames:
        raise ValueError("Нужен хотя бы один кадр")
    tw, th = tile_size or (frames[0].shape[1], frames[0].shape[0])
    rows = math.ceil(len(frames) / cols)
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        tile = cv2.resize(frame, (tw, th)) if frame.shape[:2] != (th, tw) else frame
        canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
        if labels and i < len(labels):
            draw_label(canvas, labels[i], (c * tw + 8, r * th + 8))
    return canvas


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5) -> None:
    """Белый текст на полупрозрачной тёмной плашке — читается на любом фоне."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 4, y - 4), (x + w + 4, y + h + base + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
    cv2.putText(
        img, text, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA
    )
