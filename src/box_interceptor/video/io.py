"""Ввод и вывод видео.

FrameSource — единый способ читать кадры откуда угодно:
    файл       "data/videos/epfl/lab6p/c0.mp4"
    веб-камера "0" (номер устройства)
    поток      "rtsp://user:pass@192.168.1.10:554/stream" (так работают IP-камеры)

FfmpegWriter — запись результата в H.264 MP4 через ffmpeg. Встроенный в OpenCV
VideoWriter на macOS пишет mp4v, который не везде открывается; ffmpeg надёжнее.

Кадр в OpenCV — это numpy-массив (высота, ширина, 3) типа uint8 в порядке BGR
(не RGB!). Это частый источник багов при передаче кадров в модели.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class FrameSource:
    def __init__(self, source: str | int | Path):
        src = str(source)
        self.is_live = src.isdigit() or src.startswith(("rtsp://", "http://", "https://"))
        self.name = src
        self.cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not self.cap.isOpened():
            raise OSError(f"Не удалось открыть источник видео: {src}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 25.0
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = count if count > 0 and not self.is_live else None

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FfmpegWriter:
    """Пишет BGR-кадры в mp4 (H.264), передавая сырые пиксели в ffmpeg через pipe."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], crf: int = 23):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Нужен ffmpeg: brew install ffmpeg")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            # libx264 + yuv420p требует чётных размеров кадра — подрежем на 1 пиксель, если надо
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
            "-movflags", "+faststart",
            str(self.path),
        ]  # fmt: skip
        self.size = size
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg завершился с ошибкой при записи {self.path}")

    def __enter__(self) -> FfmpegWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Window:
    """Окно предпросмотра. Закрыть — клавиша q или Esc."""

    def __init__(self, title: str = "box-interceptor"):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def show(self, frame: np.ndarray) -> bool:
        """Показывает кадр. Возвращает False, если пользователь попросил выйти."""
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        cv2.destroyWindow(self.title)
        cv2.waitKey(1)
