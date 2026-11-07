"""Движок "живой" обработки для веб-панели.

Работает в отдельном потоке:
  1. читает кадры всех камер сцены В РЕАЛЬНОМ ВРЕМЕНИ — видео идёт со своей
     скоростью (25 к/с), как будто это живые камеры;
  2. прогоняет их через ИИ: детекция -> трекинг -> зоны/линии -> вид сверху;
  3. кладёт последние готовые кадры (JPEG) и цифры в общую память.
Веб-сервер в это время отправляет их браузеру.

Реальное время = пропуск кадров. Видео не ждёт ИИ. Если обработка момента
заняла 60 мс, а между кадрами 40 мс, то следующий кадр уже "в прошлом" —
мы его пропускаем и берём тот, что "в эфире" сейчас. Так работают настоящие
системы видеоаналитики: лучше пропустить кадр, чем отставать от жизни.
Доля пропущенных кадров — честная мера того, успевает ли модель.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from box_interceptor.analytics.multicam import MultiCamResult, MultiCamRunner
from box_interceptor.config import SceneConfig
from box_interceptor.detection.detector import build_detector
from box_interceptor.tracking.tracker import TRACK_DET_CONF

JPEG_QUALITY = 80


def encode_jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Не удалось закодировать кадр в JPEG")
    return buf.tobytes()


class Engine:
    """Один движок на сервер: одновременно показывается одна сцена."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._seq = 0  # растёт при любом изменении — по нему сервер понимает, что есть новое
        self._frames: dict[str, bytes] = {}
        self._stats: dict[str, Any] = {"status": "idle"}
        self._overlays = {"heatmap": False, "trace": True}
        self._overlays_changed = False

    # ---------------------------------------------------------------- для сервера

    def start(self, scene: SceneConfig, model: str) -> None:
        self.stop()
        self._stop = threading.Event()
        self._publish(
            frames={},
            stats={
                "status": "loading",
                "message": "Загружаю модель и открываю камеры…",
                "scene": scene.name,
                "model": model,
                "cameras": [{"id": c.id, "key": f"cam{c.id}"} for c in scene.cameras],
            },
        )
        self._thread = threading.Thread(
            target=self._run, args=(scene, model, self._stop), name="engine", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=10)
        self._thread = None

    def set_overlays(self, heatmap: bool | None = None, trace: bool | None = None) -> dict:
        with self._lock:
            if heatmap is not None:
                self._overlays["heatmap"] = heatmap
            if trace is not None:
                self._overlays["trace"] = trace
            self._overlays_changed = True
            return dict(self._overlays)

    @property
    def overlays(self) -> dict:
        with self._lock:
            return dict(self._overlays)

    def snapshot(self) -> tuple[int, dict[str, bytes], dict[str, Any]]:
        """(номер версии, последние кадры, статистика). Словари не копируются —
        движок каждый раз подменяет их целиком, так что читать их безопасно."""
        with self._lock:
            return self._seq, self._frames, self._stats

    # ---------------------------------------------------------------- рабочий поток

    def _publish(self, frames: dict[str, bytes], stats: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            self._frames = frames
            self._stats = stats

    def _run(self, scene: SceneConfig, model: str, stop: threading.Event) -> None:
        try:
            detector = build_detector(model, conf=TRACK_DET_CONF)
            loop = 0
            while not stop.is_set():
                loop += 1
                runner = MultiCamRunner(scene, detector)
                try:
                    replay = self._play(runner, scene, model, loop, stop)
                finally:
                    runner.close()
                if not replay:
                    break
        except Exception as exc:  # noqa: BLE001 — любую ошибку показываем в интерфейсе
            with self._lock:
                stats = dict(self._stats)
            stats |= {"status": "error", "message": _explain(exc)}
            self._publish(frames={}, stats=stats)

    def _apply_overlays(self, runner: MultiCamRunner, force: bool = False) -> None:
        with self._lock:
            if not (force or self._overlays_changed):
                return
            overlays = dict(self._overlays)
            self._overlays_changed = False
        for p in runner.pipelines:
            p.set_overlays(heatmap=overlays["heatmap"], trace=overlays["trace"])

    def _play(
        self,
        runner: MultiCamRunner,
        scene: SceneConfig,
        model: str,
        loop: int,
        stop: threading.Event,
    ) -> bool:
        """Проигрывает сцену один раз. True — видео кончилось, можно по кругу."""
        live = any(s.is_live for s in runner.sources)
        fps = runner.fps
        total = runner.frame_count
        self._apply_overlays(runner, force=True)
        t0 = time.perf_counter()
        idx = processed = dropped = 0
        done_at: deque[float] = deque(maxlen=50)  # время последних обработанных кадров -> FPS
        latency_ms: float | None = None

        while not stop.is_set():
            if not live:
                # Какой кадр сейчас "в эфире"? Всё, что раньше, уже устарело — пропускаем.
                due = int((time.perf_counter() - t0) * fps)
                if due > idx:
                    for _ in range(due - idx):
                        for s in runner.sources:
                            s.cap.grab()  # пропустить кадр, не декодируя его в картинку
                    dropped += due - idx
                    idx = due
            frames = [s.read() for s in runner.sources]
            if any(f is None for f in frames):
                return not live  # файл кончился -> по кругу; живой поток оборвался -> стоп

            self._apply_overlays(runner)
            t = time.perf_counter()
            result = runner.process(idx, frames)
            ms = 1000 * (time.perf_counter() - t)
            latency_ms = ms if latency_ms is None else 0.9 * latency_ms + 0.1 * ms
            processed += 1
            done_at.append(time.perf_counter())

            ai_fps = (len(done_at) - 1) / (done_at[-1] - done_at[0]) if len(done_at) > 1 else 0.0
            self._publish(
                frames=_encode(scene, result),
                stats=_stats(scene, model, result, live=live, video_fps=fps, ai_fps=ai_fps,
                             latency_ms=latency_ms, dropped=dropped, processed=processed,
                             frame=idx, total=total, loop=loop),
            )  # fmt: skip
            idx += 1
            if not live:
                wait = t0 + idx / fps - time.perf_counter()
                if wait > 0:
                    stop.wait(wait)  # ИИ быстрее видео — ждём следующий кадр "эфира"
        return False


def _encode(scene: SceneConfig, result: MultiCamResult) -> dict[str, bytes]:
    frames = {
        f"cam{cam.id}": encode_jpeg(r.frame)
        for cam, r in zip(scene.cameras, result.cameras, strict=True)
    }
    if result.top is not None:
        frames["top"] = encode_jpeg(result.top)
    return frames


def _stats(scene: SceneConfig, model: str, result: MultiCamResult, **kw: Any) -> dict[str, Any]:
    dropped, processed = kw.pop("dropped"), kw.pop("processed")
    return {
        "status": "running",
        "message": "",
        "scene": scene.name,
        "description": scene.description,
        "model": model,
        "cameras": [
            {
                "id": cam.id,
                "key": f"cam{cam.id}",
                "people": r.people,
                "zones": r.zone_counts,
                "lines": {name: {"in": i, "out": o} for name, (i, o) in r.line_counts.items()},
            }
            for cam, r in zip(scene.cameras, result.cameras, strict=True)
        ],
        "floor": (
            {"people": result.floor_people, "zones": result.floor_zones}
            if result.top is not None
            else None
        ),
        "dropped_pct": 100 * dropped / max(dropped + processed, 1),
        **kw,
    }


def _explain(exc: Exception) -> str:
    text = str(exc)
    if isinstance(exc, OSError) and "источник" in text:
        return f"{text}. Проверьте путь к файлу или адрес камеры."
    return f"{type(exc).__name__}: {text}"
