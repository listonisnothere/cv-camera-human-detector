"""Датасет EPFL "Multi-camera pedestrians video" (POM).

Страница: https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/
Лицензия: только для исследований, при публикации — ссылка на статью POM (TPAMI 2008).

Берём три сцены, у каждой 4 синхронные камеры, 25 к/с, и есть ground truth:
  lab6p       — лаборатория, 6 человек, ~2.5 мин
  terrace1    — терраса на улице, до 9 человек, ~3.5 мин
  passageway1 — подземный переход, плохое освещение

Видео лежат в старом кодеке (MPEG-4 / Indeo 5 в контейнере AVI). Мы сразу
перекодируем их в H.264 MP4: так их читает любой плеер, OpenCV и браузер.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from box_interceptor import paths
from box_interceptor.data.download import download_many

VIDEO_BASE = "https://documents.epfl.ch/groups/c/cv"
FILES_BASE = "https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08"


@dataclass(frozen=True)
class EpflSequence:
    name: str
    video_urls: tuple[str, ...]  # по одному на камеру, в порядке номеров камер
    calibration: str  # имя файла калибровки
    ground_truth: str  # имя файла ground truth


def _urls(bucket_by_cam: list[str], stem: str) -> tuple[str, ...]:
    return tuple(
        f"{VIDEO_BASE}/{bucket}/www/{stem}-c{cam}.avi" for cam, bucket in enumerate(bucket_by_cam)
    )


# Файлы одной сцены иногда лежат в разных "корзинах" сервера (video1/2/3) — так на сайте.
SEQUENCES: dict[str, EpflSequence] = {
    "lab6p": EpflSequence(
        "lab6p", _urls(["cvlab-pom-video1"] * 4, "6p"), "calibration-6p.txt", "gt_lab_6p.txt"
    ),
    "terrace1": EpflSequence(
        "terrace1",
        _urls(["cvlab-pom-video3"] * 4, "terrace1"),
        "calibration-terrace.txt",
        "gt_terrace1.txt",
    ),
    "passageway1": EpflSequence(
        "passageway1",
        _urls(["cvlab-pom-video2"] * 3 + ["cvlab-pom-video3"], "passageway1"),
        "calibration-passageway.txt",
        "gt_passageway1.txt",
    ),
}

RAW_DIR = paths.RAW / "epfl"


def video_path(sequence: str, camera: int) -> Path:
    """Путь к перекодированному mp4 для камеры сцены."""
    return paths.VIDEOS / "epfl" / sequence / f"c{camera}.mp4"


def download(sequences: list[str]) -> None:
    jobs: list[tuple[str, Path]] = []
    for name in sequences:
        seq = SEQUENCES[name]
        for url in seq.video_urls:
            jobs.append((url, RAW_DIR / "avi" / Path(url).name))
        for fname in (seq.calibration, seq.ground_truth):
            jobs.append((f"{FILES_BASE}/{fname}", RAW_DIR / "calib" / fname))
    print(f"Скачиваю {len(jobs)} файлов EPFL (параллельно)...")
    download_many(jobs)

    for name in sequences:
        seq = SEQUENCES[name]
        for cam, url in enumerate(seq.video_urls):
            transcode(RAW_DIR / "avi" / Path(url).name, video_path(name, cam))


def transcode(src: Path, dst: Path) -> Path:
    """AVI (MPEG-4/Indeo5) -> MP4 (H.264). Пропускает, если результат уже есть."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Нужен ffmpeg: brew install ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18",            # почти без потерь качества (меньше = лучше, 18 ≈ визуально без потерь)
        "-pix_fmt", "yuv420p",   # самый совместимый формат пикселей
        "-an",                   # звука там нет
        str(tmp),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    tmp.rename(dst)
    print(f"  ▶ {dst.relative_to(paths.ROOT)}")
    return dst
