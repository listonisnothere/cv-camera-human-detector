"""Конфиг сцены: какие камеры, где линии подсчёта и зоны, где калибровка.

Сцены описываются в configs/*.yaml. Pydantic проверяет файл при загрузке:
опечатка в имени поля или зона из двух точек дадут понятную ошибку сразу,
а не странное поведение посреди обработки видео.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from box_interceptor import paths

Point = tuple[float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # неизвестное поле в YAML = ошибка (ловит опечатки)


class LineConfig(_Strict):
    """Линия подсчёта. Человек, пересёкший её, считается как 'вход' или 'выход'."""

    name: str
    start: Point
    end: Point


class ZoneConfig(_Strict):
    """Многоугольная зона. Считаем, сколько людей в ней находится прямо сейчас."""

    name: str
    polygon: list[Point] = Field(min_length=3)


def _resolve_path(v: Path | None) -> Path | None:
    return v if v is None or v.is_absolute() else paths.ROOT / v


def _resolve_source(v: str) -> str:
    """Файл -> абсолютный путь. Номер веб-камеры ("0") и URL (rtsp://...) — как есть.

    Строка, а не Path: Path("rtsp://host/stream") склеил бы "//" и сломал URL.
    """
    if v.isdigit() or "://" in v:
        return v
    p = Path(v)
    return str(p if p.is_absolute() else paths.ROOT / p)


class CameraConfig(_Strict):
    id: int
    video: str  # файл, номер веб-камеры или URL потока
    lines: list[LineConfig] = []
    zones: list[ZoneConfig] = []

    resolve_video = field_validator("video")(_resolve_source)


class GridConfig(_Strict):
    width: int
    height: int
    origin: Point
    size: Point


class GroundTruthConfig(_Strict):
    file: Path
    grid: GridConfig

    resolve_file = field_validator("file")(_resolve_path)


class BirdsEyeConfig(_Strict):
    """Параметры вида сверху. Единицы — те же, что в калибровке сцены."""

    merge_radius: float  # точки от разных камер ближе этого — один человек
    match_radius: float  # для сравнения с ground truth: ближе этого — "попал"
    min_support: int = 1  # сколько камер минимум должны увидеть человека (2 отсекает ложные)
    zones: list[ZoneConfig] = []  # зоны на ПОЛУ, в координатах вида сверху


class SceneConfig(_Strict):
    name: str
    description: str = ""
    cameras: list[CameraConfig]
    calibration: Path | None = None
    # Во сколько раз кадр, на котором делали калибровку, больше нашего видео.
    # Пример: калибровали на 720x576, а видео 360x288 -> 2.0
    calibration_scale: float = 1.0
    ground_truth: GroundTruthConfig | None = None
    birdseye: BirdsEyeConfig | None = None

    resolve_calibration = field_validator("calibration")(_resolve_path)

    def camera(self, cam_id: int) -> CameraConfig:
        for cam in self.cameras:
            if cam.id == cam_id:
                return cam
        raise KeyError(f"В сцене {self.name} нет камеры {cam_id}")


def load_scene(name_or_path: str | Path) -> SceneConfig:
    """Принимает имя сцены ('lab6p') или путь к yaml."""
    path = Path(name_or_path)
    if not path.suffix:
        path = paths.CONFIGS / f"{name_or_path}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in paths.CONFIGS.glob("*.yaml")))
        raise FileNotFoundError(f"Сцена {name_or_path} не найдена. Доступны: {available}")
    return SceneConfig.model_validate(yaml.safe_load(path.read_text()))
