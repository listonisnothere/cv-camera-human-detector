"""Пути проекта.

Всё тяжёлое (сырые данные, видео, веса моделей, результаты) лежит в папках,
которые исключены из git (см. .gitignore). В репозитории только код и конфиги.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"  # как скачали: avi, zip, txt
VIDEOS = DATA / "videos"  # перекодированные mp4, готовые к работе
DATASETS = DATA / "datasets"  # датасеты в формате для обучения (YOLO)

MODELS = ROOT / "models"  # скачанные и обученные веса
OUTPUTS = ROOT / "outputs"  # видео с разметкой, таблицы метрик, графики
CONFIGS = ROOT / "configs"  # YAML-описания сцен (зоны, линии, камеры)
