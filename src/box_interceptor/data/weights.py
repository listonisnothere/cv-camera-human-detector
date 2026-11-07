"""Предзагрузка весов моделей.

Ultralytics и RF-DETR сами скачивают веса при первом запуске, но в одно
соединение. Веса RF-DETR весят 350+ МБ (это полный чекпоинт обучения), и на
медленной сети это час. Здесь качаем кусками параллельно и сверяем MD5.
"""

from __future__ import annotations

from pathlib import Path

from box_interceptor import paths
from box_interceptor.data.download import download

YOLO_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
RFDETR_FILES = {"rfdetr-nano": "rf-detr-nano.pth", "rfdetr-small": "rf-detr-small.pth",
                "rfdetr-medium": "rf-detr-medium.pth"}  # fmt: skip


def download_weights(names: list[str]) -> list[Path]:
    done = []
    for name in names:
        if name in RFDETR_FILES:
            from rfdetr.assets.model_weights import ModelWeights, get_model_cache_dir

            fname = RFDETR_FILES[name]
            url, md5 = ModelWeights.get_url(fname), ModelWeights.get_md5(fname)
            dest = Path(get_model_cache_dir()) / fname  # туда, где RF-DETR будет их искать
        else:
            url, md5 = f"{YOLO_RELEASE}/{name}.pt", None
            dest = paths.MODELS / f"{name}.pt"
        print(f"→ {name}: {url}")
        done.append(download(url, dest, parts=16, md5=md5))
        print(f"  ✓ {dest}")
    return done
