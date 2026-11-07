"""Дообучение (fine-tuning) YOLO26 на MOT17 и экспорт модели.

Transfer learning: берём сеть, уже обученную на COCO (118 тыс. фото, 80 классов),
и продолжаем обучение на НАШИХ данных. Сеть уже умеет видеть края, текстуры,
силуэты — мы лишь подстраиваем её под нашу задачу (пешеходы, камеры наблюдения,
толпы). Это в сотни раз дешевле, чем учить с нуля.

У COCO 80 классов, у нас один (person). Ultralytics сам пересоздаёт последний
слой классификации под 1 класс, а все остальные веса переносит.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from box_interceptor import paths
from box_interceptor.detection.detector import best_device

DEFAULT_DATA = paths.DATASETS / "mot17-yolo" / "data.yaml"


def train(
    base: str = "yolo26n",
    data: Path = DEFAULT_DATA,
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    fraction: float = 1.0,
    name: str | None = None,
    device: str | None = None,
    freeze: int | None = None,
    lr0: float | None = None,
    patience: int = 10,
) -> Path:
    """freeze=10 — заморозить первые 10 слоёв (backbone): они не обучаются, сохраняя
    универсальные признаки COCO. Учится только "голова". Полезно на маленьком датасете.
    lr0 — начальный learning rate (размер шага оптимизатора). Для дообучения — меньше.
    """
    from ultralytics import YOLO

    if not data.exists():
        raise FileNotFoundError(f"{data} не найден. Сначала: uv run bi prepare-yolo")
    weights = paths.MODELS / f"{base}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    name = name or f"{base}-mot17"
    model = YOLO(str(weights))
    extra = {}
    if lr0 is not None:
        # optimizer="auto" сам выбирает lr и игнорирует lr0, поэтому задаём оптимизатор явно
        extra |= {"optimizer": "AdamW", "lr0": lr0}
    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        fraction=fraction,  # доля train-данных (для быстрых экспериментов: 0.25)
        device=device or best_device(),
        project=str(paths.MODELS / "runs"),
        name=name,
        exist_ok=True,
        freeze=freeze,
        patience=patience,  # early stopping: стоп, если N эпох подряд нет улучшения на val
        plots=True,  # графики loss/mAP, матрица ошибок, примеры батчей
        workers=4,
        seed=0,  # воспроизводимость
        deterministic=False,  # true сильно замедляет на MPS
        **extra,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    target = paths.MODELS / f"{name}.pt"
    shutil.copy2(best, target)
    print(f"Лучшие веса: {target.relative_to(paths.ROOT)}  (графики: {Path(results.save_dir).relative_to(paths.ROOT)})")  # fmt: skip
    return target


def export(
    weights: str | Path, fmt: str = "coreml", imgsz: int = 640, half: bool | None = None
) -> Path:
    """Экспорт для быстрого инференса.

    coreml — формат Apple: модель исполняется на Neural Engine / GPU / CPU,
             CoreML сам выбирает. На Mac это обычно самый быстрый вариант.
    onnx   — универсальный формат (Windows, Linux, браузер, ONNX Runtime).
    half   — веса в float16: модель вдвое меньше, точность почти та же.
             Для CoreML включено по умолчанию: Neural Engine работает ТОЛЬКО
             с float16 — модель во float32 на нём просто не запустится
             (замер на M4: 13.5 мс -> 1.7 мс на кадр).
    """
    from ultralytics import YOLO

    if half is None:
        half = fmt == "coreml"
    path = Path(weights)
    if not path.exists() and len(path.parts) == 1:
        path = paths.MODELS / path.with_suffix(".pt").name
    # Голова: по умолчанию экспортёр Ultralytics выключает NMS-free (end-to-end) голову
    # YOLO26 и отдаёт "классическую" + NMS в Python. Мы проверили оба варианта для CoreML
    # float16 на MOT17: end-to-end (nms=False) — AP50 0.501, классическая — 0.548 при той же
    # скорости. Поэтому оставляем вариант по умолчанию. Мораль: после экспорта ВСЕГДА мерь точность.
    out = YOLO(str(path)).export(
        format=fmt,
        imgsz=imgsz,
        quantize=16 if half else None,  # 16 = float16 (в Ultralytics 8.4 заменяет half=True)
    )
    return Path(out)
