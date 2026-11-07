"""Детекторы людей с единым интерфейсом.

Любой детектор здесь — это функция "кадр BGR -> sv.Detections", где:
    detections.xyxy        (N, 4)  рамки [x1, y1, x2, y2] в пикселях
    detections.confidence  (N,)    уверенность модели 0..1
    detections.class_id    (N,)    всегда 0 = person (мы приводим к этому)

Благодаря общему интерфейсу трекер, аналитика и метрики не знают, какая
модель внутри — YOLO26, RF-DETR или наша дообученная. Их можно менять одной
строкой в CLI: --model yolo26n | yolo26s | rfdetr-nano | models/my.pt

Две архитектуры:
  YOLO26 (Ultralytics) — свёрточная сеть (CNN). Очень быстрая, предсказывает
      рамки напрямую, без пост-обработки NMS (end-to-end).
  RF-DETR (Roboflow) — трансформер (DETR = DEtection TRansformer). Смотрит на
      картинку целиком через механизм внимания; обычно точнее на сложных сценах,
      но тяжелее.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import supervision as sv

from box_interceptor import paths


def best_device() -> str:
    """cuda (NVIDIA) > mps (GPU Apple Silicon) > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PersonDetector(Protocol):
    name: str

    def __call__(self, frame_bgr: np.ndarray) -> sv.Detections: ...


def _only_person(det: sv.Detections, person_class: int) -> sv.Detections:
    det = det[det.class_id == person_class]
    det.class_id = np.zeros(len(det), dtype=int)
    return det


class YoloDetector:
    """Ultralytics YOLO (по умолчанию YOLO26). В COCO класс 0 — person."""

    def __init__(
        self,
        weights: str | Path = "yolo26n.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        path = Path(weights)
        if not path.exists() and len(path.parts) == 1:
            # Просто имя ("yolo26n.pt") — Ultralytics скачает веса, кладём их в models/
            path = paths.MODELS / path.name
            path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(path))
        self.name = path.name.removesuffix(".pt")
        self.conf = conf
        self.imgsz = imgsz
        # Экспортированные модели (CoreML и т.п.) сами выбирают железо
        self.device = (device or best_device()) if path.suffix == ".pt" else None
        names = {v: k for k, v in self.model.names.items()}
        self.person_class = names.get("person", 0)

    def __call__(self, frame_bgr: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame_bgr,  # Ultralytics ожидает BGR (как OpenCV) — конвертировать не нужно
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            classes=[self.person_class],
            verbose=False,
        )[0]
        return _only_person(sv.Detections.from_ultralytics(result), self.person_class)


class RfDetrDetector:
    """RF-DETR от Roboflow. Возвращает классы в нумерации COCO (person = 1)."""

    SIZES = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "base": "RFDETRBase",
        "large": "RFDETRLarge",
    }

    def __init__(self, size: str = "nano", conf: float = 0.5, optimize: bool = True):
        import logging

        import rfdetr

        logging.getLogger("rf-detr").setLevel(logging.ERROR)  # библиотека очень многословна
        if size not in self.SIZES:
            raise ValueError(f"RF-DETR: размер {size} неизвестен, есть {list(self.SIZES)}")
        model_cls = getattr(rfdetr, self.SIZES[size])
        self.model = model_cls()
        if optimize:
            # Трассирует граф сети заранее (как экспорт, но внутри PyTorch) — быстрее инференс.
            try:
                self.model.inference(compile=True)  # torch.jit.trace графа сети
            except Exception as exc:  # noqa: BLE001 — на некоторых бэкендах трассировка не работает
                print(f"RF-DETR: оптимизация недоступна ({exc}), работаем без неё")
        self.name = f"rfdetr-{size}"
        self.conf = conf
        from rfdetr.assets.coco_classes import COCO_CLASSES  # {1: "person", 2: "bicycle", ...}

        self.person_class = next(k for k, v in COCO_CLASSES.items() if v == "person")

    def __call__(self, frame_bgr: np.ndarray) -> sv.Detections:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])  # RF-DETR ожидает RGB!
        det = self.model.predict(rgb, threshold=self.conf)
        return _only_person(det, self.person_class)


def build_detector(spec: str, conf: float | None = None, imgsz: int = 640) -> PersonDetector:
    """Фабрика: строка из CLI -> детектор.

    'yolo26n', 'yolo26s', 'models/x.pt', 'models/x.mlpackage' -> YoloDetector
    'rfdetr-nano', 'rfdetr-small', ...                       -> RfDetrDetector
    """
    if spec.startswith("rfdetr"):
        size = spec.split("-", 1)[1] if "-" in spec else "nano"
        return RfDetrDetector(size, conf=0.5 if conf is None else conf)
    path = Path(spec)
    if not path.suffix:
        path = path.with_suffix(".pt")
    return YoloDetector(path, conf=0.25 if conf is None else conf, imgsz=imgsz)
