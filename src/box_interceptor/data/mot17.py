"""Датасет MOT17 — эталон для детекции и трекинга пешеходов.

Сайт: https://motchallenge.net/data/MOT17/   Лицензия: CC BY-NC-SA 3.0.

Зачем он нам. У EPFL разметка — точки на полу, а не рамки. Чтобы честно
посчитать mAP детектора и HOTA/MOTA трекера, нужны РАМКИ на каждом кадре.
В MOT17 они есть: 7 обучающих видео, ~5300 кадров, ~112 тыс. рамок.

Особенности, которые важно понимать:
1. В архиве каждое видео лежит 3 раза (суффиксы -DPM, -FRCNN, -SDP) — отличаются
   только файлы готовых детекций det.txt, кадры одинаковые. Берём только -FRCNN.
2. gt.txt: frame,id,x,y,w,h,flag,class,visibility
     class: 1 пешеход, 2 человек на транспорте, 7 стоящий человек,
            8 "отвлекающий" объект (манекен, человек на плакате), 12 отражение...
     flag:  1 — учитывать при оценке, 0 — игнорировать
     visibility: доля видимой части человека 0..1
3. Делим train/val ПО ВИДЕО, а не по кадрам. Соседние кадры почти одинаковые:
   если перемешать кадры, модель "увидит" валидацию при обучении, и метрика
   будет завышена. Это называется утечкой данных (data leakage).
"""

from __future__ import annotations

import configparser
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from box_interceptor import paths
from box_interceptor.data.download import download

MOT17_URL = "https://motchallenge.net/data/MOT17.zip"
RAW_DIR = paths.RAW / "mot17"
TRAIN_ROOT = RAW_DIR / "MOT17" / "train"

ALL_SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
# Валидация: две статичные камеры, как у камер наблюдения.
# 04 — высокая точка обзора над площадью, 09 — камера на уровне глаз.
VAL_SEQUENCES = ["MOT17-04", "MOT17-09"]
TRAIN_SEQUENCES = [s for s in ALL_SEQUENCES if s not in VAL_SEQUENCES]

PERSON_CLASSES = (1, 2, 7)  # всё, что визуально "человек"
IGNORE_CLASSES = (8, 12)  # манекены, отражения: не штрафуем модель, если она их нашла
MIN_VISIBILITY = 0.1  # человека, видимого меньше чем на 10%, не требуем найти


# ---------- скачивание ----------


def download_and_extract(keep_zip: bool = False) -> Path:
    if all((seq_dir(s) / "gt" / "gt.txt").exists() for s in ALL_SEQUENCES):
        print(f"MOT17 уже распакован: {TRAIN_ROOT.relative_to(paths.ROOT)}")
        return TRAIN_ROOT
    zip_path = download(MOT17_URL, RAW_DIR / "MOT17.zip")
    print("Распаковываю только train/*-FRCNN (кадры в архиве продублированы 3 раза)...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if "/train/" in m and "-FRCNN/" in m]
        zf.extractall(RAW_DIR, members=members)  # extractall сам защищает от "../" в путях
    if not keep_zip:
        zip_path.unlink()
        print("Архив удалён, чтобы сэкономить ~5.9 ГБ.")
    return TRAIN_ROOT


# ---------- чтение ----------


@dataclass(frozen=True)
class SeqInfo:
    name: str
    path: Path
    length: int
    fps: float
    width: int
    height: int
    img_ext: str

    def frame_path(self, frame: int) -> Path:
        """Кадры нумеруются с 1: img1/000001.jpg"""
        return self.path / "img1" / f"{frame:06d}{self.img_ext}"


def seq_dir(name: str, root: Path = TRAIN_ROOT) -> Path:
    return root / f"{name}-FRCNN"


def read_seqinfo(path: Path) -> SeqInfo:
    cfg = configparser.ConfigParser()
    cfg.read(path / "seqinfo.ini")
    s = cfg["Sequence"]
    return SeqInfo(
        name=s["name"].replace("-FRCNN", ""),
        path=path,
        length=int(s["seqLength"]),
        fps=float(s["frameRate"]),
        width=int(s["imWidth"]),
        height=int(s["imHeight"]),
        img_ext=s.get("imExt", ".jpg"),
    )


@dataclass(frozen=True)
class FrameAnnotations:
    boxes: np.ndarray  # (N, 4) xyxy — люди, которых модель ДОЛЖНА найти
    ignore: np.ndarray  # (M, 4) xyxy — зоны, где находки не считаются ни ошибкой, ни успехом


def load_gt(path: Path) -> dict[int, FrameAnnotations]:
    """gt/gt.txt -> {номер кадра: аннотации}. Кадры без людей тоже есть (пустые массивы)."""
    info = read_seqinfo(path)
    raw = np.loadtxt(path / "gt" / "gt.txt", delimiter=",", ndmin=2)
    frames = raw[:, 0].astype(int)
    xywh = raw[:, 2:6]
    xyxy = np.hstack([xywh[:, :2], xywh[:, :2] + xywh[:, 2:4]])
    cls = raw[:, 7].astype(int)
    vis = raw[:, 8]

    is_person = np.isin(cls, PERSON_CLASSES)
    positive = is_person & (vis >= MIN_VISIBILITY)
    ignore = np.isin(cls, IGNORE_CLASSES) | (is_person & (vis < MIN_VISIBILITY))

    result = {}
    for f in range(1, info.length + 1):
        m = frames == f
        result[f] = FrameAnnotations(xyxy[m & positive], xyxy[m & ignore])
    return result


# ---------- конвертация в формат YOLO ----------


def to_yolo_lines(boxes_xyxy: np.ndarray, width: int, height: int) -> list[str]:
    """Рамки xyxy в пикселях -> строки YOLO 'класс cx cy w h' в долях от размера кадра."""
    lines = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        w, h = x2 - x1, y2 - y1
        if w < 2 or h < 2:  # рамка почти целиком за краем кадра
            continue
        cx, cy = x1 + w / 2, y1 + h / 2
        lines.append(f"0 {cx / width:.6f} {cy / height:.6f} {w / width:.6f} {h / height:.6f}")
    return lines


def build_yolo_dataset(
    out_dir: Path = paths.DATASETS / "mot17-yolo",
    train_step: int = 3,
    val_step: int = 5,
    root: Path = TRAIN_ROOT,
    train_sequences: list[str] = TRAIN_SEQUENCES,
    val_sequences: list[str] = VAL_SEQUENCES,
) -> Path:
    """Собирает датасет YOLO: images/{train,val}, labels/{train,val}, data.yaml.

    Картинки не копируем, а делаем симлинки — экономим ~1 ГБ диска.
    train_step=3: берём каждый 3-й кадр. При 30 к/с соседние кадры почти
    одинаковые — модель от них умнее не станет, а обучение замедлится.
    """
    for split, sequences, step in (
        ("train", train_sequences, train_step),
        ("val", val_sequences, val_step),
    ):
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for name in sequences:
            info = read_seqinfo(seq_dir(name, root))
            gt = load_gt(info.path)
            for frame in range(1, info.length + 1, step):
                stem = f"{name}_{frame:06d}"
                link = img_dir / f"{stem}{info.img_ext}"
                if not link.exists():
                    os.symlink(info.frame_path(frame), link)
                lines = to_yolo_lines(gt[frame].boxes, info.width, info.height)
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n_img += 1
                n_box += len(lines)
        print(f"  {split}: {n_img} кадров, {n_box} рамок ({', '.join(sequences)})")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n"
    )
    return data_yaml
