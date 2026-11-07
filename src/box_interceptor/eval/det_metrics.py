"""Метрики детекции, написанные с нуля — чтобы было видно, как они устроены.

Словарь:
  IoU (Intersection over Union) — насколько две рамки совпадают:
      площадь пересечения / площадь объединения. 1 — идеально, 0 — не касаются.
  TP (true positive)  — предсказание совпало с реальным человеком (IoU >= порога).
  FP (false positive) — предсказание есть, человека нет (ложная тревога).
  FN (false negative) — человек есть, предсказания нет (пропуск).
  Precision = TP / (TP + FP) — какая доля тревог настоящая.
  Recall    = TP / (TP + FN) — какую долю людей мы нашли.

  AP (Average Precision) — площадь под кривой precision(recall), когда мы
      двигаем порог уверенности от 1 до 0. Одно число, которое не зависит от
      выбранного порога. Считаем по-COCO: 101 точка recall 0.00, 0.01, ... 1.00.
  AP50     — AP при IoU >= 0.5 (рамка "примерно там").
  AP50-95  — среднее AP по порогам IoU 0.50, 0.55, ..., 0.95 (рамка ещё и точная).
  mAP      — среднее AP по классам. У нас класс один (person), так что mAP = AP.

Зоны игнорирования: если предсказание не совпало ни с одним человеком, но
лежит на манекене/отражении/почти невидимом человеке — его не считают ни TP,
ни FP. Так делают в MOTChallenge и CrowdHuman.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
RECALL_POINTS = np.linspace(0, 1, 101)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU всех пар рамок: a (N, 4), b (M, 4) в xyxy -> (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    inter = _intersection(a, b)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def box_ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over Area(a): какая доля рамки a лежит внутри b. (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    inter = _intersection(a, b)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    return np.divide(inter, area_a, out=np.zeros_like(inter), where=area_a > 0)


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def match_frame(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    ignore_boxes: np.ndarray | None = None,
    iou_threshold: float = 0.5,
    ignore_ioa: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Сопоставляет предсказания одного кадра с разметкой (жадно, по убыванию уверенности).

    Возвращает два булевых массива в порядке pred_boxes:
        is_tp      — совпало с человеком;
        is_ignored — не совпало, но лежит в зоне игнорирования.
    Всё остальное — FP.
    """
    n = len(pred_boxes)
    is_tp = np.zeros(n, dtype=bool)
    is_ignored = np.zeros(n, dtype=bool)
    if n == 0:
        return is_tp, is_ignored
    iou = box_iou(pred_boxes, gt_boxes)
    taken = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-np.asarray(pred_scores), kind="stable"):
        if len(gt_boxes):
            candidates = np.where(~taken & (iou[i] >= iou_threshold))[0]
            if len(candidates):
                best = candidates[np.argmax(iou[i, candidates])]
                taken[best] = True
                is_tp[i] = True
    if ignore_boxes is not None and len(ignore_boxes):
        in_ignore = box_ioa(pred_boxes, ignore_boxes).max(1) >= ignore_ioa
        is_ignored = ~is_tp & in_ignore
    return is_tp, is_ignored


def average_precision(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """AP по COCO: интерполированная precision в 101 точке recall."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    # "Огибающая": precision при recall r = максимум precision при recall >= r.
    # Так кривая становится невозрастающей и не зависит от мелких зубцов.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(sampled.mean())


def best_f1_point(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> dict[str, float]:
    """Порог уверенности, при котором F1 = 2PR/(P+R) максимален.

    У разных моделей уверенность "откалибрована" по-разному (у YOLO рабочий
    порог ~0.25, у RF-DETR ~0.5), поэтому сравнивать P/R при одном пороге
    нечестно. Сравниваем каждую модель в её лучшей рабочей точке.
    """
    if n_gt == 0 or len(scores) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "conf": float("nan")}
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    precision = tp / (tp + fp)
    recall = tp / n_gt
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)  # fmt: skip
    i = int(np.argmax(f1))
    return {
        "f1": float(f1[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "conf": float(scores[order][i]),
    }


@dataclass
class DetectionEvaluator:
    """Копит предсказания и разметку по кадрам, в конце считает метрики."""

    _scores: list[np.ndarray] = field(default_factory=list)
    _tp: dict[float, list[np.ndarray]] = field(default_factory=dict)
    _ignored: dict[float, list[np.ndarray]] = field(default_factory=dict)
    n_gt: int = 0

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        ignore_boxes: np.ndarray | None = None,
    ) -> None:
        pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
        pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
        self.n_gt += len(gt_boxes)
        self._scores.append(pred_scores)
        for thr in IOU_THRESHOLDS:
            tp, ign = match_frame(pred_boxes, pred_scores, gt_boxes, ignore_boxes, float(thr))
            self._tp.setdefault(float(thr), []).append(tp)
            self._ignored.setdefault(float(thr), []).append(ign)

    def compute(self) -> dict[str, float]:
        if not self._scores:
            raise ValueError("Нет данных: вызовите update() хотя бы раз")
        scores_all = np.concatenate(self._scores)
        aps = {}
        for thr in IOU_THRESHOLDS:
            tp = np.concatenate(self._tp[float(thr)])
            keep = ~np.concatenate(self._ignored[float(thr)])
            aps[float(thr)] = average_precision(scores_all[keep], tp[keep], self.n_gt)

        keep50 = ~np.concatenate(self._ignored[0.5])
        best = best_f1_point(scores_all[keep50], np.concatenate(self._tp[0.5])[keep50], self.n_gt)
        return {
            "AP50": aps[0.5],
            "AP75": aps[0.75],
            "AP50-95": float(np.nanmean(list(aps.values()))),
            "F1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "best_conf": best["conf"],
            "n_gt": self.n_gt,
        }
