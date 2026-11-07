import numpy as np
import pytest

from box_interceptor.eval.det_metrics import (
    DetectionEvaluator,
    average_precision,
    box_ioa,
    box_iou,
    match_frame,
)

BOX = np.array([[0, 0, 10, 10]])


def test_iou_basic_cases():
    assert box_iou(BOX, BOX)[0, 0] == pytest.approx(1.0)
    assert box_iou(BOX, [[20, 20, 30, 30]])[0, 0] == 0.0
    # половина рамки перекрыта: пересечение 50, объединение 150
    assert box_iou(BOX, [[5, 0, 15, 10]])[0, 0] == pytest.approx(50 / 150)


def test_ioa_is_asymmetric():
    small, big = np.array([[2, 2, 4, 4]]), np.array([[0, 0, 10, 10]])
    assert box_ioa(small, big)[0, 0] == pytest.approx(1.0)  # маленькая целиком внутри большой
    assert box_ioa(big, small)[0, 0] == pytest.approx(4 / 100)


def test_match_one_gt_cannot_be_used_twice():
    preds = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    tp, ign = match_frame(preds, np.array([0.9, 0.8]), BOX)
    assert tp.tolist() == [True, False]  # дубликат — это FP
    assert ign.tolist() == [False, False]


def test_match_ignore_region():
    preds = np.array([[50, 50, 60, 60]])
    tp, ign = match_frame(preds, np.array([0.9]), BOX, ignore_boxes=np.array([[45, 45, 65, 65]]))
    assert not tp[0] and ign[0]


def test_ap_perfect_and_false_positive_first():
    assert average_precision(np.array([0.9]), np.array([True]), n_gt=1) == pytest.approx(1.0)
    # самая уверенная находка — ложная: precision при полном recall = 1/2
    assert average_precision(np.array([0.9, 0.8]), np.array([False, True]), 1) == pytest.approx(0.5)
    assert average_precision(np.array([]), np.array([], dtype=bool), 1) == 0.0


def test_evaluator_end_to_end():
    ev = DetectionEvaluator()
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]])
    # кадр 1: обе рамки найдены; кадр 2: один пропуск и ложная тревога в зоне игнорирования
    ev.update(gt, [0.9, 0.8], gt)
    ev.update([[100, 100, 110, 110]], [0.95], gt[:1], ignore_boxes=[[95, 95, 115, 115]])
    m = ev.compute()
    assert m["n_gt"] == 3
    assert m["precision"] == pytest.approx(1.0)  # ложная тревога в зоне игнорирования не считается
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["F1"] == pytest.approx(0.8)
    assert 0.6 < m["AP50"] < 0.7  # recall не дотягивает до 1 -> AP ~ 2/3


def test_best_f1_picks_threshold():
    from box_interceptor.eval.det_metrics import best_f1_point

    # 2 человека. Уверенности: 0.9 TP, 0.8 FP, 0.7 TP, 0.1 FP
    r = best_f1_point(np.array([0.9, 0.8, 0.7, 0.1]), np.array([True, False, True, False]), 2)
    assert r["conf"] == pytest.approx(
        0.7
    )  # на пороге 0.7: P=2/3, R=1 -> F1=0.8 (лучше, чем 0.9: F1=0.67)
    assert r["f1"] == pytest.approx(0.8)


def test_matches_supervision_map():
    """Наша реализация mAP должна совпадать с библиотечной (supervision) на случайных данных."""
    sv = pytest.importorskip("supervision")
    from supervision.metrics import MeanAveragePrecision

    rng = np.random.default_rng(0)
    ours = DetectionEvaluator()
    lib = MeanAveragePrecision()
    for _ in range(30):
        n = rng.integers(1, 6)
        xy = rng.uniform(0, 500, size=(n, 2))
        gt = np.hstack([xy, xy + rng.uniform(20, 80, size=(n, 2))])
        noisy = gt + rng.normal(0, 6, size=gt.shape)  # почти верные рамки
        fp_xy = rng.uniform(0, 500, size=(2, 2))
        false_alarms = np.hstack([fp_xy, fp_xy + 40])  # две случайные ложные рамки
        pred = np.vstack([noisy, false_alarms])
        scores = rng.uniform(0.05, 1, size=len(pred))
        ours.update(pred, scores, gt)
        lib.update(
            sv.Detections(xyxy=pred, confidence=scores, class_id=np.zeros(len(pred), int)),
            sv.Detections(xyxy=gt, class_id=np.zeros(len(gt), int)),
        )
    res = lib.compute()
    m = ours.compute()
    assert m["AP50"] == pytest.approx(res.map50, abs=0.01)
    assert m["AP50-95"] == pytest.approx(res.map50_95, abs=0.01)
