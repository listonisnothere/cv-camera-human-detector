"""Трекинг: даём каждому человеку постоянный номер (track ID) между кадрами.

Детектор смотрит на каждый кадр независимо: "здесь 5 людей". Он не знает,
что человек в рамке слева — тот же, что был там кадр назад. Трекер связывает
детекции во времени. Без этого нельзя посчитать, сколько людей ПРОШЛО через
линию (иначе один человек засчитается 25 раз в секунду).

ByteTrack (2021) — простой и очень сильный трекер, стандарт индустрии:
  1. Для каждого трека фильтр Калмана предсказывает, где рамка будет на
     следующем кадре (модель "постоянной скорости").
  2. Детекции с ВЫСОКОЙ уверенностью (>= high_conf) сопоставляются с
     предсказаниями по IoU (венгерский алгоритм — оптимальное назначение пар).
  3. Главная идея ByteTrack: оставшиеся треки пробуем сопоставить с детекциями
     НИЗКОЙ уверенности. Когда человека частично закрыли, детектор видит его
     с уверенностью 0.1 — обычные трекеры такую детекцию выбросили бы и
     потеряли человека. ByteTrack — нет.
  4. Несопоставленные уверенные детекции (>= activation) -> новые треки.
     Треки без детекций живут ещё lost_buffer кадров (вдруг человек вынырнет
     из-за колонны).

Пороги трекера ДОЛЖНЫ соответствовать детектору: если детектор в толпе редко
даёт уверенность выше 0.5, а трекер создаёт треки только от 0.7 — половина
людей так и останется без трека. Подбор — в docs/03-tracking.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # нужен только для подсказок типов — не тянем тяжёлый импорт в CLI
    import supervision as sv


class Tracker(Protocol):
    def update(self, detections: sv.Detections) -> sv.Detections: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class TrackerParams:
    high_conf: float = 0.6  # 1-й шаг: "уверенные" детекции
    activation: float = 0.7  # уверенность, нужная для рождения нового трека
    lost_buffer: int = 30  # сколько кадров помнить потерянный трек


# Подобрано на MOT17 val для YOLO26 (см. docs/03-tracking.md): уверенности YOLO26
# в толпе заметно ниже, чем у детекторов, под которые подбирались значения по умолчанию.
TUNED = TrackerParams(high_conf=0.2, activation=0.3)

# При трекинге детектор отдаёт и слабые детекции: ByteTrack использует их на 2-м шаге,
# а новые треки всё равно рождаются только от уверенности >= TUNED.activation.
TRACK_DET_CONF = 0.1


def make_tracker(fps: float = 25.0, params: TrackerParams = TUNED) -> Tracker:
    """ByteTrack из пакета `trackers` (Roboflow)."""
    from trackers import ByteTrackTracker

    return ByteTrackTracker(
        frame_rate=fps,
        high_conf_det_threshold=params.high_conf,
        track_activation_threshold=params.activation,
        lost_track_buffer=params.lost_buffer,
    )
