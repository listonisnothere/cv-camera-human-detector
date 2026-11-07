# box-interceptor

Учебный проект по Computer Vision: ИИ находит людей на видео с камер,
ведёт их (трекинг), считает в зонах и на линиях, сводит 4 камеры в один
**вид сверху** — и всё это измерено честными метриками.

Стек: **YOLO26** (Ultralytics) · **RF-DETR** (Roboflow) · **ByteTrack** ·
**supervision** · PyTorch на GPU Apple Silicon (MPS) · CoreML · uv.

![4 камеры лаборатории EPFL: трекинг, зона, линия подсчёта и вид сверху](docs/img/multicam_lab6p.jpg)

*4 синхронные камеры: рамки с ID трека, зона "ковёр", линия подсчёта. Справа —
вид сверху: цветные точки — люди с каждой камеры, чёрные — итог после слияния.*

## Быстрый старт

```bash
uv sync                                   # окружение (Python 3.12 + всё нужное)
uv run bi download models                 # веса YOLO26n/s и RF-DETR (параллельно, с проверкой MD5)
uv run bi download epfl                   # видео 4 камер EPFL (~950 МБ)
uv run bi web                             # пульт камер в браузере: ИИ в реальном времени
uv run bi multicam lab6p                  # 4 камеры + вид сверху -> outputs/multicam/lab6p.mp4
```

![пульт камер в браузере](docs/img/web_panel.jpeg)

## Что получилось (Apple M4)

| Что | Результат | Урок |
|---|---|---|
| Детекция, MOT17 val | YOLO26n AP50 0.558 @ 78 FPS · YOLO26s 0.588 · RF-DETR-nano 0.569 | [6](docs/06-metrics.md) |
| Вход 1280 вместо 640 | YOLO26n AP50 0.558 → **0.706** (мелкие люди в толпе) | [6](docs/06-metrics.md) |
| Пороги ByteTrack под детектор | HOTA 0.248 → **0.375** (на входе 1280 — 0.465) | [3](docs/03-tracking.md) |
| Вид сверху, 4 камеры (lab6p) | F1 одной камеры 0.82–0.88 → слияние **0.960**, precision 1.0 | [5](docs/05-multicam-birdseye.md) |
| Дообучение на MOT17 | AP50 0.558 → **0.662** на MOT17, но **хуже** на тёмных камерах EPFL | [7](docs/07-training.md) |
| CoreML float16 (Neural Engine) | 11.6 → **7.0 мс/кадр** (142 FPS), точность та же | [8](docs/08-deployment.md) |

Своё видео или веб-камера:

```bash
uv run bi detect path/to/video.mp4        # только рамки
uv run bi track path/to/video.mp4 --heatmap
uv run bi live                            # камера Mac в реальном времени (q — выход)
uv run bi live rtsp://user:pass@ip:554/stream   # IP-камера
```

## Все команды

| Команда | Что делает |
|---|---|
| `bi download epfl` / `mot17` / `models` | Скачать датасеты и веса моделей |
| `bi prepare-yolo` | MOT17 → формат YOLO для обучения |
| `bi web` | Пульт камер в браузере: живая обработка, выбор сцены и модели |
| `bi detect SOURCE` | Детекция людей на фото/видео |
| `bi track [SOURCE] --scene S` | Трекинг + линии/зоны/тепловая карта |
| `bi live [SOURCE]` | Реальное время с веб-камеры или RTSP |
| `bi multicam SCENE` | Мозаика 2x2 + вид сверху |
| `bi birdseye SCENE` | Точность вида сверху по ground truth |
| `bi eval-det --models ...` | Сравнение детекторов: AP, F1, FPS |
| `bi eval-track --models ...` | Метрики трекинга: HOTA, MOTA, IDF1 |
| `bi train [--freeze 10 --lr0 0.0005]` | Дообучение YOLO26 на MOT17 |
| `bi export WEIGHTS` | Экспорт в CoreML / ONNX |
| `bi bench --models ...` | Скорость моделей |

`--model` везде принимает: `yolo26n`, `yolo26s`, `rfdetr-nano`, `rfdetr-small`,
путь к своим весам `models/*.pt` или к CoreML `models/*.mlpackage`.

## Уроки

Читать по порядку — каждый урок объясняет код, который ты уже можешь запустить.

0. [Окружение](docs/00-setup.md) — uv, PyTorch, MPS, структура проекта
1. [Данные](docs/01-datasets.md) — EPFL и MOT17, форматы разметки, ловушки
2. [Детекция](docs/02-detection.md) — рамки, уверенность, YOLO26 vs RF-DETR
3. [Трекинг](docs/03-tracking.md) — Калман, венгерский алгоритм, ByteTrack
4. [Видеоаналитика](docs/04-analytics.md) — линии, зоны, тепловые карты
5. [Несколько камер и вид сверху](docs/05-multicam-birdseye.md) — гомография, слияние камер
6. [Метрики](docs/06-metrics.md) — IoU, precision/recall, mAP, HOTA/MOTA/IDF1
7. [Дообучение](docs/07-training.md) — transfer learning, loss, переобучение, утечка данных
8. [Ускорение](docs/08-deployment.md) — CoreML, Neural Engine, скорость
9. [Пульт камер в браузере](docs/09-web.md) — реальное время, WebSocket, пропуск кадров

## Тесты

```bash
uv run pytest
uv run ruff check .
```

## Лицензии

Код проекта — твой. Зависимости и данные:
- **Ultralytics YOLO — AGPL-3.0.** Для учёбы и open source — свободно. Для закрытого
  коммерческого продукта нужна коммерческая лицензия Ultralytics (или модель с
  другой лицензией, например RF-DETR — Apache-2.0).
- **EPFL POM** — только исследования, с ссылкой на статью POM (TPAMI 2008).
- **MOT17** — CC BY-NC-SA 3.0 (некоммерческое использование).
