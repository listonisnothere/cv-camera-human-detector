"""Командная строка проекта. Запуск: `uv run bi --help`.

Каждая команда — обычная функция Python. Typer превращает аргументы функции
в аргументы командной строки, а docstring — в справку.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import cv2
import typer

from box_interceptor import paths
from box_interceptor.tracking.tracker import TRACK_DET_CONF

app = typer.Typer(
    help="box-interceptor: детекция, трекинг и аналитика людей на видео с камер.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
download_app = typer.Typer(help="Скачать датасеты.", no_args_is_help=True)
app.add_typer(download_app, name="download")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

Model = Annotated[str, typer.Option("--model", "-m", help="yolo26n | yolo26s | rfdetr-nano | путь к .pt/.mlpackage")]  # fmt: skip
Conf = Annotated[float | None, typer.Option(help="Порог уверенности (по умолчанию: YOLO 0.25, RF-DETR 0.5)")]  # fmt: skip
Out = Annotated[Path | None, typer.Option("--out", "-o", help="Куда сохранить результат")]
Show = Annotated[bool, typer.Option(help="Показывать окно с видео (q — выход)")]
MaxFrames = Annotated[int | None, typer.Option(help="Обработать не больше N кадров")]


def _models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


# ---------------------------------------------------------------- данные


@download_app.command("epfl")
def download_epfl(
    scene: Annotated[list[str] | None, typer.Option(help="Сцены (можно несколько)")] = None,
) -> None:
    """EPFL multi-camera: видео 4 камер, калибровка, ground truth -> data/videos/epfl/."""
    from box_interceptor.data import epfl

    epfl.download(scene or list(epfl.SEQUENCES))


@download_app.command("mot17")
def download_mot17(keep_zip: bool = False) -> None:
    """MOT17 (5.9 ГБ архив) -> распаковываем только нужное (861 МБ)."""
    from box_interceptor.data import mot17

    mot17.download_and_extract(keep_zip=keep_zip)


@download_app.command("models")
def download_models(
    name: Annotated[
        list[str] | None, typer.Option(help="yolo26n, yolo26s, rfdetr-nano, ...")
    ] = None,
) -> None:
    """Веса моделей (параллельно, с проверкой MD5) — быстрее встроенной загрузки."""
    from box_interceptor.data.weights import download_weights

    download_weights(name or ["yolo26n", "yolo26s", "rfdetr-nano"])


@app.command("prepare-yolo")
def prepare_yolo(train_step: int = 3, val_step: int = 5) -> None:
    """MOT17 -> датасет в формате YOLO (для обучения): data/datasets/mot17-yolo/."""
    from box_interceptor.data import mot17

    data_yaml = mot17.build_yolo_dataset(train_step=train_step, val_step=val_step)
    typer.echo(f"Готово: {_rel(data_yaml)}")


# ---------------------------------------------------------------- видео


def _rel(path: Path) -> str:
    """Путь для печати: относительно корня проекта, если возможно."""
    path = Path(path).resolve()
    return str(path.relative_to(paths.ROOT)) if path.is_relative_to(paths.ROOT) else str(path)


def _default_out(kind: str, source: str, suffix: str = ".mp4") -> Path:
    stem = Path(source).stem if not source.isdigit() else f"camera{source}"
    parent = Path(source).parent.name if not source.isdigit() else "live"
    return paths.OUTPUTS / kind / f"{parent}_{stem}{suffix}"


def _run_video(pipeline, source: str, out: Path | None, show: bool, max_frames: int | None) -> None:  # noqa: ANN001
    from rich.progress import Progress

    from box_interceptor.video.io import FfmpegWriter, FrameSource, Window

    with FrameSource(source) as src:
        writer = FfmpegWriter(out, src.fps, (src.width, src.height)) if out else None
        window = Window() if show else None
        total = min(filter(None, [src.frame_count, max_frames]), default=None)
        people, frames, t0 = 0, 0, time.perf_counter()
        last = None
        with Progress() as progress:
            task = progress.add_task("обработка", total=total)
            for i, frame in enumerate(src):
                if max_frames is not None and i >= max_frames:
                    break
                last = pipeline.process(frame)
                people += last.people
                frames += 1
                if writer:
                    writer.write(last.frame)
                if window and not window.show(last.frame):
                    break
                progress.advance(task)
        if writer:
            writer.close()
        if window:
            window.close()
    elapsed = time.perf_counter() - t0
    typer.echo(f"Кадров: {frames}, в среднем людей в кадре: {people / max(frames, 1):.1f}, "
               f"скорость: {frames / elapsed:.1f} к/с")  # fmt: skip
    if last:
        for name, (n_in, n_out) in last.line_counts.items():
            typer.echo(f"Линия '{name}': вошли {n_in}, вышли {n_out}")
    if out:
        typer.echo(f"Сохранено: {_rel(out)}")


@app.command()
def detect(
    source: Annotated[str, typer.Argument(help="Картинка, видео, номер веб-камеры или rtsp://")],
    model: Model = "yolo26n",
    conf: Conf = None,
    imgsz: int = 640,
    out: Out = None,
    show: Show = False,
    max_frames: MaxFrames = None,
) -> None:
    """Детекция людей (без трекинга): рамки + уверенность."""
    from box_interceptor.analytics.pipeline import CameraPipeline
    from box_interceptor.detection.detector import build_detector

    detector = build_detector(model, conf, imgsz)
    pipeline = CameraPipeline(detector, track=False, trace=False)
    if Path(source).suffix.lower() in IMAGE_SUFFIXES:
        img = cv2.imread(source)
        if img is None:
            raise typer.BadParameter(f"Не удалось прочитать {source}")
        result = pipeline.process(img)
        out = out or _default_out("detect", source, ".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result.frame)
        typer.echo(f"Людей: {result.people}. Сохранено: {_rel(out)}")
        return
    _run_video(pipeline, source, out or _default_out("detect", source), show, max_frames)


@app.command()
def track(
    source: Annotated[
        str | None, typer.Argument(help="Видео/камера. Если не указано — берётся из --scene")
    ] = None,  # fmt: skip
    scene: Annotated[str | None, typer.Option(help="Сцена из configs/ (линии, зоны)")] = None,
    camera: Annotated[int, typer.Option(help="Номер камеры в сцене")] = 0,
    model: Model = "yolo26n",
    conf: Conf = None,
    heatmap: Annotated[bool, typer.Option(help="Тепловая карта перемещений")] = False,
    out: Out = None,
    show: Show = False,
    max_frames: MaxFrames = None,
) -> None:
    """Трекинг (ID людей) + подсчёт пересечений линий и людей в зонах."""
    from box_interceptor.analytics.pipeline import CameraPipeline
    from box_interceptor.config import load_scene
    from box_interceptor.detection.detector import build_detector

    cam_cfg = load_scene(scene).camera(camera) if scene else None
    source = source or (str(cam_cfg.video) if cam_cfg else None)
    if source is None:
        raise typer.BadParameter("Укажите видео или --scene")
    from box_interceptor.video.io import FrameSource

    with FrameSource(source) as probe:
        fps = probe.fps  # трекеру (фильтру Калмана) важна реальная частота кадров
    detector = build_detector(model, TRACK_DET_CONF if conf is None else conf)
    pipeline = CameraPipeline(detector, cam_cfg, fps=fps, heatmap=heatmap)
    _run_video(pipeline, source, out or _default_out("track", source), show, max_frames)


@app.command()
def live(
    source: Annotated[str, typer.Argument(help="0 — встроенная камера Mac, или rtsp://...")] = "0",
    model: Model = "yolo26n",
    conf: Conf = None,
    out: Out = None,
) -> None:
    """Реальное время: веб-камера или IP-камера. Окно с FPS; q — выход."""
    from box_interceptor.analytics.pipeline import CameraPipeline
    from box_interceptor.detection.detector import build_detector
    from box_interceptor.video.io import FfmpegWriter, FrameSource, Window
    from box_interceptor.video.mosaic import draw_label

    detector = build_detector(model, TRACK_DET_CONF if conf is None else conf)
    window = Window("live (q — выход)")
    with FrameSource(source) as src:
        pipeline = CameraPipeline(detector, fps=src.fps)  # трекеру важна реальная частота кадров
        writer = FfmpegWriter(out, src.fps, (src.width, src.height)) if out else None
        t_prev = time.perf_counter()
        fps = 0.0
        for frame in src:
            result = pipeline.process(frame)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)  # сглаженный FPS
            t_prev = now
            draw_label(result.frame, f"{fps:.1f} FPS | people: {result.people}", (8, 8))
            if writer:
                writer.write(result.frame)
            if not window.show(result.frame):
                break
        if writer:
            writer.close()
    window.close()


# ---------------------------------------------------------------- несколько камер


@app.command()
def multicam(
    scene: Annotated[str, typer.Argument(help="Сцена: lab6p | terrace1 | passageway1")],
    model: Model = "yolo26n",
    conf: Conf = None,
    birdseye: Annotated[
        bool, typer.Option(help="Добавить вид сверху (если есть калибровка)")
    ] = True,
    out: Out = None,
    show: Show = False,
    max_frames: MaxFrames = None,
) -> None:
    """Все камеры сцены сразу: мозаика 2x2 + вид сверху."""
    from rich.progress import Progress

    from box_interceptor.analytics.multicam import MultiCamRunner
    from box_interceptor.config import load_scene
    from box_interceptor.detection.detector import build_detector
    from box_interceptor.video.io import FfmpegWriter, Window

    cfg = load_scene(scene)
    runner = MultiCamRunner(cfg, build_detector(model, TRACK_DET_CONF if conf is None else conf), birdseye=birdseye)  # fmt: skip
    out = out or paths.OUTPUTS / "multicam" / f"{cfg.name}.mp4"
    writer = None
    window = Window(f"multicam {cfg.name}") if show else None
    total = min(filter(None, [runner.frame_count, max_frames]), default=None)
    t0 = time.perf_counter()
    n = 0
    with Progress() as progress:
        task = progress.add_task(cfg.name, total=total)
        for idx, frames in runner.frames():
            if max_frames is not None and idx >= max_frames:
                break
            canvas = runner.step(idx, frames)
            if writer is None:
                writer = FfmpegWriter(out, runner.fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            n += 1
            if window and not window.show(canvas):
                break
            progress.advance(task)
    runner.close()
    if writer:
        writer.close()
    if window:
        window.close()
    typer.echo(f"Кадров: {n} x {len(cfg.cameras)} камер, {n / (time.perf_counter() - t0):.1f} к/с. "
               f"Сохранено: {_rel(out)}")  # fmt: skip


@app.command("birdseye")
def birdseye_eval(
    scene: Annotated[str, typer.Argument(help="Сцена с калибровкой: lab6p | passageway1")],
    model: Model = "yolo26n",
    conf: Conf = None,
    max_frames: MaxFrames = None,
    sweep: Annotated[
        bool, typer.Option(help="Перебрать радиус слияния и мин. число камер")
    ] = False,
) -> None:
    """Проверить вид сверху по ground truth: каждая камера отдельно vs все вместе."""
    import numpy as np
    import pandas as pd

    from box_interceptor.analytics.multicam import TopView, collect_birdseye_samples, score_birdseye
    from box_interceptor.config import load_scene
    from box_interceptor.detection.detector import build_detector
    from box_interceptor.eval.report import save_table, to_markdown

    def row(r) -> dict:  # noqa: ANN001
        f1 = 2 * r.precision * r.recall / max(r.precision + r.recall, 1e-9)
        err = float(np.mean(r.errors)) if r.errors else float("nan")
        return {"precision": r.precision, "recall": r.recall, "F1": f1, "mean_error": err,
                "TP": r.tp, "FP": r.fp, "FN": r.fn}  # fmt: skip

    cfg = load_scene(scene)
    top = TopView.from_scene(cfg)
    if top is None:
        raise typer.BadParameter(f"У сцены {scene} нет калибровки/ground truth")
    samples = collect_birdseye_samples(cfg, build_detector(model, conf), top, max_frames)
    cam_ids = [c.id for c in cfg.cameras]
    tag = f"{cfg.name}_{Path(model).stem}"

    totals = score_birdseye(samples, cam_ids, top.merge_radius, top.match_radius, top.min_support)
    df = pd.DataFrame([{"source": k, **row(r)} for k, r in totals.items()])
    typer.echo(to_markdown(df))
    typer.echo(f"\nСохранено: {_rel(save_table(df, f'birdseye_{tag}'))}")

    if sweep:
        rows = []
        for support in (1, 2):
            for radius in np.linspace(0.25, 2.0, 8) * top.merge_radius:
                fusion = score_birdseye(samples, cam_ids, radius, top.match_radius, support)[
                    "fusion"
                ]
                rows.append({"min_support": support, "merge_radius": round(float(radius), 1), **row(fusion)})  # fmt: skip
        sweep_df = pd.DataFrame(rows)
        typer.echo("\nПеребор параметров слияния:\n" + to_markdown(sweep_df))
        typer.echo(f"\nСохранено: {_rel(save_table(sweep_df, f'birdseye_sweep_{tag}'))}")


# ---------------------------------------------------------------- оценка и обучение


@app.command("eval-det")
def eval_det(
    models: Annotated[str, typer.Option(help="Через запятую")] = "yolo26n,yolo26s,rfdetr-nano",
    step: Annotated[int, typer.Option(help="Брать каждый N-й кадр")] = 1,
    imgsz: int = 640,
    max_frames: MaxFrames = None,
    name: Annotated[str, typer.Option(help="Имя файла с результатами")] = "detection_mot17",
) -> None:
    """Сравнить детекторы на MOT17 val: AP50, AP50-95, F1, скорость."""
    from box_interceptor.eval.detection import compare_detectors
    from box_interceptor.eval.report import save_table, to_markdown

    df = compare_detectors(_models(models), step=step, imgsz=imgsz, max_frames=max_frames)
    typer.echo(to_markdown(df))
    typer.echo(f"\nСохранено: {_rel(save_table(df, name))}")


@app.command("eval-track")
def eval_track(
    models: Annotated[str, typer.Option(help="Через запятую")] = "yolo26n",
    imgsz: int = 640,
    det_conf: Annotated[
        float, typer.Option(help="Порог детектора (низкий: трекеру нужны и слабые детекции)")
    ] = 0.05,  # fmt: skip
    high_conf: Annotated[
        str, typer.Option(help="Порог 'уверенной' детекции ByteTrack; через запятую — перебор")
    ] = "0.3",  # fmt: skip
    activation: Annotated[
        str, typer.Option(help="Порог рождения трека; через запятую — перебор")
    ] = "0.4",  # fmt: skip
    name: Annotated[str, typer.Option(help="Имя файла с результатами")] = "tracking_mot17",
) -> None:
    """Трекинг на MOT17 val: HOTA, MOTA, IDF1, переключения ID."""
    import pandas as pd

    from box_interceptor.eval.report import save_table, to_markdown
    from box_interceptor.eval.tracking import TrackingRun, evaluate_run
    from box_interceptor.tracking.tracker import TrackerParams

    runs = [
        TrackingRun(m, det_conf, imgsz, TrackerParams(high_conf=float(h), activation=float(a)))
        for m in _models(models)
        for h in high_conf.split(",")
        for a in activation.split(",")
    ]
    rows = []
    for run in runs:
        typer.echo(f"→ {run.label}")
        rows.append(evaluate_run(run))
    df = pd.DataFrame(rows)
    typer.echo(to_markdown(df))
    typer.echo(f"\nСохранено: {_rel(save_table(df, name))}")


@app.command()
def train(
    model: Annotated[str, typer.Option(help="Базовая модель")] = "yolo26n",
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    fraction: Annotated[
        float, typer.Option(help="Доля train-данных (0.25 — быстрый эксперимент)")
    ] = 1.0,  # fmt: skip
    name: Annotated[str | None, typer.Option(help="Имя результата в models/")] = None,
    freeze: Annotated[
        int | None, typer.Option(help="Заморозить первые N слоёв (10 = backbone)")
    ] = None,  # fmt: skip
    lr0: Annotated[float | None, typer.Option(help="Начальный learning rate (AdamW)")] = None,
    patience: Annotated[int, typer.Option(help="Early stopping: эпох без улучшения")] = 10,
) -> None:
    """Дообучить YOLO на MOT17 (сначала: bi prepare-yolo)."""
    from box_interceptor.training import train as run_train

    run_train(
        model, epochs=epochs, imgsz=imgsz, batch=batch, fraction=fraction, name=name,
        freeze=freeze, lr0=lr0, patience=patience,
    )  # fmt: skip


@app.command()
def export(
    weights: Annotated[str, typer.Argument(help="Путь к .pt или имя модели")] = "yolo26n",
    format: Annotated[str, typer.Option(help="coreml | onnx | ...")] = "coreml",  # noqa: A002
    imgsz: int = 640,
    half: Annotated[
        bool | None,
        typer.Option("--half/--no-half", help="float16 веса (по умолчанию: да для CoreML)"),
    ] = None,
) -> None:
    """Экспорт модели (по умолчанию CoreML для Neural Engine на Mac)."""
    from box_interceptor.training import export as run_export

    out = run_export(weights, fmt=format, imgsz=imgsz, half=half)
    typer.echo(f"Экспортировано: {out}")


@app.command()
def bench(
    models: Annotated[
        str, typer.Option(help="Через запятую; @cpu — принудительно CPU")
    ] = "yolo26n,yolo26n@cpu",  # fmt: skip
    n_frames: int = 150,
    name: Annotated[str, typer.Option(help="Имя файла с результатами")] = "speed",
) -> None:
    """Скорость моделей на Full HD кадрах (MOT17-09)."""
    from box_interceptor.eval.report import save_table, to_markdown
    from box_interceptor.eval.speed import run

    df = run(_models(models), n_frames=n_frames)
    typer.echo(to_markdown(df, ".1f"))
    typer.echo(f"\nСохранено: {_rel(save_table(df, name))}")


@app.command()
def web(
    host: Annotated[
        str, typer.Option(help="127.0.0.1 — доступно только с этого компьютера")
    ] = "127.0.0.1",  # fmt: skip
    port: int = 8000,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Открыть браузер")] = True,
) -> None:
    """Пульт камер в браузере: видео в реальном времени, ИИ обрабатывает на лету."""
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{host}:{port}"
    typer.echo(f"Пульт камер: {url}   (остановить: Ctrl+C)")
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run("box_interceptor.web.server:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
