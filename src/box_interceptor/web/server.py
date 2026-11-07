"""Веб-панель: страница, управление и живой поток кадров.

    браузер ──HTTP──> /api/start, /api/overlays     (управление: какую сцену, какую модель)
    браузер <──WebSocket── /ws                       (поток: цифры JSON + кадры JPEG)

Почему WebSocket, а не просто <img src="поток.mjpg">: браузер держит не больше
6 HTTP-соединений с одним сайтом. 4 камеры + план пола = 5 вечных соединений,
и кнопки панели встали бы в очередь. WebSocket — одно соединение, по которому
сервер сам присылает всё новое, как только ИИ обработал кадр.

Формат бинарного сообщения с кадром: [1 байт: длина ключа][ключ "cam0"/"top"][JPEG].
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from box_interceptor import paths
from box_interceptor.config import CameraConfig, SceneConfig, load_scene
from box_interceptor.web.engine import Engine

STATIC = Path(__file__).parent / "static"
engine = Engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    engine.stop()  # при остановке сервера (Ctrl+C) аккуратно гасим рабочий поток


app = FastAPI(title="box-interceptor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------------------------------------------------------------- что можно выбрать


def list_scenes() -> list[dict]:
    scenes = []
    for path in sorted(paths.CONFIGS.glob("*.yaml")):
        s = load_scene(path)
        scenes.append({"name": s.name, "description": s.description,
                       "cameras": len(s.cameras), "floor": s.birdseye is not None})  # fmt: skip
    return scenes


def list_models() -> list[dict]:
    """Только модели, веса которых уже скачаны — чтобы выбор не начинался с часовой загрузки."""
    m = paths.MODELS
    models = []
    if (m / "yolo26n.mlpackage").exists():
        models.append({"id": str(m / "yolo26n.mlpackage"), "label": "YOLO26n на Neural Engine (CoreML), быстрее всех"})  # fmt: skip
    models.append({"id": "yolo26n", "label": "YOLO26n на GPU (PyTorch)"})
    if (m / "yolo26s.pt").exists():
        models.append({"id": "yolo26s", "label": "YOLO26s на GPU, точнее и медленнее"})
    rf_home = os.environ.get("RF_HOME") or os.environ.get("ROBOFLOW_HOME") or "~/.roboflow/models"
    if (Path(rf_home).expanduser() / "rf-detr-nano.pth").exists():
        models.append({"id": "rfdetr-nano", "label": "RF-DETR nano, трансформер"})
    for p in sorted(m.glob("*-mot17*.pt")):
        models.append({"id": str(p), "label": f"{p.stem}, дообучена на MOT17"})
    return models


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/options")
def options() -> dict:
    return {"scenes": list_scenes(), "models": list_models(), "overlays": engine.overlays}


# ---------------------------------------------------------------- управление


class StartRequest(BaseModel):
    model: str
    scene: str | None = None  # сцена из configs/
    source: str | None = None  # или свой источник: путь к видео, "0" (камера Mac), rtsp://...


@app.post("/api/start")
def start(req: StartRequest) -> dict:
    if req.source:
        scene = SceneConfig(
            name="custom",
            description=req.source,
            cameras=[CameraConfig(id=0, video=req.source.strip())],
        )
    elif req.scene:
        try:
            scene = load_scene(req.scene)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
    else:
        raise HTTPException(422, "Укажите сцену или свой источник видео")
    engine.start(scene, req.model)
    return {"ok": True}


class OverlayRequest(BaseModel):
    heatmap: bool | None = None
    trace: bool | None = None


@app.post("/api/overlays")
def overlays(req: OverlayRequest) -> dict:
    return engine.set_overlays(heatmap=req.heatmap, trace=req.trace)


# ---------------------------------------------------------------- живой поток


def pack_frame(key: str, jpeg: bytes) -> bytes:
    k = key.encode()
    return bytes([len(k)]) + k + jpeg


@app.websocket("/ws")
async def feed(ws: WebSocket) -> None:
    await ws.accept()
    last = -1
    try:
        while True:
            seq, frames, stats = engine.snapshot()
            if seq == last:
                await asyncio.sleep(0.01)
                continue
            last = seq
            await ws.send_json(stats)
            for key, jpeg in frames.items():
                await ws.send_bytes(pack_frame(key, jpeg))
    except WebSocketDisconnect:
        pass
