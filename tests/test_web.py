import time

import numpy as np
import pytest
import supervision as sv
from fastapi.testclient import TestClient

from box_interceptor.config import CameraConfig, SceneConfig
from box_interceptor.video.io import FfmpegWriter
from box_interceptor.web import engine as engine_mod
from box_interceptor.web.server import app, pack_frame


class FakeDetector:
    """Вместо нейросети: всегда "видит" одного человека в центре кадра. Тесты без GPU."""

    name = "fake"

    def __call__(self, frame):
        h, w = frame.shape[:2]
        return sv.Detections(
            xyxy=np.array([[w * 0.4, h * 0.3, w * 0.6, h * 0.9]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )


@pytest.fixture
def fake_detector(monkeypatch):
    monkeypatch.setattr(engine_mod, "build_detector", lambda model, conf: FakeDetector())


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.mp4"
    with FfmpegWriter(path, fps=25, size=(64, 48)) as w:
        for _ in range(30):
            w.write(np.full((48, 64, 3), 100, np.uint8))
    return path


def wait_for(engine, status, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, frames, stats = engine.snapshot()
        if stats.get("status") == status and (status != "running" or frames):
            return frames, stats
        time.sleep(0.05)
    raise AssertionError(f"Не дождались статуса {status}: {engine.snapshot()[2]}")


def test_pack_frame():
    packed = pack_frame("cam0", b"\xff\xd8jpeg")
    assert packed[0] == 4
    assert packed[1:5] == b"cam0"
    assert packed[5:] == b"\xff\xd8jpeg"


def test_engine_streams_frames_and_stats(video, fake_detector):
    zone = {"name": "z", "polygon": [[0, 0], [64, 0], [64, 48], [0, 48]]}
    scene = SceneConfig(name="t", cameras=[CameraConfig(id=0, video=str(video), zones=[zone])])
    engine = engine_mod.Engine()
    engine.start(scene, "fake")
    try:
        frames, stats = wait_for(engine, "running")
        assert frames["cam0"][:2] == b"\xff\xd8"  # это JPEG
        cam = stats["cameras"][0]
        assert cam["key"] == "cam0"
        assert "z" in cam["zones"]
        assert stats["floor"] is None  # у сцены нет калибровки -> нет плана пола
        assert stats["video_fps"] == pytest.approx(25, abs=1)
    finally:
        engine.stop()


def test_engine_reports_bad_source(fake_detector):
    scene = SceneConfig(name="t", cameras=[CameraConfig(id=0, video="/no/such/video.mp4")])
    engine = engine_mod.Engine()
    engine.start(scene, "fake")
    try:
        _, stats = wait_for(engine, "error")
        assert "Проверьте путь" in stats["message"]
    finally:
        engine.stop()


def test_overlays_toggle():
    engine = engine_mod.Engine()
    assert engine.set_overlays(heatmap=True) == {"heatmap": True, "trace": True}
    assert engine.set_overlays(trace=False) == {"heatmap": True, "trace": False}


def test_api_options_and_validation():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Пульт камер" in page.text

    opts = client.get("/api/options").json()
    assert {"lab6p", "terrace1", "passageway1"} <= {s["name"] for s in opts["scenes"]}
    assert any(m["id"] == "yolo26n" for m in opts["models"])

    assert client.post("/api/start", json={"model": "yolo26n"}).status_code == 422
    assert client.post("/api/start", json={"model": "yolo26n", "scene": "nope"}).status_code == 404
