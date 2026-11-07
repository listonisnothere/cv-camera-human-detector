from pathlib import Path

import pytest
from pydantic import ValidationError

from box_interceptor import paths
from box_interceptor.config import CameraConfig, SceneConfig, load_scene

SCENES = sorted(p.stem for p in paths.CONFIGS.glob("*.yaml"))


def test_scenes_exist():
    assert {"lab6p", "terrace1", "passageway1"} <= set(SCENES)


@pytest.mark.parametrize("name", SCENES)
def test_scene_loads(name):
    scene = load_scene(name)
    assert scene.name == name
    assert len(scene.cameras) == 4
    # пути разрешены от корня проекта
    assert all(Path(cam.video).is_absolute() for cam in scene.cameras)
    if scene.birdseye is not None:
        # вид сверху без калибровки и GT невозможен
        assert scene.calibration is not None and scene.ground_truth is not None


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("0", "0"),  # веб-камера
        ("rtsp://user:pw@10.0.0.5:554/stream", "rtsp://user:pw@10.0.0.5:554/stream"),  # "//" цел
        ("/abs/video.mp4", "/abs/video.mp4"),
    ],
)
def test_camera_source_kept_as_is(given, expected):
    assert CameraConfig(id=0, video=given).video == expected


def test_camera_relative_path_resolved_from_root():
    assert CameraConfig(id=0, video="data/x.mp4").video == str(paths.ROOT / "data/x.mp4")


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError, match="merge_raduis"):
        SceneConfig.model_validate(
            {
                "name": "x",
                "cameras": [{"id": 0, "video": "v.mp4"}],
                "birdseye": {"merge_raduis": 1, "match_radius": 1},  # опечатка
            }
        )


def test_zone_needs_three_points():
    with pytest.raises(ValidationError):
        SceneConfig.model_validate(
            {
                "name": "x",
                "cameras": [
                    {
                        "id": 0,
                        "video": "v.mp4",
                        "zones": [{"name": "z", "polygon": [[0, 0], [1, 1]]}],
                    }
                ],
            }
        )


def test_missing_scene_lists_available():
    with pytest.raises(FileNotFoundError, match="lab6p"):
        load_scene("no-such-scene")
