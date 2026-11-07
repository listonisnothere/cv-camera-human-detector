import numpy as np

from box_interceptor.data import mot17

SEQINFO = """[Sequence]
name=MOT17-99-FRCNN
imDir=img1
frameRate=30
seqLength=3
imWidth=100
imHeight=50
imExt=.jpg
"""

# frame,id,x,y,w,h,flag,class,visibility
GT = """1,1,10,10,20,30,1,1,1.0
1,2,50,5,10,20,0,7,0.8
1,3,0,0,5,5,0,8,1.0
1,4,30,10,10,10,1,1,0.05
2,1,12,10,20,30,1,1,1.0
"""


def make_seq(root, name="MOT17-99"):
    d = root / f"{name}-FRCNN"
    (d / "gt").mkdir(parents=True)
    (d / "img1").mkdir()
    (d / "seqinfo.ini").write_text(SEQINFO.replace("MOT17-99", name))
    (d / "gt" / "gt.txt").write_text(GT)
    for f in range(1, 4):
        (d / "img1" / f"{f:06d}.jpg").write_bytes(b"fake")
    return d


def test_read_seqinfo(tmp_path):
    info = mot17.read_seqinfo(make_seq(tmp_path))
    assert (info.name, info.length, info.width, info.height) == ("MOT17-99", 3, 100, 50)
    assert info.frame_path(2).name == "000002.jpg"


def test_load_gt_positive_and_ignore(tmp_path):
    gt = mot17.load_gt(make_seq(tmp_path))
    # кадр 1: пешеход (класс 1) и стоящий человек (класс 7) — позитивы
    np.testing.assert_allclose(gt[1].boxes, [[10, 10, 30, 40], [50, 5, 60, 25]])
    # манекен (класс 8) и почти невидимый человек (vis=0.05) — зоны игнорирования
    assert len(gt[1].ignore) == 2
    assert len(gt[2].boxes) == 1
    assert len(gt[3].boxes) == 0  # кадр без людей тоже присутствует


def test_to_yolo_lines_normalizes_and_clips():
    lines = mot17.to_yolo_lines(
        np.array([[10, 10, 30, 40], [90, 0, 120, 50], [-50, 0, -10, 10]]), 100, 50
    )
    assert lines[0] == "0 0.200000 0.500000 0.200000 0.600000"
    # вторая рамка вылезает вправо -> обрезается по краю кадра (x 90..100)
    assert lines[1] == "0 0.950000 0.500000 0.100000 1.000000"
    assert len(lines) == 2  # третья целиком за кадром — выброшена


def test_build_yolo_dataset(tmp_path):
    root = tmp_path / "train"
    make_seq(root, "MOT17-01")
    make_seq(root, "MOT17-02")
    out = tmp_path / "yolo"
    data_yaml = mot17.build_yolo_dataset(
        out, train_step=1, val_step=2, root=root,
        train_sequences=["MOT17-01"], val_sequences=["MOT17-02"],
    )  # fmt: skip
    assert data_yaml.exists()
    assert len(list((out / "images" / "train").iterdir())) == 3
    assert len(list((out / "images" / "val").iterdir())) == 2  # кадры 1 и 3
    label = (out / "labels" / "train" / "MOT17-01_000001.txt").read_text().splitlines()
    assert len(label) == 2
    # картинка — симлинк на исходный кадр, а не копия
    assert (out / "images" / "train" / "MOT17-01_000001.jpg").is_symlink()
