import hashlib
import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from foam_cell_analysis.inference import mask_export
from foam_cell_analysis.inference.mask_export import (
    ExportError,
    ExportRequest,
    ExportSource,
    plan_export,
    run_export,
)


def _save(path, labels):
    Image.fromarray(labels.astype(np.uint16)).save(path)
    data = path.read_bytes()
    return path, len(data), hashlib.sha256(data).hexdigest()


def _touching():
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[2:6, 1:4] = 1
    labels[2:6, 4:7] = 2
    return labels


def _source(tmp_path, cand, items):
    preds = []
    for n, (item_id, labels) in enumerate(items):
        path, size, sha = _save(tmp_path / f"{cand}_{n}.png", labels)
        preds.append((item_id, path, size, sha))
    return ExportSource(cand, "ev1", "v1", "fp", preds)


def _request(tmp_path, sources, contents=None, fmt="png"):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return ExportRequest(out, sources, contents or {"label", "binary"}, fmt, "0.1")


def _run(request, cancel=lambda: False):
    return run_export(request, lambda d, t: None, cancel)


def test_happy_path(tmp_path):
    a = _source(tmp_path, "A", [("img1", _touching()), ("img2", np.zeros((8, 8)))])
    b = _source(tmp_path, "B", [("img1", _touching())])
    result = _run(_request(tmp_path, [a, b]))
    assert result.folder and result.folder.is_dir()
    assert not result.folder.name.endswith(".partial")
    assert result.n_images == 3
    assert result.vanished_count == 0 and result.split_images == 0
    label = np.asarray(Image.open(result.folder / "A_ev1" / "img1_label.png"))
    assert label.dtype == np.uint16 and np.array_equal(label, _touching())
    binary = np.asarray(Image.open(result.folder / "B_ev1" / "img1_binary.png"))
    assert (
        binary.dtype == np.uint8 and binary[3, 2] == 255 and binary[3, 3] == 0 and binary[3, 4] == 0
    )
    rows = (result.folder / "export_report.csv").read_bytes()
    assert rows.startswith(b"\xef\xbb\xbf") and rows.count(b"\n") == 4
    info = json.loads((result.folder / "export_info.json").read_text(encoding="utf-8"))
    assert info["schema"] == 1 and info["sources"][0]["item_ids"] == ["img1", "img2"]


def test_png_limit_error_and_tiff_uint32(tmp_path):
    big = np.zeros((4, 4), dtype=np.uint32)
    big[1, 1] = 70000
    path = tmp_path / "big.tif"
    tifffile.imwrite(path, big)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    src = ExportSource("A", "e", "v", "f", [("x", path, len(data), sha)])
    with pytest.raises(ValueError, match="TIFF"):
        plan_export(_request(tmp_path, [src]))
    result = _run(_request(tmp_path, [src], fmt="tiff"))
    assert tifffile.imread(result.folder / "A_e" / "x_label.tif").dtype == np.uint32


def test_sanitize_collision(tmp_path):
    src = _source(tmp_path, "A", [("a:b", _touching()), ("a?b", _touching())])
    with pytest.raises(ValueError, match="重複"):
        plan_export(_request(tmp_path, [src]))
    assert list((tmp_path / "out").iterdir()) == []


def test_cancel_keeps_partial(tmp_path):
    src = _source(tmp_path, "A", [("i", _touching())])
    result = _run(_request(tmp_path, [src]), cancel=lambda: True)
    assert result.cancelled and result.folder is None and result.partial_folder.is_dir()
    names = [p.name for p in (tmp_path / "out").iterdir()]
    assert len(names) == 1 and names[0].endswith(".partial")


def test_existing_folder_not_overwritten(tmp_path, monkeypatch):
    src = _source(tmp_path, "A", [("i", _touching())])
    request = _request(tmp_path, [src])
    existing = request.output_parent / "export_X_aaaa"
    existing.mkdir()
    (existing / "keep.txt").write_text("x")
    names = iter(["export_X_aaaa", "export_X_bbbb"])
    monkeypatch.setattr(mask_export, "_make_folder_name", lambda: next(names))
    result = _run(request)
    assert result.folder.name == "export_X_bbbb"
    assert (existing / "keep.txt").read_text() == "x"


def test_sha_mismatch_fails(tmp_path):
    src = _source(tmp_path, "A", [("i", _touching())])
    item, path, size, _ = src.predictions[0]
    src.predictions[0] = (item, path, size, "0" * 64)
    with pytest.raises(ExportError) as info:
        _run(_request(tmp_path, [src]))
    assert info.value.partial_folder.is_dir()
    names = [p.name for p in (tmp_path / "out").iterdir()]
    assert names and all(n.endswith(".partial") for n in names)
