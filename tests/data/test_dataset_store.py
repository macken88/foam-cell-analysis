import csv
import json

import numpy as np
import pytest
from PIL import Image

from foam_cell_analysis.data.dataset_store import DatasetStore


def _write_dataset(root):
    folder = root / "datasets" / "train_v000"
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir()
    (folder / "dataset_info.json").write_text(
        json.dumps({"purpose": "train", "status": "RELEASED", "dataset_version": "train_v000"}),
        encoding="utf-8",
    )
    with (folder / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "item_id",
                "source_relpath",
                "channel",
                "usage",
                "classification",
                "quality",
                "mask_revision",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "item_id": "z",
                    "source_relpath": "g2/z.png",
                    "channel": "A",
                    "usage": "train",
                    "classification": "B",
                    "quality": "良",
                    "mask_revision": "r1",
                },
                {
                    "item_id": "a",
                    "source_relpath": "g1/a.png",
                    "channel": "B",
                    "usage": "train",
                    "classification": "A",
                    "quality": "不良",
                    "mask_revision": "r1",
                },
            ]
        )
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "image_path", "mask_path"])
        writer.writeheader()
        for item_id in ("z", "a"):
            image_path, mask_path = f"images/{item_id}.png", f"masks/{item_id}.png"
            Image.fromarray(np.full((4, 5), 200, dtype=np.uint8)).save(folder / image_path)
            Image.fromarray(np.zeros((4, 5), dtype=np.uint16)).save(folder / mask_path)
            writer.writerow({"item_id": item_id, "image_path": image_path, "mask_path": mask_path})


def test_dataset_enumeration_join_sort_filter_and_lazy_arrays(tmp_path):
    _write_dataset(tmp_path)
    store = DatasetStore(tmp_path, cache_bytes=50)
    assert store.list_versions() == ["train_v000"]
    items = store.get_items("train_v000")
    assert [item.item_id for item in items] == ["a", "z"]
    config = {
        "data": {"dataset_version": "train_v000", "input_channels": ["A"], "quality_filter": "all"}
    }
    assert [item.item_id for item in store.select_training_items(config)] == ["z"]
    assert store.get_image("train_v000", "z", "A").dtype == np.uint8
    assert store.get_mask("train_v000", "z", "r1").dtype == np.uint16


def test_dataset_channel_mismatch_rejected(tmp_path):
    _write_dataset(tmp_path)
    config = {"data": {"dataset_version": "train_v000", "input_channels": ["A", "B"]}}
    with pytest.raises(ValueError, match="1 つ"):
        DatasetStore(tmp_path).select_training_items(config)


def test_train_v000_like_metadata_counts_and_groups(tmp_path):
    folder = tmp_path / "datasets" / "train_v000"
    folder.mkdir(parents=True)
    (folder / "dataset_info.json").write_text(
        json.dumps({"purpose": "train", "status": "RELEASED", "dataset_version": "train_v000"}),
        encoding="utf-8",
    )
    columns = [
        "item_id",
        "source_relpath",
        "channel",
        "usage",
        "classification",
        "quality",
        "mask_revision",
    ]
    with (folder / "metadata.csv").open("w", newline="", encoding="utf-8") as metadata_stream:
        writer = csv.DictWriter(metadata_stream, fieldnames=columns)
        writer.writeheader()
        for index in range(12):
            writer.writerow(
                {
                    "item_id": f"i{index:02d}",
                    "source_relpath": f"group_{index // 2}/i{index:02d}.png",
                    "channel": "A",
                    "usage": "train",
                    "classification": "分類A" if index < 6 else "分類B",
                    "quality": "不良" if index in {0, 1} else "良",
                    "mask_revision": "r1",
                }
            )
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest_stream:
        writer = csv.DictWriter(manifest_stream, fieldnames=["item_id", "image_path", "mask_path"])
        writer.writeheader()
        for index in range(12):
            writer.writerow(
                {
                    "item_id": f"i{index:02d}",
                    "image_path": f"images/i{index:02d}.png",
                    "mask_path": f"masks/i{index:02d}.png",
                }
            )
    store = DatasetStore(tmp_path)
    items = store.get_items("train_v000")
    assert len(items) == 12
    assert len({item.source_folder for item in items}) == 6
    config = {
        "data": {
            "dataset_version": "train_v000",
            "input_channels": ["A"],
            "quality_filter": "good_only",
        }
    }
    assert len(store.select_training_items(config)) == 10


def _clone_as(root, source, target, purpose, usage):
    import shutil

    src, dst = root / "datasets" / source, root / "datasets" / target
    shutil.copytree(src, dst)
    info = json.loads((dst / "dataset_info.json").read_text(encoding="utf-8"))
    info.update(purpose=purpose, dataset_version=target)
    (dst / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    text = (dst / "metadata.csv").read_text(encoding="utf-8").replace(",train,", f",{usage},")
    (dst / "metadata.csv").write_text(text, encoding="utf-8")
    return dst


def test_val_version_listing_purpose_and_evaluation_items(tmp_path):
    _write_dataset(tmp_path)
    _clone_as(tmp_path, "train_v000", "val_v000", "val", "val")
    store = DatasetStore(tmp_path)
    assert store.list_versions() == ["train_v000"]
    assert store.list_versions("val") == ["val_v000"]
    assert [item.item_id for item in store.select_evaluation_items("val_v000")] == ["a", "z"]
    assert store.get_items("val_v000", expected_purpose="val")[0].usage == "val"
    assert store.get_image("val_v000", "z").dtype == np.uint8
    assert store.get_mask("val_v000", "z").shape == (4, 5)
    with pytest.raises(ValueError):
        store.get_items("val_v000", expected_purpose="train")
    with pytest.raises(ValueError):
        store.get_items("train_v000", expected_purpose="val")
    with pytest.raises(ValueError):
        store.select_evaluation_items("train_v000")


def test_duplicate_item_id_or_name_mismatch_version_is_excluded(tmp_path):
    _write_dataset(tmp_path)
    dup = _clone_as(tmp_path, "train_v000", "val_v001", "val", "val")
    with (dup / "manifest.csv").open("a", newline="", encoding="utf-8") as stream:
        stream.write("z,images/z.png,masks/z.png\n")
    _clone_as(tmp_path, "train_v000", "val_v002", "val", "val")
    info_path = tmp_path / "datasets" / "val_v002" / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["dataset_version"] = "val_v999"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    assert DatasetStore(tmp_path).list_versions("val") == []
