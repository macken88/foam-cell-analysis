"""確定済み学習データセットの遅延読み込みストア。"""

from __future__ import annotations

import csv
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from foam_cell_analysis.services.models import DataItem

logger = logging.getLogger(__name__)
MAX_ARRAY_CACHE_BYTES = 2 * 1024**3


class DatasetStore:
    """workspace/datasets 以下の RELEASED 版を読み込む。"""

    def __init__(self, workspace_root: str | Path, cache_bytes: int = 2 * 1024**3) -> None:
        self.workspace_root = Path(workspace_root)
        self.datasets_root = self.workspace_root / "datasets"
        self.cache_bytes = min(MAX_ARRAY_CACHE_BYTES, max(0, int(cache_bytes)))
        self._cache = ArrayLRU(self.cache_bytes)

    @property
    def _cache_size(self) -> int:
        """画像と外部キャッシュを合わせた使用量を返す。"""
        return self._cache.size

    def list_versions(self) -> list[str]:
        """読み込み可能な学習用 RELEASED 版を版名順に返す。"""
        versions = []
        if not self.datasets_root.exists():
            return versions
        for folder in sorted(self.datasets_root.iterdir(), key=lambda path: path.name):
            if not folder.is_dir():
                continue
            try:
                info = json.loads((folder / "dataset_info.json").read_text(encoding="utf-8"))
                if info.get("purpose") == "train" and info.get("status") == "RELEASED":
                    self._read_items(folder)
                    versions.append(str(info.get("dataset_version", folder.name)))
            except (OSError, ValueError, KeyError, csv.Error) as error:
                logger.warning("学習データセットを読み飛ばしました (%s): %s", folder, error)
        return versions

    def get_items(self, version: str) -> list[DataItem]:
        """メタデータと manifest を item_id で結合して返す。"""
        folder = self.datasets_root / version
        info = json.loads((folder / "dataset_info.json").read_text(encoding="utf-8"))
        if info.get("purpose") != "train" or info.get("status") != "RELEASED":
            raise ValueError(f"学習用 RELEASED データセットではありません: {version}")
        return self._read_items(folder)

    @staticmethod
    def _read_csv(path: Path) -> dict[str, dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            if not rows.fieldnames or "item_id" not in rows.fieldnames:
                raise ValueError(f"item_id 列がありません: {path}")
            return {row["item_id"]: row for row in rows if row.get("item_id")}

    def _read_items(self, folder: Path) -> list[DataItem]:
        metadata = self._read_csv(folder / "metadata.csv")
        manifest = self._read_csv(folder / "manifest.csv")
        if metadata.keys() != manifest.keys():
            raise ValueError(f"metadata.csv と manifest.csv の項目が一致しません: {folder}")
        items = []
        for item_id in sorted(metadata.keys() & manifest.keys()):
            meta, files = metadata[item_id], manifest[item_id]
            relpath = meta.get("source_relpath", "")
            revision = meta.get("mask_revision") or files.get("mask_revision", "rev_001")
            items.append(
                DataItem(
                    item_id=item_id,
                    source_filename=meta.get("source_filename", Path(relpath).name),
                    source_relpath=relpath,
                    channels=[meta.get("channel", "")],
                    classification=meta.get("classification") or None,
                    quality=meta.get("quality") or None,
                    mask_revisions=[revision],
                    selected_mask_revision=revision,
                    usage=meta.get("usage", "train"),
                    sha256=files.get("image_sha256", ""),
                    seed=int(meta.get("seed") or 0),
                )
            )
        return items

    def _paths(self, version: str, item_id: str) -> tuple[Path, Path]:
        folder = self.datasets_root / version
        manifest = self._read_csv(folder / "manifest.csv")
        row = manifest[item_id]
        return folder / row["image_path"], folder / row["mask_path"]

    def _read_array(self, path: Path) -> np.ndarray:
        key = ("dataset", path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with Image.open(path) as image:
            array = np.asarray(image).copy()
        if array.ndim != 2 or array.dtype not in (np.uint8, np.uint16, np.int32, np.uint32):
            raise ValueError(f"画像は 2 次元 uint8/uint16 整数である必要があります: {path}")
        self._cache.put(key, array)
        return array

    def get_image(self, version: str, item_id: str, channel: str | None = None) -> np.ndarray:
        """画像を必要時に読み込む。現行形式のチャンネルは 1 つ。"""
        item = next((item for item in self.get_items(version) if item.item_id == item_id), None)
        if item is None:
            raise KeyError(item_id)
        if channel is not None and channel not in item.channels:
            raise ValueError(f"指定チャンネルがありません: {channel}")
        array = self._read_array(self._paths(version, item_id)[0])
        if array.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"画像は uint8 または uint16 である必要があります: {item_id}")
        return array

    def get_mask(self, version: str, item_id: str, revision: str | None = None) -> np.ndarray:
        """整数ラベルのマスクを必要時に読み込む。"""
        item = next((item for item in self.get_items(version) if item.item_id == item_id), None)
        if item is None:
            raise KeyError(item_id)
        if revision is not None and revision != item.selected_mask_revision:
            raise ValueError(f"指定マスク版がありません: {revision}")
        image, mask = self._paths(version, item_id)
        image_array, mask_array = self._read_array(image), self._read_array(mask)
        if not np.issubdtype(mask_array.dtype, np.integer):
            raise ValueError(f"マスクは整数ラベルである必要があります: {item_id}")
        if image_array.shape != mask_array.shape:
            raise ValueError(f"画像とマスクのサイズが一致しません: {item_id}")
        return mask_array

    def select_training_items(self, config: dict[str, Any]) -> list[DataItem]:
        """設計書 3 章の順に条件を適用し、item_id 昇順で返す。"""
        data = config.get("data", {})
        version = data["dataset_version"]
        channels = data.get("input_channels", [])
        if len(channels) != 1:
            raise ValueError("現行データ形式では input_channels は 1 つ必要です")
        selected = [item for item in self.get_items(version) if item.usage == "train"]
        classification = data.get("classification", "all")
        if classification != "all":
            selected = [item for item in selected if item.classification == classification]
        quality = data.get("quality_filter", "all")
        if quality == "good_only":
            selected = [item for item in selected if item.quality == "良"]
        elif quality == "good_and_acceptable":
            selected = [item for item in selected if item.quality in {"良", "可"}]
        selected = [item for item in selected if item.channels == channels]
        return sorted(selected, key=lambda item: item.item_id)


class ArrayLRU:
    """配列種別をまたいで容量を共有する重み付き LRU キャッシュ。"""

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = max(0, int(capacity_bytes))
        self._items: OrderedDict[Any, np.ndarray] = OrderedDict()
        self.size = 0

    def get(self, key: Any) -> np.ndarray | None:
        """配列を返し、使用順を最新にする。"""
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: Any, value: np.ndarray) -> None:
        """配列を追加し、共有容量を超えた古い項目を追い出す。"""
        value_size = int(value.nbytes)
        if value_size > self.capacity_bytes:
            return
        existing = self._items.pop(key, None)
        if existing is not None:
            self.size -= int(existing.nbytes)
        while self._items and self.size + value_size > self.capacity_bytes:
            _, removed = self._items.popitem(last=False)
            self.size -= int(removed.nbytes)
        self._items[key] = value
        self.size += value_size
