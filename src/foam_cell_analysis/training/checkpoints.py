"""チェックポイントと検証ラベルの保存・整理。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def save_state(adapter: Any, path: str | Path) -> Path:
    """state_dict を同じディレクトリの一時ファイル経由で保存する。"""
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(adapter.state_dict(), temporary)
    os.replace(temporary, target)
    return target


def write_checkpoint(
    adapter: Any,
    path: str | Path,
    *,
    model_type: str,
    fold: int | None,
    epoch: int,
    kind: str,
    run_id: str,
) -> Path:
    """重みと 4.2 の sidecar を原子的に保存する。"""
    target = save_state(adapter, path)
    metadata = {
        "model_type": model_type,
        "fold": fold,
        "epoch": epoch,
        "kind": kind,
        "run_id": run_id,
        "sha256": _file_sha256(target),
    }
    sidecar = target.with_suffix(target.suffix + ".json")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temporary, sidecar)
    return target


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checkpoint_metadata(
    path: Path,
    *,
    model_type: str,
    fold: int | None,
    epoch: int,
    kind: str,
    run_id: str,
) -> None:
    """既に移動・リンクしたチェックポイント用 sidecar を作る。"""
    metadata = {
        "model_type": model_type,
        "fold": fold,
        "epoch": epoch,
        "kind": kind,
        "run_id": run_id,
        "sha256": _file_sha256(path),
    }
    sidecar = path.with_suffix(path.suffix + ".json")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temporary, sidecar)


def create_selected(source: Path, target: Path) -> str:
    """temporary は移動、永続 checkpoint は hardlink、失敗時はコピーする。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.parts and "tmp_ckpt" in source.parts:
        os.replace(source, target)
        return "moved"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def write_label(path_without_suffix: Path, labels: np.ndarray) -> Path:
    """ラベル数に応じて uint16 PNG または uint32 TIFF を書く。"""
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("予測ラベルは 2 次元整数画像である必要があります")
    if int(values.max(initial=0)) > 65535:
        import tifffile

        target = path_without_suffix.with_suffix(".tif")
        target.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(target, values.astype(np.uint32, copy=False))
    else:
        target = path_without_suffix.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(values.astype(np.uint16, copy=False)).save(target)
    return target


def artifact_record(
    path: Path, run_dir: Path, hash_cache: dict[tuple[int, int, int, int], str] | None = None
) -> dict[str, Any]:
    """run_dir 相対のパス、サイズ、SHA-256 を記録する。"""
    stat = path.stat()
    cache_key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    sha256 = hash_cache.get(cache_key) if hash_cache is not None else None
    if sha256 is None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        sha256 = digest.hexdigest()
        if hash_cache is not None:
            hash_cache[cache_key] = sha256
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "size": stat.st_size,
        "sha256": sha256,
    }
