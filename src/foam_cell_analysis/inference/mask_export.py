"""粒子解析用マスクの出力。Qt に依存しない（ワーカースレッドから呼ぶ想定）。

設計書: docs/design/comparison_inference_backend_design.html 12 章。
出力は ``<name>.partial`` に書き、export_info.json を最後に書いてから ``<name>`` に改名する。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import secrets
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from foam_cell_analysis.inference.particle_split import (
    PARTICLE_SPLIT_ID,
    binary_mask,
    split_report,
)

_CONTENTS = {"label", "binary"}
_FORMATS = {"png": ".png", "tiff": ".tif"}
_MARGIN_BYTES = 100 * 1024 * 1024
_PNG_MAX_LABEL = 65535
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_REPORT_COLUMNS = [
    "candidate_id",
    "evaluation_id",
    "item_id",
    "n_labels",
    "vanished",
    "split",
    "removed_pixels",
    "removed_fraction",
    "remaining_contacts",
]


class ExportError(Exception):
    """出力の失敗。途中の .partial フォルダは削除せず残す。"""

    def __init__(self, message: str, partial_folder: Path | None = None):
        super().__init__(message)
        self.partial_folder = partial_folder


@dataclass
class ExportSource:
    """1 候補の採用済み評価。"""

    candidate_id: str
    evaluation_id: str
    validation_version: str
    input_fingerprint: str
    # (item_id, 予測ファイルの絶対パス, 期待サイズ, 期待 sha256)
    predictions: list[tuple[str, Path, int, str]]


@dataclass
class ExportRequest:
    output_parent: Path
    sources: list[ExportSource]
    contents: set[str]
    file_format: str
    app_version: str


@dataclass
class ExportResult:
    folder: Path | None = None
    partial_folder: Path | None = None
    n_images: int = 0
    vanished_count: int = 0
    vanished_images: int = 0
    split_count: int = 0
    split_images: int = 0
    cancelled: bool = False


@dataclass
class ExportPlan:
    """事前チェックの結果。"""

    total_images: int
    required_bytes: int
    free_bytes: int
    # ソースごとのサブフォルダ名と、item_id -> 安全なファイル名
    subfolders: list[str] = field(default_factory=list)
    safe_names: list[dict[str, str]] = field(default_factory=list)


def sanitize_name(name: str) -> str:
    """Windows で使えない文字と末尾のドットや空白を「_」にする。"""
    safe = _INVALID.sub("_", name)
    stripped = safe.rstrip(". ")
    safe = stripped + "_" * (len(safe) - len(stripped))
    return safe or "_"


def _make_folder_name() -> str:
    return f"export_{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"


def _decode(data: bytes, path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        return np.asarray(tifffile.imread(io.BytesIO(data)))
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image)


def _read_verified(path: Path, size: int, sha256: str) -> np.ndarray:
    """サイズと sha256 を確認してからラベル画像を読む。"""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExportError(f"予測ファイルを読み込めません: {path}（{exc}）") from exc
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise ExportError(f"予測ファイルが記録時から変更されています: {path}")
    try:
        return _decode(data, path)
    except Exception as exc:
        raise ExportError(f"予測ファイルを画像として読めません: {path}（{exc}）") from exc


def plan_export(request: ExportRequest) -> ExportPlan:
    """書き込み前の検査。問題があれば日本語メッセージの ValueError を送出する。"""
    if not request.contents or not request.contents <= _CONTENTS:
        raise ValueError("出力内容にはラベル画像・二値マスクのいずれかを選んでください")
    if request.file_format not in _FORMATS:
        raise ValueError("出力形式は PNG または TIFF を選んでください")
    if not request.sources or not any(s.predictions for s in request.sources):
        raise ValueError("出力する画像がありません")
    parent = Path(request.output_parent)
    if not parent.is_dir():
        raise ValueError(f"出力先フォルダが存在しません: {parent}")

    plan = ExportPlan(total_images=0, required_bytes=_MARGIN_BYTES, free_bytes=0)
    label_bytes = 2 if request.file_format == "png" else 4
    per_pixel = (label_bytes if "label" in request.contents else 0) + (
        1 if "binary" in request.contents else 0
    )
    seen_folders: set[str] = set()
    for source in request.sources:
        folder = sanitize_name(f"{source.candidate_id}_{source.evaluation_id}")
        if folder.lower() in seen_folders:
            raise ValueError(f"候補と評価の組み合わせが重複しています: {folder}")
        seen_folders.add(folder.lower())
        names: dict[str, str] = {}
        used: dict[str, str] = {}
        for item_id, path, _size, _sha in source.predictions:
            safe = sanitize_name(item_id)
            if safe.lower() in used:
                raise ValueError(
                    f"画像名を安全な名前に直すと重複します: {used[safe.lower()]} と {item_id}"
                )
            used[safe.lower()] = item_id
            names[item_id] = safe
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception as exc:
                raise ValueError(f"予測ファイルを開けません: {path}（{exc}）") from exc
            plan.required_bytes += width * height * per_pixel
            plan.total_images += 1
            if request.file_format == "png" and "label" in request.contents:
                maximum = int(_decode(Path(path).read_bytes(), Path(path)).max(initial=0))
                if maximum > _PNG_MAX_LABEL:
                    raise ValueError(
                        f"ラベル番号が {_PNG_MAX_LABEL} を超える画像があります（{item_id}）。"
                        "形式に TIFF を選んでください"
                    )
        plan.subfolders.append(folder)
        plan.safe_names.append(names)

    plan.free_bytes = shutil.disk_usage(parent).free
    if plan.required_bytes > plan.free_bytes:
        need = plan.required_bytes / 1024 / 1024
        free = plan.free_bytes / 1024 / 1024
        raise ValueError(
            f"出力先の空き容量が足りません（必要 約{need:.0f} MB、空き 約{free:.0f} MB）"
        )
    return plan


def _write_file(path: Path, writer: Callable[[Path], None]) -> None:
    """一時名に書いてから os.replace する。既存ファイルは上書きしない。"""
    if path.exists():
        raise ExportError(f"同名のファイルが既にあります: {path}")
    tmp = path.with_name(f".{path.name}.tmp")
    writer(tmp)
    os.replace(tmp, path)


def _save_image(array: np.ndarray, file_format: str) -> Callable[[Path], None]:
    def writer(tmp: Path) -> None:
        if file_format == "png":
            Image.fromarray(array).save(tmp, format="PNG")
        else:
            tifffile.imwrite(tmp, array)

    return writer


def _label_array(labels: np.ndarray, file_format: str) -> np.ndarray:
    maximum = int(labels.max(initial=0))
    if maximum > _PNG_MAX_LABEL:
        if file_format == "png":
            raise ExportError("ラベル番号が 65535 を超えるため PNG では出力できません")
        return labels.astype(np.uint32)
    return labels.astype(np.uint16)


def run_export(
    request: ExportRequest,
    progress: Callable[[int, int], None],
    is_cancelled: Callable[[], bool],
) -> ExportResult:
    """マスクを出力する。失敗時は ExportError、中止時は cancelled=True の結果を返す。"""
    plan = plan_export(request)
    parent = Path(request.output_parent)
    name = _make_folder_name()
    while (parent / name).exists() or (parent / f"{name}.partial").exists():
        name = _make_folder_name()
    final = parent / name
    partial = parent / f"{name}.partial"
    partial.mkdir()

    result = ExportResult(partial_folder=partial)
    total = plan.total_images
    ext = _FORMATS[request.file_format]
    report_rows: list[dict] = []
    done = 0
    progress(0, total)
    try:
        for source, subfolder, names in zip(
            request.sources, plan.subfolders, plan.safe_names, strict=True
        ):
            target = partial / subfolder
            target.mkdir()
            for item_id, path, size, sha in source.predictions:
                if is_cancelled():
                    result.cancelled = True
                    return result
                labels = _read_verified(Path(path), size, sha)
                if labels.ndim != 2:
                    raise ExportError(f"ラベル画像が2次元ではありません: {path}")
                report = split_report(labels)
                if report["remaining_contacts"] > 0:
                    raise ExportError(f"粒子分離後も接触が残っています（{item_id}）")
                safe = names[item_id]
                if "label" in request.contents:
                    array = _label_array(labels, request.file_format)
                    _write_file(
                        target / f"{safe}_label{ext}", _save_image(array, request.file_format)
                    )
                if "binary" in request.contents:
                    _write_file(
                        target / f"{safe}_binary{ext}",
                        _save_image(binary_mask(labels), request.file_format),
                    )
                n_vanished = len(report["vanished_labels"])
                n_split = len(report["split_labels"])
                result.vanished_count += n_vanished
                result.vanished_images += n_vanished > 0
                result.split_count += n_split
                result.split_images += n_split > 0
                report_rows.append(
                    {
                        "candidate_id": source.candidate_id,
                        "evaluation_id": source.evaluation_id,
                        "item_id": item_id,
                        "n_labels": int(np.unique(labels[labels > 0]).size),
                        "vanished": n_vanished,
                        "split": n_split,
                        "removed_pixels": report["removed_pixels"],
                        "removed_fraction": report["removed_fraction"],
                        "remaining_contacts": report["remaining_contacts"],
                    }
                )
                done += 1
                progress(done, total)

        def write_report(tmp: Path) -> None:
            with open(tmp, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=_REPORT_COLUMNS)
                writer.writeheader()
                writer.writerows(report_rows)

        _write_file(partial / "export_report.csv", write_report)
        info = {
            "schema": 1,
            "export_id": name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "app_version": request.app_version,
            "particle_split_id": PARTICLE_SPLIT_ID,
            "format": request.file_format,
            "contents": sorted(request.contents),
            "sources": [
                {
                    "candidate_id": s.candidate_id,
                    "evaluation_id": s.evaluation_id,
                    "validation_version": s.validation_version,
                    "input_fingerprint": s.input_fingerprint,
                    "item_ids": [p[0] for p in s.predictions],
                }
                for s in request.sources
            ],
        }

        def write_info(tmp: Path) -> None:
            tmp.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

        _write_file(partial / "export_info.json", write_info)
        if final.exists():
            raise ExportError(f"出力フォルダが既に存在します: {final}")
        os.rename(partial, final)
    except ExportError as exc:
        exc.partial_folder = partial
        raise
    except Exception as exc:
        raise ExportError(f"出力中にエラーが発生しました: {exc}", partial) from exc

    result.folder = final
    result.partial_folder = None
    result.n_images = done
    return result
