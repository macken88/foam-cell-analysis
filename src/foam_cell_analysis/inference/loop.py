"""評価プロセスの本体（比較・推論設計 3.3・3.4・7.3・9 章）。

事前検査 → resolved_data.json → preflight → 画像ごとの推論・予測保存・AP のマッチング →
per_image.csv・eval_counts.npz・instances.csv → result.json の順に進める。
torch は推論用アダプタを作るときだけ読み込む（fake_numpy では読み込まない）。
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from foam_cell_analysis.inference.protocol import (
    EVAL_COUNTS,
    INSTANCE_COLUMNS,
    INSTANCES_CSV,
    PER_IMAGE_COLUMNS,
    PER_IMAGE_CSV,
    PREDICTIONS_DIR,
    build_result,
    file_entry,
    prediction_entry,
    sha256_bytes,
    workspace_of,
)
from foam_cell_analysis.jobs.protocol import atomic_write_json

GIB = 1024**3
UINT16_LABEL_LIMIT = 65535
UNCLASSIFIED = "未分類"

Emit = Callable[..., dict[str, Any]]


def _timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat()


# ---- 原子的な書き込み（4.1） ----


def _replace(source: Path, target: Path) -> None:
    """os.replace。Windows の一時的な共有違反は 0.1 秒間隔で 5 回まで再試行する。"""
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """同じフォルダの *.tmp に書き、fsync してから完成名へ置き換える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _replace(temporary, path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """allow_nan=False で検査してから原子的に JSON を書く。"""
    json.dumps(value, ensure_ascii=False, allow_nan=False)
    atomic_write_json(path, value)


def encode_label(labels: np.ndarray) -> tuple[str, bytes, str]:
    """予測ラベルを（拡張子, バイト列, dtype）へ符号化する。

    最大ラベルが 65,535 以下なら uint16 PNG、超えれば uint32 TIFF（学習の write_label と同じ規則）。
    """
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("予測ラベルは 2 次元の整数画像である必要があります")
    if values.size and int(values.min()) < 0:
        raise ValueError("予測ラベルに負の値があります")
    buffer = io.BytesIO()
    if int(values.max(initial=0)) > UINT16_LABEL_LIMIT:
        import tifffile

        tifffile.imwrite(buffer, values.astype(np.uint32, copy=False))
        return ".tif", buffer.getvalue(), "uint32"
    Image.fromarray(values.astype(np.uint16, copy=False)).save(buffer, format="PNG")
    return ".png", buffer.getvalue(), "uint16"


def _csv_bytes(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})
    return stream.getvalue().encode("utf-8")


def _counts_bytes(rows: list[dict[str, Any]], thresholds: list[float]) -> bytes:
    """学習の eval/fold_k/epoch_XXX.npz と同じ形式（item_ids, tp, fp, fn）に thresholds を足す。

    thresholds は読み手が IoU 閾値を確かめるための追加の配列（学習側の読み方で読める）。
    """
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        item_ids=np.asarray([row["item_id"] for row in rows]),
        tp=np.stack([np.asarray(row["tp"], dtype=np.int64) for row in rows]),
        fp=np.stack([np.asarray(row["fp"], dtype=np.int64) for row in rows]),
        fn=np.stack([np.asarray(row["fn"], dtype=np.int64) for row in rows]),
        thresholds=np.asarray(thresholds, dtype=np.float64),
    )
    return buffer.getvalue()


# ---- 読み込み ----


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    """CSV を item_id で辞書化する。item_id の重複・欠落は ValueError。"""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "item_id" not in reader.fieldnames:
            raise ValueError(f"item_id 列がありません: {path.name}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            item_id = row.get("item_id")
            if not item_id:
                raise ValueError(f"item_id が空の行があります: {path.name}")
            if item_id in rows:
                raise ValueError(f"item_id が重複しています ({item_id}): {path.name}")
            rows[item_id] = row
        return list(reader.fieldnames), rows


def _dataset_file(dataset_dir: Path, relative: str | None) -> Path:
    if not relative:
        raise ValueError("manifest のパスが空です")
    path = (dataset_dir / relative).resolve()
    try:
        path.relative_to(dataset_dir.resolve())
    except ValueError as error:
        raise ValueError(f"manifest のパスがデータセット外を指しています: {relative}") from error
    if not path.is_file():
        raise ValueError(f"データファイルがありません: {relative}")
    return path


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image).copy()


def _check_pair(item_id: str, image: np.ndarray, mask: np.ndarray) -> None:
    """3.3 の 4: 画像は 2 次元 uint8/uint16、マスクは同じ大きさの非負の整数ラベル。"""
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"画像は 2 次元の uint8 / uint16 である必要があります: {item_id}")
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"マスクは 2 次元の整数ラベルである必要があります: {item_id}")
    if mask.shape != image.shape:
        raise ValueError(f"画像とマスクの大きさが一致しません: {item_id}")
    if mask.size and int(mask.min()) < 0:
        raise ValueError(f"マスクに負のラベルがあります: {item_id}")


def _n_labels(labels: np.ndarray) -> int:
    values = np.asarray(labels)
    return int(np.unique(values[values != 0]).size)


# ---- 事前検査（3.3・3.4） ----


def check_contamination(
    workspace: Path,
    source: dict[str, Any],
    validation_hashes: dict[str, str],
) -> dict[str, Any]:
    """学習混入の検査（3.4）。{status: none|found|unknown, pairs, reason} を返す。

    validation_hashes は検証画像の item_id → 照合済みの sha256。
    比べるのは、元の試行の学習版の manifest 上の画像 sha256（used_item_ids に含まれるもの）。
    """
    version = source.get("dataset_version")
    used = source.get("used_item_ids")
    if not version or used is None:
        return {"status": "unknown", "pairs": [], "reason": "学習時のデータの記録を確認できません"}
    folder = workspace / "datasets" / version
    if not folder.is_dir():
        return {
            "status": "unknown",
            "pairs": [],
            "reason": f"学習用データセット {version} が見つかりません",
        }
    try:
        recorded = source.get("dataset_sha256") or {}
        for name, expected in recorded.items():
            if name not in {"manifest.csv", "metadata.csv"}:
                continue
            if _file_sha256(folder / name) != str(expected).lower():
                return {
                    "status": "unknown",
                    "pairs": [],
                    "reason": f"学習用データセット {version} が学習時の内容と一致しません",
                }
        _columns, manifest = _read_csv(folder / "manifest.csv")
    except (OSError, ValueError, csv.Error, UnicodeDecodeError) as error:
        return {
            "status": "unknown",
            "pairs": [],
            "reason": f"学習用データセット {version} を読めません: {error}",
        }
    by_hash: dict[str, list[str]] = {}
    for item_id in used:
        row = manifest.get(item_id)
        digest = (row or {}).get("image_sha256")
        if not digest:
            return {
                "status": "unknown",
                "pairs": [],
                "reason": f"学習画像 {item_id} の記録が学習用データセットにありません",
            }
        by_hash.setdefault(digest.lower(), []).append(item_id)
    pairs = [
        {"validation_item_id": item_id, "training_item_id": training_id}
        for item_id, digest in sorted(validation_hashes.items())
        for training_id in by_hash.get(digest.lower(), [])
    ]
    return {"status": "found" if pairs else "none", "pairs": pairs, "reason": None}


def run_preflight(
    run_dir: str | Path,
    spec: dict[str, Any],
    emit: Emit,
    *,
    device: str,
    free_bytes_fn: Callable[[Path], int] | None = None,
) -> dict[str, Any]:
    """3.3 の事前検査を行い、resolved_data.json を書いて preflight を出す。

    どれかに失敗すれば ValueError（評価は failed）。学習混入が found でも続け、warning を出す。
    """
    run_path = Path(run_dir).resolve()
    workspace = workspace_of(run_path)
    validation = spec["validation"]
    version = validation["version"]
    datasets_root = (workspace / "datasets").resolve()
    dataset_dir = (workspace / validation["path"]).resolve()
    if dataset_dir.parent != datasets_root or dataset_dir.name != version:
        raise ValueError("検証用データセットの場所が不正です")
    # 1. dataset_info.json
    try:
        info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"検証用データセット {version} の情報を読めません") from error
    if info.get("purpose") != "val" or info.get("status") != "RELEASED":
        raise ValueError(f"{version} は確定済みの検証用データセットではありません")
    if info.get("dataset_version") != version:
        raise ValueError(f"データセットの版名とフォルダ名が一致しません: {version}")
    # 3（前半）. manifest・metadata の sha256
    for name, expected in validation["sha256"].items():
        path = dataset_dir / name
        if not path.is_file() or _file_sha256(path) != expected.lower():
            raise ValueError(f"検証用データセットの {name} が評価の準備時と一致しません")
    # 2. manifest と metadata の 1 対 1
    _columns, manifest = _read_csv(dataset_dir / "manifest.csv")
    _columns, metadata = _read_csv(dataset_dir / "metadata.csv")
    if manifest.keys() != metadata.keys():
        raise ValueError("manifest.csv と metadata.csv の item_id が 1 対 1 に対応していません")
    item_ids = list(validation["item_ids"])
    if item_ids != sorted(manifest):
        raise ValueError("評価対象が検証用データセットの全画像と一致しません")
    # 3（後半）・4. 画像・マスクの sha256 と形式
    verified: dict[str, dict[str, str]] = {}
    items: dict[str, dict[str, Any]] = {}
    paths: dict[str, tuple[Path, Path]] = {}
    for item_id in item_ids:
        row = manifest[item_id]
        image_path = _dataset_file(dataset_dir, row.get("image_path"))
        mask_path = _dataset_file(dataset_dir, row.get("mask_path"))
        image_bytes, mask_bytes = image_path.read_bytes(), mask_path.read_bytes()
        image_hash, mask_hash = sha256_bytes(image_bytes), sha256_bytes(mask_bytes)
        if (
            image_hash != str(row.get("image_sha256", "")).lower()
            or mask_hash != str(row.get("mask_sha256", "")).lower()
        ):
            raise ValueError(f"画像またはマスクの sha256 が manifest と一致しません: {item_id}")
        image, mask = _decode(image_bytes), _decode(mask_bytes)
        _check_pair(item_id, image, mask)
        verified[item_id] = {"image": image_hash, "mask": mask_hash}
        paths[item_id] = (image_path, mask_path)
        items[item_id] = {
            "shape": [int(value) for value in image.shape],
            "n_true": _n_labels(mask),
            "classification": metadata[item_id].get("classification") or None,
        }
    # 5. 重み
    weights = spec["weights"]
    from foam_cell_analysis.services.comparison_service import resolve_recorded_path

    weights_path = resolve_recorded_path(workspace, weights["path"])
    if not weights_path.is_file():
        raise ValueError("候補のモデルファイルがありません")
    if weights_path.stat().st_size != weights["size"]:
        raise ValueError("候補のモデルファイルの大きさが記録と一致しません")
    if _file_sha256(weights_path) != weights["sha256"].lower():
        raise ValueError("候補のモデルファイルの sha256 が記録と一致しません")
    # 6. 学習混入
    contamination = check_contamination(
        workspace,
        spec["contamination_source"],
        {item_id: hashes["image"] for item_id, hashes in verified.items()},
    )
    if contamination["status"] == "found":
        emit(
            "warning",
            message=(
                f"検証画像と同じ画像が学習データに {len(contamination['pairs'])} 件含まれています。"
                "この評価ではリリースできません"
            ),
        )
    elif contamination["status"] == "unknown":
        emit("warning", message=f"学習混入を確認できません: {contamination['reason']}")
    # 7. 容量
    required = sum(int(np.prod(item["shape"])) * 4 for item in items.values()) + GIB
    get_free = free_bytes_fn or (lambda path: shutil.disk_usage(path).free)
    free = int(get_free(run_path))
    if free < required:
        raise ValueError(f"空き容量が不足しています (必要 {required} バイト、空き {free} バイト)")
    per_class_counts: dict[str, int] = {}
    for item in items.values():
        name = item["classification"] or UNCLASSIFIED
        per_class_counts[name] = per_class_counts.get(name, 0) + 1
    resolved = {
        "schema": 1,
        "run_id": spec["run_id"],
        "validation_version": version,
        "dataset_sha256": {name: value.lower() for name, value in validation["sha256"].items()},
        "verified_hashes": verified,
        "weights_sha256": weights["sha256"].lower(),
        "items": items,
        "per_class_counts": per_class_counts,
        "contamination": contamination,
        "required_bytes": required,
        "free_bytes": free,
        "device": str(device),
    }
    write_json_atomic(run_path / "resolved_data.json", resolved)
    emit(
        "preflight",
        n_images=len(item_ids),
        per_class_counts=per_class_counts,
        contamination=contamination,
        required_bytes=required,
        free_bytes=free,
    )
    return {"resolved": resolved, "paths": paths, "weights_path": weights_path}


# ---- 評価の本体（7.3） ----


def execute_evaluation(
    run_dir: str | Path,
    spec: dict[str, Any],
    emit: Emit,
    *,
    device: Any,
    versions: dict[str, str] | None = None,
    free_bytes_fn: Callable[[Path], int] | None = None,
    adapter_factory: Callable[..., Any] | None = None,
    set_phase: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """事前検査から result.json までを行い、result の内容を返す。"""
    from foam_cell_analysis.evaluation.ap import aggregate, instance_table, match_counts
    from foam_cell_analysis.inference.adapters import build_inference_adapter, run_inference

    def phase(name: str) -> None:
        if set_phase is not None:
            set_phase(name)

    run_path = Path(run_dir).resolve()
    phase("preflight")
    checked = run_preflight(run_path, spec, emit, device=str(device), free_bytes_fn=free_bytes_fn)
    resolved = checked["resolved"]
    phase("model")
    factory = adapter_factory or build_inference_adapter
    adapter = factory(
        model_type=spec["model_type"],
        model_config=spec["model_config"],
        weights_path=checked["weights_path"],
        effective_params=spec["effective_params"],
        device=device,
    )
    phase("inference")
    item_ids = list(spec["validation"]["item_ids"])
    total = len(item_ids)
    predictions: dict[str, dict[str, Any]] = {}
    count_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    extra: dict[str, dict[str, Any]] = {}
    classifications = {
        item_id: item["classification"] for item_id, item in resolved["items"].items()
    }
    for index, item_id in enumerate(item_ids, start=1):
        image_path, mask_path = checked["paths"][item_id]
        # preflight の後に差し替えられていないかを、デコードの前に sha256 で確かめる
        image_bytes, mask_bytes = image_path.read_bytes(), mask_path.read_bytes()
        expected = resolved["verified_hashes"][item_id]
        if (
            sha256_bytes(image_bytes) != expected["image"]
            or sha256_bytes(mask_bytes) != expected["mask"]
        ):
            raise ValueError(
                f"評価の途中で画像またはマスクが変更されました。評価をやり直してください: {item_id}"
            )
        image, mask = _decode(image_bytes), _decode(mask_bytes)
        _check_pair(item_id, image, mask)
        prediction = np.asarray(
            run_inference(adapter, image, spec.get("preprocessing"), spec["model_type"])
        )
        if prediction.shape != image.shape:
            raise ValueError(f"予測の大きさが画像と一致しません: {item_id}")
        suffix, data, dtype = encode_label(prediction)
        relative = f"{PREDICTIONS_DIR}/{item_id}{suffix}"
        write_bytes_atomic(run_path / PREDICTIONS_DIR / f"{item_id}{suffix}", data)
        predictions[item_id] = prediction_entry(relative, prediction.shape, dtype, data)
        tp, fp, fn = match_counts(mask, prediction)
        count_rows.append({"item_id": item_id, "tp": tp, "fp": fp, "fn": fn})
        for row in instance_table(mask, prediction):
            instance_rows.append({"item_id": item_id, **row})
        extra[item_id] = {"n_true": _n_labels(mask), "n_pred": _n_labels(prediction)}
        emit("image_done", item_id=item_id, completed=index, total=total)
    # 9 章の集計
    phase("aggregate")
    class_names = sorted({name or UNCLASSIFIED for name in classifications.values()})
    summary = aggregate(count_rows, classifications, class_names)
    per_image_rows = [
        {
            "item_id": row["item_id"],
            "classification": classifications.get(row["item_id"]) or UNCLASSIFIED,
            "ap": summary["per_image"][row["item_id"]],
            # IoU 0.5 の値（閾値の先頭）
            "tp": int(row["tp"][0]),
            "fp": int(row["fp"][0]),
            "fn": int(row["fn"][0]),
            **extra[row["item_id"]],
        }
        for row in count_rows
    ]
    artifacts = {}
    for key, name, data in (
        ("per_image_csv", PER_IMAGE_CSV, _csv_bytes(PER_IMAGE_COLUMNS, per_image_rows)),
        ("eval_counts", EVAL_COUNTS, _counts_bytes(count_rows, list(spec["metric"]["thresholds"]))),
        ("instances_csv", INSTANCES_CSV, _csv_bytes(INSTANCE_COLUMNS, instance_rows)),
    ):
        write_bytes_atomic(run_path / name, data)
        artifacts[key] = file_entry(name, data)
    result = build_result(
        spec=spec,
        overall={"ap": summary["ap"], "n_images": summary["n_images"]},
        per_class=summary["per_class"],
        pooled_ap=summary["pooled_ap_reference"],
        contamination=resolved["contamination"],
        predictions=predictions,
        per_image_csv=artifacts["per_image_csv"],
        eval_counts=artifacts["eval_counts"],
        instances_csv=artifacts["instances_csv"],
        device=str(device),
        completed_at=_timestamp(),
        versions=versions,
    )
    # result.json は全成果物の保存が終わった後に最後に書く
    phase("finalize")
    write_json_atomic(run_path / "result.json", result)
    emit("completed", path="result.json")
    return result
