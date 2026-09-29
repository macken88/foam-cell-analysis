"""評価プロセスとの取り決め（比較・推論設計 4.2・7.4）。torch を読み込まない。

イベント外形・JSON Lines・原子的な書き込みは ``jobs/protocol.py`` と共有する。
ここには評価用の run_spec の検証、評価イベントの必須項目、result.json の組み立てを置く。
result.json の妥当性（7.5）は
``services/comparison_service.validate_evaluation_result`` で判定する。
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from foam_cell_analysis.jobs.protocol import (
    PROTOCOL_VERSION,
    append_jsonl,
    read_json,
    read_jsonl_events,
    validate_envelope,
    validate_hello,
)

SCHEMA = 1
RUN_SPEC_FIELDS = frozenset(
    {
        "schema",
        "protocol",
        "run_id",
        "candidate_id",
        "evaluation_id",
        "weights",
        "model_type",
        "model_config",
        "preprocessing",
        "effective_params",
        "validation",
        "contamination_source",
        "metric",
        "device_request",
        "app_version",
        "git_commit",
        "input_fingerprint",
    }
)
EVENT_FIELDS: dict[str, frozenset[str]] = {
    "started": frozenset({"device", "versions", "version_mismatches"}),
    "preflight": frozenset(
        {"n_images", "per_class_counts", "contamination", "required_bytes", "free_bytes"}
    ),
    "image_done": frozenset({"item_id", "completed", "total"}),
    "warning": frozenset({"message"}),
    "error": frozenset({"phase", "message"}),
    "completed": frozenset({"path"}),
}
CONTAMINATION_STATUSES = ("none", "found", "unknown")
PREDICTIONS_DIR = "predictions"
PER_IMAGE_CSV = "per_image.csv"
EVAL_COUNTS = "eval_counts.npz"
INSTANCES_CSV = "instances.csv"
PER_IMAGE_COLUMNS = ("item_id", "classification", "ap", "tp", "fp", "fn", "n_true", "n_pred")
INSTANCE_COLUMNS = ("item_id", "side", "label", "area", "best_iou", "best_label")

_CANDIDATE_ID = re.compile(r"RC-\d{3,}")
_EVALUATION_ID = re.compile(r"eval_\d{3,}")
_VERSION = re.compile(r"[A-Za-z0-9_.-]+")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
# Windows でファイル名に使えない文字と制御文字（予測は predictions/<item_id>.png に保存する）
_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_finite_number(value: Any) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def safe_item_id(value: Any) -> bool:
    """予測ファイル名に使える item_id か（空・.・..・区切り文字・禁止文字を拒否する）。"""
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and not value.endswith((" ", "."))
        and _UNSAFE_NAME.search(value) is None
    )


def _relative_path_ok(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    posix_path = PurePosixPath(value.replace("\\", "/"))
    windows_path = PureWindowsPath(value)
    return not (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or ".." in posix_path.parts
    )


def evaluation_run_id(candidate_id: str, validation_version: str, evaluation_id: str) -> str:
    """評価の run_id（RC-001/val_v000/eval_001）を作る。"""
    return f"{candidate_id}/{validation_version}/{evaluation_id}"


def parse_run_id(run_id: str) -> tuple[str, str, str]:
    """run_id を（候補 ID, 検証版, 評価 ID）へ分ける。形が違えば ValueError。"""
    parts = run_id.split("/") if isinstance(run_id, str) else []
    if (
        len(parts) != 3
        or _CANDIDATE_ID.fullmatch(parts[0]) is None
        or _VERSION.fullmatch(parts[1]) is None
        or parts[1] in {".", ".."}
        or _EVALUATION_ID.fullmatch(parts[2]) is None
    ):
        raise ValueError(f"評価の run_id が不正です: {run_id}")
    return parts[0], parts[1], parts[2]


def validate_run_spec(spec: dict[str, Any], run_dir: str | Path | None = None) -> None:
    """評価の run_spec.json（4.2）を検証する。不正なら ValueError。

    run_dir を渡すと、フォルダの位置（candidates/<候補>/evaluations/<版>/<評価>）も照合する。
    """
    if not isinstance(spec, dict) or not RUN_SPEC_FIELDS.issubset(spec):
        missing = sorted(RUN_SPEC_FIELDS - set(spec or {}))
        raise ValueError("run_spec の必須項目がありません: " + ", ".join(missing))
    if type(spec["schema"]) is not int or spec["schema"] != SCHEMA:
        raise ValueError("run_spec の schema が不正です")
    if type(spec["protocol"]) is not int or spec["protocol"] != PROTOCOL_VERSION:
        raise ValueError("run_spec の protocol が不正です")
    candidate_id, version, evaluation_id = parse_run_id(spec["run_id"])
    if spec["candidate_id"] != candidate_id or spec["evaluation_id"] != evaluation_id:
        raise ValueError("run_id と candidate_id・evaluation_id が一致しません")
    weights = spec["weights"]
    if (
        not isinstance(weights, dict)
        or not _relative_path_ok(weights.get("path"))
        or type(weights.get("size")) is not int
        or weights["size"] < 0
        or not _is_sha256(weights.get("sha256"))
    ):
        raise ValueError("run_spec の weights が不正です")
    if not isinstance(spec["model_type"], str) or not spec["model_type"]:
        raise ValueError("run_spec の model_type が不正です")
    if not isinstance(spec["model_config"], dict):
        raise ValueError("run_spec の model_config が不正です")
    preprocessing = spec["preprocessing"]
    if preprocessing is not None and not isinstance(preprocessing, dict):
        raise ValueError("run_spec の preprocessing が不正です")
    params = spec["effective_params"]
    if not isinstance(params, dict) or any(
        not isinstance(key, str)
        or type(value) not in {str, int, float, bool}
        or (type(value) in {int, float} and not _is_finite_number(value))
        for key, value in params.items()
    ):
        raise ValueError("run_spec の effective_params が不正です")
    validation = spec["validation"]
    if not isinstance(validation, dict) or validation.get("version") != version:
        raise ValueError("run_spec の validation.version が run_id と一致しません")
    if validation.get("path") != f"datasets/{version}":
        raise ValueError("validation.path は datasets/<版> である必要があります")
    hashes = validation.get("sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != {"manifest.csv", "metadata.csv"}
        or not all(_is_sha256(value) for value in hashes.values())
    ):
        raise ValueError("run_spec の validation.sha256 が不正です")
    item_ids = validation.get("item_ids")
    if (
        not isinstance(item_ids, list)
        or not item_ids
        or not all(safe_item_id(item_id) for item_id in item_ids)
        or len({item_id.casefold() for item_id in item_ids}) != len(item_ids)
    ):
        raise ValueError("run_spec の validation.item_ids が空または不正です")
    source = spec["contamination_source"]
    if not isinstance(source, dict):
        raise ValueError("run_spec の contamination_source が不正です")
    used = source.get("used_item_ids")
    if used is not None and (
        not isinstance(used, list) or not all(isinstance(item, str) for item in used)
    ):
        raise ValueError("contamination_source.used_item_ids が不正です")
    dataset_version = source.get("dataset_version")
    if dataset_version is not None and (
        not isinstance(dataset_version, str)
        or _VERSION.fullmatch(dataset_version) is None
        or dataset_version in {".", ".."}
    ):
        raise ValueError("contamination_source.dataset_version が不正です")
    metric = spec["metric"]
    if not isinstance(metric, dict) or not {
        "id",
        "thresholds",
        "aggregation",
        "empty_rule",
    }.issubset(metric):
        raise ValueError("run_spec の metric が不正です")
    if spec["device_request"] != "auto":
        raise ValueError("run_spec の device_request は auto だけに対応しています")
    if not _is_sha256(spec["input_fingerprint"]):
        raise ValueError("run_spec の input_fingerprint が不正です")
    if run_dir is not None:
        path = Path(run_dir)
        if (
            path.name != evaluation_id
            or path.parent.name != version
            or path.parents[1].name != "evaluations"
            or path.parents[2].name != candidate_id
        ):
            raise ValueError("run_spec と評価フォルダの位置が一致しません")


def workspace_of(run_dir: str | Path) -> Path:
    """評価フォルダから workspace を返す。

    評価フォルダは comparison/candidates/<候補>/evaluations/<版>/<評価> の形。
    """
    path = Path(run_dir)
    if path.parents[3].name != "candidates" or path.parents[4].name != "comparison":
        raise ValueError(f"評価フォルダの位置が不正です: {path}")
    return path.parents[5]


# ---- イベント ----


def validate_event(event: dict[str, Any], *, allow_hello: bool = True) -> None:
    """評価イベントの外形と型別の必須項目を検証する。不正なら ValueError。"""
    kind = event.get("type") if isinstance(event, dict) else None
    if kind == "hello" and allow_hello:
        validate_hello(event)
        return
    validate_envelope(event)
    if kind not in EVENT_FIELDS:
        raise ValueError(f"未対応の評価イベントです: {kind}")
    missing = EVENT_FIELDS[kind] - event.keys()
    if missing:
        raise ValueError(f"評価イベントの必須項目がありません: {', '.join(sorted(missing))}")
    if kind == "image_done":
        completed, total = event["completed"], event["total"]
        if (
            type(completed) is not int
            or type(total) is not int
            or not 0 < completed <= total
            or not isinstance(event["item_id"], str)
        ):
            raise ValueError("image_done の completed・total・item_id が不正です")


def read_events(path: str | Path, *, expected_run_id: str | None = None) -> list[dict[str, Any]]:
    """評価の events.jsonl を読む（切れた最終行は無視、seq の重複・欠落は ValueError）。"""
    return read_jsonl_events(
        path,
        validate=lambda event: validate_event(event, allow_hello=False),
        expected_run_id=expected_run_id,
    )


def append_event(path: str | Path, event: dict[str, Any]) -> None:
    """検証済みの評価イベントを追記して flush する。"""
    validate_event(event, allow_hello=False)
    append_jsonl(path, event)


def read_run_spec(run_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(run_dir) / "run_spec.json")


# ---- result.json ----


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prediction_entry(relative_path: str, shape: tuple[int, ...], dtype: str, data: bytes) -> dict:
    """result.json の predictions の 1 件（相対パス・shape・dtype・バイト数・sha256）。"""
    return {
        "path": relative_path,
        "shape": [int(value) for value in shape],
        "dtype": str(dtype),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def file_entry(relative_path: str, data: bytes) -> dict[str, Any]:
    """per_image.csv などの成果物の記録（相対パス・バイト数・sha256）。"""
    return {"path": relative_path, "bytes": len(data), "sha256": sha256_bytes(data)}


def build_result(
    *,
    spec: dict[str, Any],
    overall: dict[str, Any],
    per_class: dict[str, dict[str, Any]],
    pooled_ap: float | None,
    contamination: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    per_image_csv: dict[str, Any],
    eval_counts: dict[str, Any],
    instances_csv: dict[str, Any],
    device: str,
    completed_at: str,
    versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """評価の result.json（4.2 の完成 manifest）を組み立てる。"""
    if contamination.get("status") not in CONTAMINATION_STATUSES:
        raise ValueError(f"学習混入の検査結果が不正です: {contamination.get('status')}")
    missing = [item_id for item_id in spec["validation"]["item_ids"] if item_id not in predictions]
    if missing:
        raise ValueError(f"予測がない評価対象があります: {missing[:5]}")
    return {
        "schema": SCHEMA,
        "run_id": spec["run_id"],
        "candidate_id": spec["candidate_id"],
        "evaluation_id": spec["evaluation_id"],
        "validation_version": spec["validation"]["version"],
        "input_fingerprint": spec["input_fingerprint"],
        "overall": {"ap": overall.get("ap"), "n_images": int(overall.get("n_images") or 0)},
        "per_class": {
            str(name): {"ap": value.get("ap"), "n_images": int(value.get("n_images") or 0)}
            for name, value in per_class.items()
        },
        "pooled_ap": pooled_ap,
        "metric": dict(spec["metric"]),
        "contamination": {
            "status": contamination["status"],
            "pairs": list(contamination.get("pairs") or []),
            "reason": contamination.get("reason"),
        },
        "predictions": predictions,
        "per_image_csv": per_image_csv,
        "eval_counts": eval_counts,
        "instances_csv": instances_csv,
        "device": str(device),
        "versions": dict(versions or {}),
        "completed_at": completed_at,
    }
