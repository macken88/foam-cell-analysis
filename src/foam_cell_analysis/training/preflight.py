"""学習開始前のデータ・重み・容量検査。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from foam_cell_analysis.data.dataset_store import DatasetStore
from foam_cell_analysis.training.protocol import atomic_write_json, write_resolved_data

GIB = 1024**3


class _RunArrays:
    """DatasetStore の上限付き LRU を使う item_id→画像・ラベル参照。"""

    def __init__(self, store: DatasetStore, paths: dict[str, tuple[Path, Path]]) -> None:
        self.store = store
        self.paths = paths

    def __getitem__(self, item_id: str) -> tuple[Any, Any]:
        image_path, mask_path = self.paths[item_id]
        image = self.store._read_array(image_path)
        mask = self.store._read_array(mask_path)
        if image.dtype not in ("uint8", "uint16"):
            raise ValueError(f"画像は uint8 または uint16 である必要があります: {item_id}")
        if image.shape != mask.shape:
            raise ValueError(f"画像とマスクのサイズが一致しません: {item_id}")
        return image, mask

    def __iter__(self):
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


def estimate_required_bytes(
    weight_bytes: int,
    n_folds: int,
    n_epochs: int,
    validation_interval: int,
    save_every: int,
    save_fold_models: bool,
    image_shapes: list[tuple[int, int]],
) -> dict[str, int]:
    """設計書 12.2 の上限見積りを内訳付きで返す。"""
    if min(weight_bytes, n_folds, n_epochs, validation_interval, save_every) < 1:
        raise ValueError("容量見積りの引数は正の値が必要です")
    validation_epochs = set(range(validation_interval, n_epochs + 1, validation_interval))
    validation_epochs.add(n_epochs)
    save_epochs = set(range(save_every, n_epochs + 1, save_every))
    prediction_bytes = sum(height * width * 4 for height, width in image_shapes)
    fold_weights = (
        n_folds * len(validation_epochs | save_epochs) * weight_bytes if save_fold_models else 0
    )
    selected = n_folds * weight_bytes if save_fold_models else 0
    predictions = len(validation_epochs) * prediction_bytes
    final = weight_bytes
    writing = weight_bytes
    margin = GIB
    return {
        "fold_weights": fold_weights,
        "selected": selected,
        "predictions": predictions,
        "final": final,
        "writing": writing,
        "margin": margin,
        "required": fold_weights + selected + predictions + final + writing + margin,
        "validation_epochs": len(validation_epochs),
        "prediction_bytes_per_epoch": prediction_bytes,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_mismatches(constraints: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    versions: dict[str, str] = {}
    mismatches = []
    for package in ("torch", "torchvision", "cellpose"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "未インストール"
    if not constraints.is_file():
        return versions, mismatches
    for raw_line in constraints.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line or "==" not in line:
            continue
        package, expected = (part.strip() for part in line.split("==", maxsplit=1))
        package = package.replace("_", "-")
        actual = versions.get(package)
        if actual is None:
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                actual = "未インストール"
            versions[package] = actual
        normalized_expected = expected.split("+", maxsplit=1)[0]
        if actual.split("+", maxsplit=1)[0] != normalized_expected:
            mismatches.append({"package": package, "expected": expected, "actual": actual})
    return versions, mismatches


def _safe_dataset_path(dataset_dir: Path, relative: str) -> Path:
    path = (dataset_dir / relative).resolve()
    try:
        path.relative_to(dataset_dir.resolve())
    except ValueError as error:
        raise ValueError(f"manifest のパスがデータセット外を指しています: {relative}") from error
    if not path.is_file():
        raise ValueError(f"データファイルがありません: {relative}")
    return path


def run_preflight(
    run_dir: str | Path,
    spec: dict[str, Any],
    adapter: Any,
    device: Any,
    emit: Callable[..., dict[str, Any]],
    *,
    free_bytes_fn: Callable[[Path], int] | None = None,
    weight_size_fn: Callable[[Any], int] | None = None,
    constraints_path: str | Path | None = None,
) -> dict[str, Any]:
    """検査に合格した resolved_data を保存し、preflight イベントを出す。"""
    run_path = Path(run_dir).resolve()
    workspace_root = run_path.parents[3]
    constraints = (
        Path(constraints_path)
        if constraints_path is not None
        else Path(__file__).resolve().parents[3] / "constraints-ml.txt"
    )
    versions, mismatches = _version_mismatches(constraints)
    atomic_write_json(
        run_path / "environment.json",
        {"device": str(device), "versions": versions, "version_mismatches": mismatches},
    )
    for mismatch in mismatches:
        emit("warning", message=f"依存パッケージの版が制約と異なります: {mismatch}")
    dataset_version = str(spec["dataset"]["version"])
    dataset_dir = (workspace_root / spec["dataset"]["path"]).resolve()
    if dataset_dir.parent != (workspace_root / "datasets").resolve():
        raise ValueError("dataset path は workspace/datasets/<version> である必要があります")
    dataset_hashes = spec["dataset"].get("sha256", {})
    for filename, expected_hash in dataset_hashes.items():
        path = dataset_dir / filename
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"データセット metadata の sha256 が一致しません: {filename}")
    store = DatasetStore(workspace_root)
    items = {item.item_id: item for item in store.get_items(dataset_version)}
    manifest = DatasetStore._read_csv(dataset_dir / "manifest.csv")
    used_ids = list(spec["used_item_ids"])
    if any(
        not isinstance(item_id, str)
        or not item_id
        or item_id in {".", ".."}
        or "/" in item_id
        or "\\" in item_id
        for item_id in used_ids
    ):
        raise ValueError("used_item_ids に不正なファイル名があります")
    missing_ids = sorted(set(used_ids) - items.keys())
    if missing_ids:
        raise ValueError(f"run_spec の学習画像がデータセットにありません: {missing_ids}")
    model_config = spec["config"]["model"]
    paths: dict[str, tuple[Path, Path]] = {}
    image_shapes: dict[str, tuple[int, int]] = {}
    instance_counts: dict[str, int] = {}
    model_excluded: set[str] = set()
    exclusion_reasons: dict[str, str] = {}
    verified_hashes: dict[str, dict[str, str]] = {}
    for item_id in used_ids:
        row = manifest[item_id]
        image_path = _safe_dataset_path(dataset_dir, row["image_path"])
        mask_path = _safe_dataset_path(dataset_dir, row["mask_path"])
        image_hash, mask_hash = _sha256(image_path), _sha256(mask_path)
        if image_hash != row.get("image_sha256") or mask_hash != row.get("mask_sha256"):
            raise ValueError(f"画像またはマスクの sha256 が一致しません: {item_id}")
        paths[item_id] = (image_path, mask_path)
        image = store.get_image(dataset_version, item_id)
        mask = store.get_mask(dataset_version, item_id)
        if image.shape != mask.shape:
            raise ValueError(f"画像とマスクのサイズが一致しません: {item_id}")
        image_shapes[item_id] = image.shape
        instance_counts[item_id] = int(np.unique(mask[mask != 0]).size)
        reason = adapter.training_exclusion_reason(mask, model_config)
        if reason is None and adapter.should_exclude_training_item(mask, model_config):
            reason = "model_filter"
        if reason is not None:
            model_excluded.add(item_id)
            exclusion_reasons[item_id] = reason
        verified_hashes[item_id] = {"image": image_hash, "mask": mask_hash}
    arrays = _RunArrays(store, paths)

    fold_assignments = {str(key): int(value) for key, value in spec["fold_assignments"].items()}
    folds = sorted(set(fold_assignments.values()))
    if set(used_ids) != set(fold_assignments):
        raise ValueError("used_item_ids と fold_assignments が一致しません")
    excluded: list[dict[str, Any]] = []
    excluded_final: list[str] = []
    training_ids: dict[int, list[str]] = {}
    validation_ids: dict[int, list[str]] = {}
    final_training_ids = [item_id for item_id in used_ids if item_id not in model_excluded]
    excluded_final.extend(sorted(model_excluded))
    if not final_training_ids:
        raise ValueError("モデル固有の除外後に学習画像がありません")
    for fold in folds:
        validation_ids[fold] = [
            item_id for item_id in used_ids if fold_assignments[item_id] == fold
        ]
        if not validation_ids[fold]:
            raise ValueError(f"fold {fold} の検証画像がありません")
        train_ids = [item_id for item_id in used_ids if fold_assignments[item_id] != fold]
        retained = []
        for item_id in train_ids:
            if item_id in model_excluded:
                excluded.append(
                    {"fold": fold, "item_id": item_id, "reason": exclusion_reasons[item_id]}
                )
            else:
                retained.append(item_id)
        if not retained:
            raise ValueError(f"fold {fold} の学習画像がありません")
        training_ids[fold] = retained

    initial_file = adapter.initial_weights_file(model_config)
    if initial_file and not Path(initial_file).is_file():
        emit(
            "warning", message="初期重みが未取得です。初回実行ではダウンロードされる場合があります"
        )
    adapter.build(model_config, device)
    initial_file = adapter.initial_weights_file(model_config)
    initial_hash = (
        _sha256(Path(initial_file)) if initial_file and Path(initial_file).is_file() else None
    )
    expected_hash = spec.get("expected_initial_weights_sha256")
    if expected_hash and initial_hash != expected_hash:
        raise ValueError("初期重みの sha256 が前回試行と一致しません")
    weight_bytes = int(weight_size_fn(adapter) if weight_size_fn else adapter.weights_nbytes())
    training = spec["config"]["training"]
    checkpoint = spec["config"]["checkpoint"]
    capacity = estimate_required_bytes(
        weight_bytes,
        len(folds),
        int(training["epochs"]),
        int(checkpoint["validation_interval"]),
        int(checkpoint["save_every"]),
        bool(checkpoint.get("save_fold_models", True)),
        [image_shapes[item_id] for item_id in used_ids],
    )
    get_free = free_bytes_fn or (lambda path: shutil.disk_usage(path).free)
    free_bytes = int(get_free(run_path))
    if free_bytes < capacity["required"]:
        raise ValueError(
            f"空き容量が不足しています (required={capacity['required']}, free={free_bytes})"
        )

    fold_counts = {
        str(fold): {
            "train": len(training_ids[fold]),
            "validation": len(validation_ids[fold]),
        }
        for fold in folds
    }
    resolved = {
        "schema": 1,
        "run_id": spec["run_id"],
        "used_item_ids": used_ids,
        "fold_assignments": fold_assignments,
        "training_item_ids": {str(k): value for k, value in training_ids.items()},
        "final_training_item_ids": final_training_ids,
        "validation_item_ids": {str(k): value for k, value in validation_ids.items()},
        "excluded": excluded,
        "excluded_final": excluded_final,
        "excluded_final_reasons": [
            {"item_id": item_id, "reason": exclusion_reasons[item_id]}
            for item_id in sorted(model_excluded)
        ],
        "image_shapes": {item_id: list(shape) for item_id, shape in image_shapes.items()},
        "ground_truth_instance_counts": instance_counts,
        "excluded_by_fold": {
            str(fold): [row for row in excluded if row["fold"] == fold] for fold in folds
        },
        "verified_hashes": verified_hashes,
        "initial_weights_sha256": initial_hash,
        "free_bytes": free_bytes,
        "weights_nbytes": weight_bytes,
        "required_bytes": capacity["required"],
        "capacity": capacity,
    }
    write_resolved_data(run_path, resolved)
    emit(
        "preflight",
        n_used=len(used_ids),
        folds=fold_counts,
        excluded=excluded,
        required_bytes=capacity["required"],
        free_bytes=free_bytes,
    )
    return {
        "resolved": resolved,
        "arrays": arrays,
        "folds": folds,
        "versions": versions,
        "items": list(items.values()),
    }
