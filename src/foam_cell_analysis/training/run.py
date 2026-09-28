"""子プロセス起動・go ハンドシェイク・学習実行の入口。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TextIO


def _timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat()


def _is_finite_number(value: Any) -> bool:
    """bool を除く有限の整数または浮動小数点数か確認する。"""
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _validate_spec(run_dir: Path, spec: dict[str, Any]) -> None:
    required = {
        "schema",
        "protocol",
        "run_id",
        "experiment_id",
        "attempt",
        "config",
        "dataset",
        "used_item_ids",
        "fold_assignments",
        "metric",
        "seeds",
        "augmentation_profile",
        "preprocessing",
        "eval_params",
    }
    if (
        not required.issubset(spec)
        or type(spec.get("schema")) is not int
        or spec.get("schema") != 1
        or type(spec.get("protocol")) is not int
        or spec.get("protocol") != 1
    ):
        raise ValueError("run_spec の必須項目または schema/protocol が不正です")
    if not (run_dir / "run_spec.json").is_file():
        raise ValueError("run_spec.json がありません")
    experiment_id = spec["experiment_id"]
    attempt = spec["attempt"]
    if not isinstance(experiment_id, str) or not experiment_id or "/" in experiment_id:
        raise ValueError("experiment_id が不正です")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt が不正です")
    if spec["run_id"] != f"{experiment_id}/attempt_{attempt:03d}":
        raise ValueError("run_id が不正です")
    if (
        not isinstance(spec["used_item_ids"], list)
        or not spec["used_item_ids"]
        or any(
            not isinstance(item_id, str)
            or not item_id
            or item_id in {".", ".."}
            or "/" in item_id
            or "\\" in item_id
            for item_id in spec["used_item_ids"]
        )
        or len(set(spec["used_item_ids"])) != len(spec["used_item_ids"])
    ):
        raise ValueError("used_item_ids が空または不正です")
    dataset = spec["dataset"]
    if not isinstance(dataset, dict):
        raise ValueError("dataset が不正です")
    version = dataset.get("version")
    dataset_path = dataset.get("path")
    dataset_hashes = dataset.get("sha256")
    if not isinstance(version, str) or not version:
        raise ValueError("dataset.version が不正です")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError("dataset.path が不正です")
    posix_path = PurePosixPath(dataset_path.replace("\\", "/"))
    windows_path = PureWindowsPath(dataset_path)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or posix_path.parts != ("datasets", version)
    ):
        raise ValueError("dataset.path は datasets/<version> の相対パスである必要があります")
    if not isinstance(dataset_hashes, dict) or set(dataset_hashes) != {
        "manifest.csv",
        "metadata.csv",
    }:
        raise ValueError("dataset.sha256 が不正です")
    for name in ("manifest.csv", "metadata.csv"):
        digest = dataset_hashes[name]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"dataset.sha256.{name} が不正です")
    config = spec["config"]
    if not isinstance(config, dict) or not {"model", "training", "data", "checkpoint"}.issubset(
        config
    ):
        raise ValueError("config の必須項目がありません")
    model = config["model"]
    training = config["training"]
    checkpoint = config["checkpoint"]
    data = config["data"]
    if not isinstance(model, dict) or not isinstance(model.get("type"), str) or not model["type"]:
        raise ValueError("config.model.type が不正です")
    if not isinstance(training, dict):
        raise ValueError("config.training が不正です")
    if type(training.get("epochs")) is not int or training["epochs"] < 1:
        raise ValueError("config.training.epochs が不正です")
    learning_rate = training.get("learning_rate")
    if not _is_finite_number(learning_rate) or learning_rate <= 0:
        raise ValueError("config.training.learning_rate が不正です")
    if "weight_decay" in training and (
        not _is_finite_number(training["weight_decay"]) or training["weight_decay"] < 0
    ):
        raise ValueError("config.training.weight_decay が不正です")
    early = training.get("early_stopping")
    if not isinstance(early, dict) or type(early.get("enabled")) is not bool:
        raise ValueError("config.training.early_stopping が不正です")
    if type(early.get("patience")) is not int or early["patience"] < 1:
        raise ValueError("config.training.early_stopping.patience が不正です")
    if not isinstance(checkpoint, dict):
        raise ValueError("config.checkpoint が不正です")
    for key in ("validation_interval", "save_every"):
        if type(checkpoint.get(key)) is not int or checkpoint[key] < 1:
            raise ValueError(f"config.checkpoint.{key} が不正です")
    if type(checkpoint.get("save_fold_models")) is not bool:
        raise ValueError("config.checkpoint.save_fold_models が不正です")
    if not isinstance(data, dict) or data.get("dataset_version") != version:
        raise ValueError("config.data.dataset_version が不正です")
    if not isinstance(spec["fold_assignments"], dict) or not spec["fold_assignments"]:
        raise ValueError("fold_assignments が空または不正です")
    if not isinstance(spec["seeds"], dict) or not {"folds", "final"}.issubset(spec["seeds"]):
        raise ValueError("seeds が不正です")
    folds = set(spec["fold_assignments"].values())
    if any(type(fold) is not int or fold < 1 for fold in folds):
        raise ValueError("fold_assignments の fold 番号が不正です")
    if set(spec["fold_assignments"]) != set(spec["used_item_ids"]):
        raise ValueError("fold_assignments と used_item_ids が一致しません")
    if len(folds) < 2:
        raise ValueError("fold_assignments には 2 つ以上の fold が必要です")
    fold_seeds = spec["seeds"].get("folds")
    if not isinstance(fold_seeds, dict) or set(fold_seeds) != {str(fold) for fold in folds}:
        raise ValueError("seeds.folds が不正です")
    if any(type(seed) is not int for seed in fold_seeds.values()):
        raise ValueError("seeds.folds の値が不正です")
    if type(spec["seeds"].get("final")) is not int:
        raise ValueError("seeds.final が不正です")
    metric = spec["metric"]
    if not isinstance(metric, dict) or not {
        "id",
        "thresholds",
        "aggregation",
        "empty_rule",
    }.issubset(metric):
        raise ValueError("metric が不正です")
    if (
        not isinstance(metric["id"], str)
        or not isinstance(metric["thresholds"], list)
        or not metric["thresholds"]
        or any(
            not _is_finite_number(threshold) or not 0 <= threshold <= 1
            for threshold in metric["thresholds"]
        )
        or not isinstance(metric["aggregation"], str)
        or not isinstance(metric["empty_rule"], str)
    ):
        raise ValueError("metric の値が不正です")
    profile = spec["augmentation_profile"]
    if not isinstance(profile, dict) or not isinstance(profile.get("value"), dict):
        raise ValueError("augmentation_profile が不正です")
    profile_value = profile["value"]
    transforms = profile_value.get("transforms", [])
    order = profile_value.get("pipeline_order", profile_value.get("order", []))
    if (
        not isinstance(transforms, list)
        or any(
            not isinstance(item, dict) or not isinstance(item.get("key"), str)
            for item in transforms
        )
        or not isinstance(order, list)
        or any(not isinstance(key, str) for key in order)
    ):
        raise ValueError("augmentation_profile.value が不正です")
    for setting in transforms:
        probability = setting.get("probability", 1.0)
        if (
            type(setting.get("enabled", True)) is not bool
            or not _is_finite_number(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("augmentation_profile.transforms の値が不正です")
        low, high = setting.get("range_min"), setting.get("range_max")
        if (low is None) != (high is None) or (
            low is not None and (not _is_finite_number(low) or not _is_finite_number(high))
        ):
            raise ValueError("augmentation_profile.transforms の範囲が不正です")
    preprocessing = spec["preprocessing"]
    if preprocessing is not None and not isinstance(preprocessing, dict):
        raise ValueError("preprocessing が不正です")
    if isinstance(preprocessing, dict):
        low = preprocessing.get("low_percentile", 1.0)
        high = preprocessing.get("high_percentile", 99.0)
        if not _is_finite_number(low) or not _is_finite_number(high) or not 0 <= low <= high <= 100:
            raise ValueError("preprocessing percentile が不正です")
    eval_params = spec["eval_params"]
    if not isinstance(eval_params, dict) or any(
        not isinstance(key, str)
        or type(value) not in {str, int, float, bool}
        or (type(value) in {int, float} and not _is_finite_number(value))
        for key, value in eval_params.items()
    ):
        raise ValueError("eval_params が不正です")


def _versions() -> dict[str, str]:
    import importlib.metadata

    versions = {}
    for package in ("torch", "torchvision", "cellpose"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "未インストール"
    return versions


def _watch_stdin(stream: TextIO) -> None:
    """go 後の親プロセス終了を監視し、EOF で子プロセスを終了する。"""
    if os.name == "nt":
        import ctypes
        import msvcrt
        import time

        handle = msvcrt.get_osfhandle(stream.fileno())
        available = ctypes.c_ulong()
        while ctypes.windll.kernel32.PeekNamedPipe(
            handle, None, 0, None, ctypes.byref(available), None
        ):
            if available.value:
                os.read(stream.fileno(), available.value)
            time.sleep(0.1)
    else:
        import select

        while True:
            select.select([stream], [], [])
            if not os.read(stream.fileno(), 4096):
                break
    os._exit(3)


def run_job(
    run_dir: str | Path,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    start_watchdog: bool = True,
    free_bytes_fn: Any = None,
    weight_size_fn: Any = None,
    adapter_factory: Callable[[], Any] | None = None,
) -> int:
    """テスト差し替え口を持つ run protocol 実装。"""
    from foam_cell_analysis.training.protocol import (
        append_event,
        atomic_write_json,
        read_run_spec,
    )

    run_path = Path(run_dir).resolve()
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    try:
        spec = read_run_spec(run_path)
        _validate_spec(run_path, spec)
    except (OSError, ValueError, KeyError, TypeError):
        return 2

    run_id = spec["run_id"]
    hello = {"v": 1, "run_id": run_id, "time": _timestamp(), "type": "hello", "pid": os.getpid()}
    output_stream.write(json.dumps(hello, ensure_ascii=False, separators=(",", ":")) + "\n")
    output_stream.flush()
    line = input_stream.readline()
    if not line:
        return 3
    if line.strip() != "go":
        return 3
    if start_watchdog:
        threading.Thread(target=_watch_stdin, args=(input_stream,), daemon=True).start()

    sequence = 0

    def emit(event_type: str, **fields: Any) -> dict[str, Any]:
        nonlocal phase, sequence
        sequence += 1
        event = {
            "v": 1,
            "run_id": run_id,
            "seq": sequence,
            "time": _timestamp(),
            "type": event_type,
            **fields,
        }
        append_event(run_path / "events.jsonl", event)
        output_stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        output_stream.flush()
        if event_type == "phase":
            phase = "final_training" if fields.get("phase") == "final" else "cross_validation"
        return event

    phase = "preflight"
    try:
        import torch

        torch.set_num_threads(1)

        from foam_cell_analysis.training.preflight import _version_mismatches, run_preflight

        constraints = Path(__file__).resolve().parents[3] / "constraints-ml.txt"
        versions = _versions()
        _, mismatches = _version_mismatches(constraints)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emit(
            "started",
            device=str(device),
            versions=versions,
            version_mismatches=mismatches,
        )

        model_type = spec["config"]["model"].get("type")
        if model_type != "fake":
            raise ValueError(f"段階 B で利用できるアダプタではありません: {model_type}")
        from foam_cell_analysis.training.adapters.fake import FakeAdapter
        from foam_cell_analysis.training.loop import execute_training

        model_factory = adapter_factory or FakeAdapter
        adapter = model_factory()
        preflight_data = run_preflight(
            run_path,
            spec,
            adapter,
            device,
            emit,
            free_bytes_fn=free_bytes_fn,
            weight_size_fn=weight_size_fn,
        )
        phase = "cross_validation"
        execute_training(run_path, spec, preflight_data, model_factory, device, emit)
        return 0
    except Exception as error:
        message = str(error) or type(error).__name__
        atomic_write_json(
            run_path / "error.json",
            {
                "schema": 1,
                "phase": phase,
                "exception_type": type(error).__name__,
                "message": message,
                "traceback": traceback.format_exc(),
                "time": _timestamp(),
            },
        )
        try:
            emit("error", phase=phase, message=message)
        except Exception:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    arguments = parser.parse_args(argv)
    return run_job(arguments.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
