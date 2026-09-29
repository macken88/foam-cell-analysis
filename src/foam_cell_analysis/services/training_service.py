"""学習実験、試行、キュー、設定の Qt/torch 非依存な永続化。"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from foam_cell_analysis.data.dataset_store import DatasetStore
from foam_cell_analysis.services.models import (
    ArtifactGroup,
    AugmentationProfile,
    CandidateSnapshot,
    Experiment,
    ExperimentConfig,
    ExperimentDeletionInfo,
    JobExit,
    PreparedRun,
    PruneResult,
    RunAttempt,
    TrainingOutcome,
)
from foam_cell_analysis.services.profile_store import ProfileStore
from foam_cell_analysis.training.folds import FOLD_ALGORITHM, assign_folds
from foam_cell_analysis.training.protocol import (
    atomic_write_json,
    read_events,
    read_json,
    read_run_spec,
    validate_event,
    write_run_spec,
    write_status,
)
from foam_cell_analysis.training.seeds import derive

logger = logging.getLogger(__name__)


def _process_created_at(process: dict[str, Any]) -> float | None:
    """process.json の作成時刻を Unix 秒へ変換する。"""
    for key in ("creation_time", "create_time", "created_at", "process_created_at"):
        value = process.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
    return None


def _windows_process_handle(pid: int) -> tuple[Any, float] | None:
    """PID と作成時刻を照合し、生存中の Windows プロセスハンドルを返す。"""
    if os.name != "nt":
        return None

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FileTime),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    created = FileTime()
    exit_code = ctypes.c_uint32()
    if not kernel32.GetProcessTimes(handle, ctypes.byref(created), None, None, None):
        kernel32.CloseHandle(handle)
        return None
    if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
        kernel32.CloseHandle(handle)
        return None
    windows_ticks = (created.high << 32) | created.low
    created_at = windows_ticks / 10_000_000 - 11_644_473_600
    return handle, created_at


def _default_process_alive(process: dict[str, Any]) -> bool:
    """保存 PID と作成時刻が一致する Windows プロセスだけを生存扱いする。"""
    try:
        pid = int(process["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_created = _process_created_at(process)
    if expected_created is None:
        return False
    identity = _windows_process_handle(pid)
    if identity is None:
        return False
    handle, actual_created = identity
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.CloseHandle(handle)
    return abs(expected_created - actual_created) <= 2.0


def _default_terminate_process(process: dict[str, Any]) -> None:
    """PID と作成時刻が一致するプロセスを終了して待機する。"""
    try:
        pid = int(process["pid"])
    except (KeyError, TypeError, ValueError):
        return
    expected_created = _process_created_at(process)
    identity = _windows_process_handle(pid) if expected_created is not None else None
    if identity is None:
        return
    handle, actual_created = identity
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.TerminateProcess.restype = ctypes.c_int
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    if abs(expected_created - actual_created) <= 2.0:
        kernel32.TerminateProcess(handle, 1)
        kernel32.WaitForSingleObject(handle, 5000)
    kernel32.CloseHandle(handle)


class TrainingService:
    """workspace 配下に状態を保存して復元する学習サービス。"""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        process_alive: Callable[[dict[str, Any]], bool] | None = None,
        process_terminator: Callable[[dict[str, Any]], None] | None = None,
        app_version: str = "0.1.0",
        git_commit: str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.root = self.workspace_root / "experiments"
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_store = DatasetStore(self.workspace_root)
        self.profile_store = ProfileStore(self.workspace_root)
        self.process_alive = process_alive or _default_process_alive
        self.process_terminator = process_terminator or _default_terminate_process
        self.app_version = app_version
        self.git_commit = git_commit
        self.experiments: dict[str, Experiment] = {}
        self.queue: dict[str, Any] = {"schema": 1, "retry_seq": 0, "rows": []}
        self._last_event_seq: dict[tuple[str, int], int] = {}
        self._load()

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat()

    def _queue_path(self) -> Path:
        return self.root / "queue.json"

    def _save_queue(self) -> None:
        atomic_write_json(self._queue_path(), self.queue)

    def _load(self) -> None:
        if self._queue_path().exists():
            self.queue = read_json(self._queue_path())
        for path in sorted(self.root.glob("exp_*/experiment.json")):
            record = read_json(path)
            expid = record["experiment_id"]
            config_path = path.parent / "config.yaml"
            config = (
                yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if config_path.exists()
                else {}
            )
            experiment = Experiment(
                expid,
                record.get("study_id", "foam_study"),
                record.get("description", ""),
                record["model_type"],
                ExperimentConfig(config),
                "draft" if record.get("is_draft") else "queued",
                created_at=datetime.fromisoformat(record["created_at"]),
            )
            for run_dir in sorted((path.parent / "runs").glob("attempt_*")):
                spec_file = run_dir / "run_spec.json"
                if not spec_file.exists():
                    continue
                spec = read_json(spec_file)
                status_path = run_dir / "status.json"
                status = read_json(status_path).get("status") if status_path.exists() else "running"
                experiment.runs.append(
                    RunAttempt(
                        spec["attempt"], datetime.fromisoformat(spec["created_at"]), result=status
                    )
                )
            if experiment.runs:
                experiment.status = experiment.runs[-1].result
            self.experiments[expid] = experiment

    def _write_experiment(self, experiment: Experiment, is_draft: bool) -> None:
        directory = self.root / experiment.experiment_id
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": 1,
            "experiment_id": experiment.experiment_id,
            "study_id": experiment.study_id,
            "description": experiment.description,
            "model_type": experiment.model_type,
            "created_at": experiment.created_at.isoformat(),
            "is_draft": is_draft,
        }
        atomic_write_json(directory / "experiment.json", record)
        config_path = directory / "config.yaml"
        if not config_path.exists() or not experiment.runs:
            temporary = config_path.with_name(config_path.name + ".tmp")
            temporary.write_text(
                yaml.safe_dump(experiment.config.values, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            os.replace(temporary, config_path)

    def _is_draft(self, experiment_id: str) -> bool:
        path = self.root / experiment_id / "experiment.json"
        return bool(read_json(path).get("is_draft", False)) if path.exists() else True

    def next_experiment_id(self) -> str:
        numbers = [int(key[-4:]) for key in self.experiments if key.startswith("exp_")]
        # 削除した実験の番号は再利用しない（古い比較候補などが別の実験を指さないように）
        numbers.append(int(self.queue.get("max_deleted_experiment_number", 0)))
        return f"exp_{max(numbers) + 1:04d}"

    def default_experiment_config(self, model_type: str) -> dict[str, Any]:
        """13 章に沿う現行モデル設定を作る。"""
        versions = self.dataset_store.list_versions()
        version = versions[-1] if versions else "train_v000"
        items = self.dataset_store.get_items(version) if version in versions else []
        channels = sorted({channel for item in items for channel in item.channels}) or ["A"]
        common: dict[str, Any] = {
            "experiment": {"id": None, "study_id": "foam_study", "description": ""},
            "data": {
                "dataset_version": version,
                "cv": {
                    "n_folds": 5,
                    "stratify_by_classification": True,
                    "group_by_source_folder": True,
                },
                "seed": 42,
                "classification": "all",
                "quality_filter": "all",
                "input_channels": channels[:1],
                "used_item_ids": [],
            },
            "training": {
                "epochs": 40,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "early_stopping": {"enabled": True, "patience": 10},
            },
            "augmentation": {"profile": "aug_v001"},
            "checkpoint": {
                "save_every": 10,
                "best_metric": "oof_instance_map",
                "validation_interval": 5,
                "save_fold_models": True,
            },
        }
        if model_type == "mask_rcnn":
            common["model"] = {
                "type": model_type,
                "pretrained_weights": "coco",
                "backbone": "resnet50_fpn_v2",
                "trainable_backbone_layers": 3,
                "num_classes": 2,
                "input": {
                    "min_size": 800,
                    "max_size": 1333,
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225],
                    "normalization": {
                        "method": "percentile",
                        "low_percentile": 1.0,
                        "high_percentile": 99.0,
                    },
                },
                "optimizer": "SGD",
                "anchors": {"sizes": [32, 64, 128, 256, 512], "aspect_ratios": [0.5, 1.0, 2.0]},
                "rpn": {
                    "fg_iou_thresh": 0.7,
                    "bg_iou_thresh": 0.3,
                    "batch_size_per_image": 256,
                    "positive_fraction": 0.5,
                    "pre_nms_top_n": 2000,
                    "post_nms_top_n": 1000,
                    "nms_thresh": 0.7,
                },
                "roi": {
                    "fg_iou_thresh": 0.5,
                    "bg_iou_thresh": 0.5,
                    "batch_size_per_image": 512,
                    "positive_fraction": 0.25,
                },
            }
        elif model_type == "cellpose":
            common["model"] = {
                "type": model_type,
                "pretrained_model": "cpsam",
                "bsize": 256,
                "scale_range": 0.5,
                "nimg_per_epoch": None,
                "min_train_masks": 5,
                "input_channels": channels[:1],
                "input": {
                    "normalization": {
                        "method": "percentile",
                        "low_percentile": 1.0,
                        "high_percentile": 99.0,
                    }
                },
            }
            common["training"].update({"batch_size": 1, "learning_rate": 1e-5, "weight_decay": 0.1})
        else:
            raise ValueError(f"未対応のモデル種類です: {model_type}")
        return common

    def migrate_experiment_config(self, config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """設定を現行形式へ補完し、旧 Cellpose 形式は再試行不可として識別する。"""
        result = copy.deepcopy(config)
        model = result.get("model", {})
        model_type = model.get("type", "mask_rcnn")
        legacy = model_type == "cellpose" and (
            model.get("pretrained_model") in {"cyto3", "nuclei"} or "optimizer" in model
        )
        defaults = self.default_experiment_config(model_type)
        changed = False

        def merge_missing(target: dict[str, Any], source: dict[str, Any]) -> None:
            nonlocal changed
            for key, value in source.items():
                if key not in target:
                    target[key] = copy.deepcopy(value)
                    changed = True
                elif isinstance(value, dict) and isinstance(target[key], dict):
                    merge_missing(target[key], value)

        merge_missing(result, defaults)
        data = result["data"]
        if "split_id" in data:
            del data["split_id"]
            changed = True
        checkpoint = result["checkpoint"]
        for key in ("save_best", "save_last", "best_mode"):
            changed = key in checkpoint or changed
            checkpoint.pop(key, None)
        if checkpoint.get("best_metric") != "oof_instance_map":
            checkpoint["best_metric"] = "oof_instance_map"
            changed = True
        if model_type == "cellpose" and not legacy:
            model.pop("optimizer", None)
            model.pop("class_weights", None)
            model.pop("rescale", None)
        return result, bool(changed or legacy)

    @staticmethod
    def _is_legacy_config(config: dict[str, Any]) -> bool:
        model = config.get("model", {})
        return model.get("type") == "cellpose" and (
            model.get("pretrained_model") in {"cyto3", "nuclei"} or "optimizer" in model
        )

    def validate_experiment_config(self, config: dict[str, Any]) -> list[dict[str, str]]:
        """ファイル本体を読まず、設定と metadata.csv だけを軽く確認する。"""
        issues: list[dict[str, str]] = []
        data, model = config.get("data", {}), config.get("model", {})
        model_type = model.get("type")
        if model_type not in {"mask_rcnn", "cellpose"}:
            issues.append({"level": "error", "message": "学習モデルが不正です"})
        version = data.get("dataset_version")
        try:
            items = self.dataset_store.get_items(version)
        except (OSError, ValueError, KeyError, TypeError):
            items = []
            issues.append({"level": "error", "message": "学習データセット版がありません"})
        channels = data.get("input_channels", [])
        if len(channels) != 1 or (
            items and channels[0] not in {c for i in items for c in i.channels}
        ):
            issues.append({"level": "error", "message": "学習版のチャンネルと設定が一致しません"})
        training_items = (
            self.dataset_store.select_training_items(config) if items and len(channels) == 1 else []
        )
        if len(training_items) < int(data.get("cv", {}).get("n_folds", 5)):
            issues.append({"level": "error", "message": "交差検証に必要な学習画像数がありません"})
        cv = data.get("cv", {})
        if cv.get("group_by_source_folder", True):
            group_count = len({item.source_folder for item in training_items})
            if group_count < int(cv.get("n_folds", 5)):
                issues.append(
                    {"level": "error", "message": "交差検証に必要な取込元フォルダ数がありません"}
                )
        if model_type == "cellpose":
            if importlib.util.find_spec("cellpose") is None:
                issues.append(
                    {"level": "error", "message": "Cellpose がインストールされていません"}
                )
            if model.get("pretrained_model") not in {"cpsam", "cpsam_v2"}:
                issues.append(
                    {"level": "error", "message": "Cellpose の学習版は cpsam または cpsam_v2 です"}
                )
            if model.get("bsize") != 256:
                issues.append({"level": "error", "message": "Cellpose の bsize は 256 固定です"})
            scale_range = model.get("scale_range", -1)
            if (
                type(scale_range) not in {int, float}
                or not math.isfinite(scale_range)
                or not 0 <= scale_range <= 1
            ):
                issues.append(
                    {"level": "error", "message": "scale_range は 0〜1 で指定してください"}
                )
            nimg_per_epoch = model.get("nimg_per_epoch")
            if nimg_per_epoch is not None and (
                type(nimg_per_epoch) is not int or nimg_per_epoch < 1
            ):
                issues.append(
                    {"level": "error", "message": "nimg_per_epoch は正の整数または自動です"}
                )
            min_train_masks = model.get("min_train_masks", 5)
            if type(min_train_masks) is not int or min_train_masks < 0:
                issues.append({"level": "error", "message": "min_train_masks は 0 以上の整数です"})
            batch_size = config.get("training", {}).get("batch_size")
            if type(batch_size) is not int or batch_size < 1:
                issues.append(
                    {"level": "error", "message": "Cellpose の学習 batch_size は 1 以上です"}
                )
        if model_type == "mask_rcnn" and model.get("optimizer", "SGD") not in {"SGD", "AdamW"}:
            issues.append({"level": "error", "message": "最適化手法は SGD または AdamW です"})
        if model_type == "mask_rcnn":
            if importlib.util.find_spec("torch") is None:
                issues.append({"level": "error", "message": "PyTorch がインストールされていません"})
            if (
                model.get("backbone") == "resnet101_fpn"
                and model.get("pretrained_weights") == "coco"
            ):
                issues.append(
                    {"level": "error", "message": "ResNet101 と COCO 重みは組み合わせできません"}
                )
            if len(model.get("anchors", {}).get("sizes", [])) != 5:
                issues.append({"level": "error", "message": "アンカーサイズは 5 個必要です"})
        profile = config.get("augmentation", {}).get("profile", "aug_v001")
        if profile not in self.profile_store.profiles:
            issues.append({"level": "error", "message": "拡張プロファイルがありません"})
        return issues

    def estimate_training_items(
        self,
        dataset_version: str | None = None,
        classification: str = "all",
        quality_filter: str = "all",
        n_folds: int = 5,
        **filters: Any,
    ) -> tuple[int, int]:
        """metadata.csv の件数を条件で数える。"""
        version = dataset_version or self.dataset_store.list_versions()[-1]
        config = {
            "data": {
                "dataset_version": version,
                "input_channels": filters.get("input_channels")
                or sorted(
                    {c for item in self.dataset_store.get_items(version) for c in item.channels}
                )[:1],
                "classification": classification,
                "quality_filter": quality_filter,
            }
        }
        count = len(self.dataset_store.select_training_items(config))
        folds = max(2, min(10, int(n_folds)))
        return count, (count + folds - 1) // folds

    def _experiment_from_config(
        self,
        config: dict[str, Any],
        experiment_id: str,
        status: str,
        existing: Experiment | None = None,
    ) -> Experiment:
        model_type = config["model"]["type"]
        exp = config.get("experiment", {})
        if existing:
            existing.config = ExperimentConfig(config)
            existing.status = status
            existing.description = exp.get("description", "")
            return existing
        return Experiment(
            experiment_id,
            exp.get("study_id", "foam_study"),
            exp.get("description", ""),
            model_type,
            ExperimentConfig(config),
            status,
            total_epochs=int(config.get("training", {}).get("epochs", 40)),
        )

    def save_experiment_draft(
        self, config: dict[str, Any], experiment_id: str | None = None
    ) -> Experiment:
        migrated, _ = self.migrate_experiment_config(config)
        expid = (
            experiment_id or migrated.get("experiment", {}).get("id") or self.next_experiment_id()
        )
        if self._has_attempts(expid):
            raise ValueError("試行がある実験の設定は上書きできません")
        migrated.setdefault("experiment", {})["id"] = expid
        experiment = self._experiment_from_config(
            migrated, expid, "draft", self.experiments.get(expid)
        )
        self.experiments[expid] = experiment
        self._write_experiment(experiment, True)
        return experiment

    def start_training(
        self, config: dict[str, Any], experiment_id: str | None = None
    ) -> Experiment:
        """設定を実験へ保存して実行待ちにする。試行は prepare_training_run が作る。"""
        experiment = self.save_experiment_draft(config, experiment_id)
        experiment.status = "running"
        self._write_experiment(experiment, False)
        return experiment

    def retry_experiment(self, experiment_id: str) -> Experiment:
        """既存実験を再実行待ちに戻す。試行の作成は行わない。"""
        experiment = self.experiments[experiment_id]
        config, _migrated = self.migrate_experiment_config(experiment.config.values)
        if self._is_legacy_config(config):
            raise ValueError("旧形式設定は再試行できません。設定を複製して新規実験にしてください")
        if not experiment.runs:
            raise ValueError("再試行できる学習試行がありません")
        experiment.config = ExperimentConfig(config)
        experiment.status = "running"
        self._reset_experiment_history(experiment)
        self._write_experiment(experiment, False)
        return experiment

    def record_training_process(
        self, experiment_id: str, attempt: int, pid: int, creation_time: float
    ) -> None:
        """子プロセスの PID と作成時刻を復旧用ファイルへ保存する。"""
        directory = self.root / experiment_id / "runs" / f"attempt_{attempt:03d}"
        atomic_write_json(
            directory / "process.json",
            {"pid": int(pid), "creation_time": float(creation_time)},
        )

    def fail_training_preparation(self, experiment_id: str) -> None:
        """試行ディレクトリを作る前の準備失敗を実験へ記録する。"""
        experiment = self.experiments.get(experiment_id)
        if experiment is None or experiment.runs:
            return
        experiment.status = "failed"
        self._write_experiment(experiment, False)

    def get_dataset_item_image(self, version: str, item_id: str, channel: str | None = None):
        """確定済み学習版の画像を返す。"""
        return self.dataset_store.get_image(version, item_id, channel)

    def get_dataset_item_mask(self, version: str, item_id: str, revision: str | None = None):
        """確定済み学習版の整数ラベルを返す。"""
        return self.dataset_store.get_mask(version, item_id, revision)

    def add_training_queue_item(self, config: dict[str, Any]) -> Experiment:
        migrated, _ = self.migrate_experiment_config(config)
        expid = migrated.get("experiment", {}).get("id") or self.next_experiment_id()
        existing = self.experiments.get(expid)
        # 試行のない下書きは同じ識別子のままキューへ移す
        reuse_draft = (
            existing is not None and existing.status == "draft" and not self._has_attempts(expid)
        )
        if existing is not None and not reuse_draft:
            expid = self.next_experiment_id()
        migrated.setdefault("experiment", {})["id"] = expid
        experiment = self._experiment_from_config(migrated, expid, "queued")
        self.experiments[expid] = experiment
        self._write_experiment(experiment, False)
        self.queue["rows"].append(
            {
                "queue_id": expid,
                "experiment_id": expid,
                "kind": "new",
                "state": "queued",
                "attempt": None,
            }
        )
        self._save_queue()
        return self.list_training_queue()[-1]

    def add_training_retry_reservation(self, experiment_id: str) -> Experiment:
        experiment = self.experiments[experiment_id]
        if self._is_legacy_config(experiment.config.values):
            raise ValueError("旧形式設定は再試行できません。設定を複製して新規実験にしてください")
        self.queue["retry_seq"] += 1
        queue_id = f"retry:{experiment_id}:{self.queue['retry_seq']}"
        self.queue["rows"].append(
            {
                "queue_id": queue_id,
                "experiment_id": experiment_id,
                "kind": "retry",
                "state": "queued",
                "attempt": None,
            }
        )
        self._save_queue()
        return self.list_training_queue()[-1]

    def list_training_queue(self) -> list[Experiment]:
        result = []
        for row in self.queue["rows"]:
            experiment = self.experiments.get(row["experiment_id"])
            if experiment is None:
                continue
            view = copy.copy(experiment)
            view.status = row["state"]
            view.queue_id = row["queue_id"]
            view.queue_is_retry = row["kind"] == "retry"
            view.queue_retry_attempt = row["attempt"]
            result.append(view)
        return result

    def update_training_queue_item(self, experiment_id: str, config: dict[str, Any]) -> Experiment:
        row = next((item for item in self.queue["rows"] if item["queue_id"] == experiment_id), None)
        if row is None or row["state"] != "queued" or row["kind"] != "new":
            raise ValueError("待機中の新規キュー項目のみ編集できます")
        if self._has_attempts(experiment_id):
            raise ValueError("試行がある実験の設定は上書きできません")
        migrated, _ = self.migrate_experiment_config(config)
        migrated.setdefault("experiment", {})["id"] = row["experiment_id"]
        experiment = self._experiment_from_config(
            migrated, row["experiment_id"], "queued", self.experiments[row["experiment_id"]]
        )
        self.experiments[row["experiment_id"]] = experiment
        self._write_experiment(experiment, False)
        return experiment

    def _has_attempts(self, experiment_id: str) -> bool:
        experiment = self.experiments.get(experiment_id)
        if experiment and experiment.runs:
            return True
        return any(
            (path / "run_spec.json").is_file()
            for path in (self.root / experiment_id / "runs").glob("attempt_*")
        )

    def reorder_training_queue(self, experiment_ids: list[str]) -> list[Experiment]:
        selected = [
            row
            for queue_id in experiment_ids
            for row in self.queue["rows"]
            if row["queue_id"] == queue_id
        ]
        selected_ids = {row["queue_id"] for row in selected}
        self.queue["rows"] = selected + [
            row for row in self.queue["rows"] if row["queue_id"] not in selected_ids
        ]
        self._save_queue()
        return self.list_training_queue()

    def duplicate_training_queue_items(self, experiment_ids: list[str]) -> list[Experiment]:
        result = []
        for queue_id in experiment_ids:
            row = next(item for item in self.queue["rows"] if item["queue_id"] == queue_id)
            config = copy.deepcopy(self.experiments[row["experiment_id"]].config.values)
            config["experiment"]["id"] = None
            duplicate = self.add_training_queue_item(config)
            duplicate_row = next(
                item for item in self.queue["rows"] if item["queue_id"] == duplicate.experiment_id
            )
            self.queue["rows"].remove(duplicate_row)
            self.queue["rows"].insert(
                self.queue["rows"].index(row) + 1,
                {
                    "queue_id": duplicate.experiment_id,
                    "experiment_id": duplicate.experiment_id,
                    "kind": "new",
                    "state": "queued",
                    "attempt": None,
                },
            )
            self._save_queue()
            result.append(duplicate)
        return result

    def delete_training_queue_items(self, experiment_ids: list[str]) -> None:
        removed = [
            row
            for row in self.queue["rows"]
            if row["queue_id"] in experiment_ids and row["state"] == "queued"
        ]
        self.queue["rows"] = [row for row in self.queue["rows"] if row not in removed]
        for row in removed:
            if row["kind"] == "new":
                shutil.rmtree(self.root / row["experiment_id"], ignore_errors=True)
                self.experiments.pop(row["experiment_id"], None)
        self._save_queue()

    def clear_finished_training_queue_items(self) -> None:
        self.queue["rows"] = [
            row
            for row in self.queue["rows"]
            if row["state"] not in {"completed", "failed", "stopped"}
        ]
        self._save_queue()

    def take_next_training_queue_item(self) -> Experiment | None:
        row = next(
            (
                item
                for item in self.queue["rows"]
                if item["state"] == "queued"
                and not any(
                    x["level"] == "error"
                    for x in self.validate_experiment_config(
                        self.experiments[item["experiment_id"]].config.values
                    )
                )
            ),
            None,
        )
        if row is None:
            return None
        row["state"] = "running"
        self._save_queue()
        experiment = self.experiments[row["experiment_id"]]
        experiment.status = "running"
        view = copy.copy(experiment)
        view.queue_id = row["queue_id"]
        view.queue_is_retry = row["kind"] == "retry"
        view.queue_retry_attempt = len(experiment.runs) + 1 if row["kind"] == "retry" else None
        return view

    def finish_training_queue_item(self, queue_id: str | None, status: str) -> None:
        row = next((item for item in self.queue["rows"] if item["queue_id"] == queue_id), None)
        if row:
            row["state"] = status
            if row["experiment_id"] in self.experiments:
                self.experiments[row["experiment_id"]].status = status
            self._save_queue()

    def list_augmentation_profiles(self) -> list[AugmentationProfile]:
        """保存済み拡張プロファイルを返す。"""
        return self.profile_store.list()

    def get_augmentation_profile(self, profile_id: str) -> AugmentationProfile:
        """拡張プロファイルを識別子で返す。"""
        return self.profile_store.get(profile_id)

    def save_augmentation_profile(self, profile: AugmentationProfile) -> AugmentationProfile:
        """新しい版として拡張プロファイルを保存する。"""
        return self.profile_store.save(profile)

    def _dataset_fingerprint(self, version: str) -> dict[str, str]:
        folder = self.workspace_root / "datasets" / version
        return {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("manifest.csv", "metadata.csv")
        }

    def _prepared_for(self, run_dir: Path, spec: dict[str, Any]) -> PreparedRun:
        program = str(Path(sys.executable).resolve())
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "TORCH_HOME": str(self.workspace_root / "pretrained" / "torch"),
                "CELLPOSE_LOCAL_MODELS_PATH": str(self.workspace_root / "pretrained" / "cellpose"),
            }
        )
        return PreparedRun(
            spec["run_id"],
            str(run_dir.resolve()),
            program,
            ["-m", "foam_cell_analysis.training.run", "--run-dir", str(run_dir.resolve())],
            env,
        )

    def prepare_training_run(
        self, experiment_id: str, queue_id: str | None = None, retry: bool = False
    ) -> PreparedRun:
        """run_spec を完成してから attempt 名へ移し、同じ queue 要求を冪等に返す。"""
        experiment = self.experiments[experiment_id]
        config, _migrated = self.migrate_experiment_config(experiment.config.values)
        if self._is_legacy_config(config):
            raise ValueError("旧形式設定は再試行できません。設定を複製して新規実験にしてください")
        errors = [
            item["message"]
            for item in self.validate_experiment_config(config)
            if item["level"] == "error"
        ]
        if errors:
            raise ValueError("設定エラー: " + "、".join(errors))
        row = (
            next((item for item in self.queue["rows"] if item["queue_id"] == queue_id), None)
            if queue_id
            else None
        )
        if queue_id and (row is None or row["experiment_id"] != experiment_id):
            raise ValueError("queue_id が実験に対応していません")
        if row and ((row["kind"] == "retry") != retry):
            raise ValueError("キュー種別と retry 指定が一致しません")
        runs_root = self.root / experiment_id / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        if queue_id:
            for path in sorted(runs_root.glob("attempt_*")):
                spec_file = path / "run_spec.json"
                if not spec_file.exists():
                    continue
                spec = read_json(spec_file)
                if spec.get("queue_id") == queue_id:
                    if (path / "status.json").exists() or (path / "process.json").exists():
                        raise ValueError("このキュー行の試行はすでに起動または確定しています")
                    config_path = self.root / experiment_id / "config.yaml"
                    if not config_path.exists():
                        temporary = config_path.with_name(config_path.name + ".tmp")
                        temporary.write_text(
                            yaml.safe_dump(spec["config"], allow_unicode=True, sort_keys=False),
                            encoding="utf-8",
                        )
                        os.replace(temporary, config_path)
                    if not any(item.attempt == spec["attempt"] for item in experiment.runs):
                        experiment.runs.append(
                            RunAttempt(
                                spec["attempt"],
                                datetime.fromisoformat(spec["created_at"]),
                                used_item_ids=list(spec["used_item_ids"]),
                                fold_assignments=dict(spec["fold_assignments"]),
                            )
                        )
                    experiment.config = ExperimentConfig(spec["config"])
                    experiment.used_item_ids = list(spec["used_item_ids"])
                    experiment.fold_assignments = dict(spec["fold_assignments"])
                    experiment.status = "running"
                    self._write_experiment(experiment, False)
                    if row:
                        row.update(state="running", attempt=spec["attempt"])
                        self._save_queue()
                    return self._prepared_for(path, spec)
        attempt = max((int(path.name[-3:]) for path in runs_root.glob("attempt_*")), default=0) + 1
        if attempt > 1 and not retry:
            raise ValueError("既存試行がある実験の開始には retry=True が必要です")
        if attempt == 1 and retry:
            raise ValueError("再試行の対象となる初回試行がありません")
        used_items = self.dataset_store.select_training_items(config)
        used_ids = [item.item_id for item in used_items]
        cv = config["data"].get("cv", {})
        if retry and attempt > 1:
            first_spec = read_run_spec(runs_root / "attempt_001")
            dataset = self._dataset_fingerprint(config["data"]["dataset_version"])
            if first_spec["dataset"]["sha256"] != dataset:
                raise ValueError(
                    "データセットが初回試行と異なります。設定を複製して新規実験にしてください"
                )
            fold_assignments = first_spec["fold_assignments"]
        else:
            fold_assignments = assign_folds(
                used_items,
                int(cv.get("n_folds", 5)),
                int(config["data"].get("seed", 42)),
                group_by_source_folder=cv.get("group_by_source_folder", True),
                stratify_by_classification=cv.get("stratify_by_classification", True),
            )
        profile = self.profile_store.get(config["augmentation"]["profile"])
        profile_value = ProfileStore._profile_dict(profile)
        profile_hash = hashlib.sha256(
            json.dumps(profile_value, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        run_id = f"{experiment_id}/attempt_{attempt:03d}"
        spec = {
            "schema": 1,
            "protocol": 1,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "attempt": attempt,
            "queue_id": queue_id,
            "created_at": self._now(),
            "config": config,
            "dataset": {
                "version": config["data"]["dataset_version"],
                "path": f"datasets/{config['data']['dataset_version']}",
                "sha256": self._dataset_fingerprint(config["data"]["dataset_version"]),
            },
            "used_item_ids": used_ids,
            "fold_assignments": fold_assignments,
            "fold_algorithm": FOLD_ALGORITHM,
            "augmentation_profile": {"value": profile_value, "sha256": profile_hash},
            "preprocessing": config["model"].get("input", {}).get("normalization"),
            "eval_params": (
                {
                    "channel_axis": 2,
                    "normalize": False,
                    "flow_threshold": 0.4,
                    "cellprob_threshold": 0.0,
                    "min_size": 15,
                    "max_size_fraction": 0.4,
                    "bsize": 256,
                }
                if config["model"]["type"] == "cellpose"
                else {
                    "box_score_thresh": 0.5,
                    "box_nms_thresh": 0.5,
                    "box_detections_per_img": 300,
                    "mask_thresh": 0.5,
                }
            ),
            "metric": {
                "id": "cellpose_ap_iou50_95_image_mean_v1",
                "thresholds": [round(0.5 + 0.05 * i, 2) for i in range(10)],
                "aggregation": "image_mean",
                "empty_rule": "both_empty_is_1",
            },
            "seeds": {
                "folds": {
                    str(k): derive(int(config["data"].get("seed", 42)), "fold", k)
                    for k in sorted(set(fold_assignments.values()))
                },
                "final": derive(int(config["data"].get("seed", 42)), "final"),
            },
            "app_version": self.app_version,
            "git_commit": self.git_commit,
        }
        if retry:
            for previous_dir in sorted(runs_root.glob("attempt_*")):
                resolved_path = previous_dir / "resolved_data.json"
                if resolved_path.exists():
                    expected_hash = read_json(resolved_path).get("initial_weights_sha256")
                    if expected_hash:
                        spec["expected_initial_weights_sha256"] = expected_hash
                        break
        preparing = runs_root / f".preparing_{uuid.uuid4()}"
        preparing.mkdir()
        write_run_spec(preparing, spec)
        final_dir = runs_root / f"attempt_{attempt:03d}"
        os.replace(preparing, final_dir)
        config_path = self.root / experiment_id / "config.yaml"
        if not config_path.exists():
            temporary = config_path.with_name(config_path.name + ".tmp")
            temporary.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            os.replace(temporary, config_path)
        self._write_experiment(experiment, False)
        experiment.config = ExperimentConfig(config)
        if attempt > 1:
            self._reset_experiment_history(experiment)
        experiment.used_item_ids = used_ids
        experiment.fold_assignments = fold_assignments
        experiment.runs.append(
            RunAttempt(
                attempt,
                datetime.fromisoformat(spec["created_at"]),
                used_item_ids=used_ids,
                fold_assignments=fold_assignments,
            )
        )
        experiment.status = "running"
        if row:
            row.update(state="running", attempt=attempt)
            self._save_queue()
        return self._prepared_for(final_dir, spec)

    @staticmethod
    def _reset_experiment_history(experiment: Experiment) -> None:
        experiment.history = []
        experiment.fold_histories = {}
        experiment.oof_history = []
        experiment.final_history = []
        experiment.checkpoints = []
        experiment.selected_epoch = None
        experiment.oof_evaluation = None
        experiment.oof_predictions = {}
        experiment.current_epoch = 0
        experiment.phase = "cross_validation"

    def apply_training_event(self, experiment_id: str, event: dict[str, Any]) -> Experiment:
        """seq を照合してイベントを反映し、飛びがあればイベントログから再生する。"""
        validate_event(event, allow_hello=True)
        if event["type"] == "hello":
            return self.experiments[experiment_id]
        run_id = event["run_id"]
        parts = run_id.split("/")
        if len(parts) != 2 or parts[0] != experiment_id:
            raise ValueError("イベント run_id が実験に対応していません")
        try:
            attempt = int(parts[1].removeprefix("attempt_"))
        except ValueError as error:
            raise ValueError("イベント run_id の試行番号が不正です") from error
        latest_attempt = max(
            (run.attempt for run in self.experiments[experiment_id].runs), default=0
        )
        if attempt != latest_attempt:
            logger.warning(
                "最新でない試行のイベントを無視します: experiment_id=%s attempt=%s latest=%s",
                experiment_id,
                attempt,
                latest_attempt,
            )
            return self.experiments[experiment_id]
        key = (experiment_id, attempt)
        last_seq = self._last_event_seq.get(key, 0)
        if event["seq"] <= last_seq:
            return self.experiments[experiment_id]
        if event["seq"] > last_seq + 1:
            self._replay_attempt(experiment_id, attempt)
            last_seq = self._last_event_seq.get(key, 0)
            if event["seq"] <= last_seq:
                return self.experiments[experiment_id]
            if event["seq"] != last_seq + 1:
                raise ValueError("イベント seq の欠落を events.jsonl から復元できません")
        experiment = self._apply_event_unsequenced(experiment_id, event)
        self._last_event_seq[key] = event["seq"]
        return experiment

    def _apply_event_unsequenced(self, experiment_id: str, event: dict[str, Any]) -> Experiment:
        """検証済みイベントを seq 管理なしで Experiment へ反映する。"""
        experiment = self.experiments[experiment_id]
        kind = event["type"]
        fold, epoch = event.get("fold"), event.get("epoch")
        if kind == "preflight":
            run_id = event.get("run_id", "")
            parts = run_id.split("/")
            spec_path = self.root / parts[0] / "runs" / parts[1] / "run_spec.json"
            if spec_path.exists():
                spec = read_run_spec(spec_path.parent)
                experiment.used_item_ids = list(spec["used_item_ids"])
                experiment.fold_assignments = dict(spec["fold_assignments"])
        elif kind == "phase":
            experiment.phase = "cross_validation" if event["phase"] == "cv" else "final_training"
            experiment.current_epoch = 0
        elif kind == "epoch":
            from foam_cell_analysis.services.models import EpochMetrics

            metrics = EpochMetrics(epoch, float(event["loss"]), None)
            if event["phase"] == "cv":
                experiment.fold_histories.setdefault(int(fold), []).append(metrics)
            else:
                experiment.final_history.append(metrics)
                experiment.phase = "final_training"
            experiment.current_epoch = epoch
        elif kind == "val":
            points = experiment.fold_histories.setdefault(int(fold), [])
            point = next((item for item in points if item.epoch == epoch), None)
            if point:
                point.map = float(event["ap"])
            for checkpoint in experiment.checkpoints:
                if checkpoint.fold == fold and checkpoint.epoch == epoch:
                    checkpoint.map = float(event["ap"])
        elif kind == "oof":
            from foam_cell_analysis.services.models import EpochMetrics

            fold_losses = [
                next(
                    (p.loss for p in experiment.fold_histories.get(int(k), []) if p.epoch == epoch),
                    0.0,
                )
                for k in event.get("per_fold", {})
            ]
            loss = sum(fold_losses) / len(fold_losses) if fold_losses else 0.0
            experiment.oof_history.append(EpochMetrics(epoch, loss, float(event["ap"])))
            experiment.history = experiment.oof_history
        elif kind == "selected":
            from foam_cell_analysis.services.models import Evaluation

            experiment.selected_epoch = epoch
            experiment.oof_evaluation = Evaluation(
                float(event["ap"]),
                {
                    key: (float(value[0]) if value[0] is not None else None, int(value[1]))
                    for key, value in event["per_class"].items()
                },
            )
        elif kind == "checkpoint":
            from foam_cell_analysis.services.models import Checkpoint

            name = Path(event["path"]).name
            score = (
                next(
                    (
                        p.map
                        for p in experiment.fold_histories.get(int(fold or 0), [])
                        if p.epoch == epoch
                    ),
                    None,
                )
                if fold
                else None
            )
            experiment.checkpoints.append(Checkpoint(name, epoch, score, fold=fold))
            experiment.current_epoch = epoch
            if fold is None:
                experiment.phase = "final_training"
        self._capture_event_run(experiment, event.get("run_id"))
        return experiment

    @staticmethod
    def _capture_event_run(experiment: Experiment, run_id: str | None = None) -> None:
        if not experiment.runs:
            return
        attempt = None
        if run_id and "/attempt_" in run_id:
            attempt = int(run_id.rsplit("attempt_", maxsplit=1)[1])
        run = next(
            (item for item in experiment.runs if item.attempt == attempt),
            experiment.runs[-1],
        )
        run.history = copy.deepcopy(experiment.history)
        run.fold_histories = copy.deepcopy(experiment.fold_histories)
        run.oof_history = copy.deepcopy(experiment.oof_history)
        run.final_history = copy.deepcopy(experiment.final_history)
        run.checkpoints = copy.deepcopy(experiment.checkpoints)
        run.selected_epoch = experiment.selected_epoch
        run.oof_evaluation = copy.deepcopy(experiment.oof_evaluation)
        run.oof_predictions = copy.deepcopy(experiment.oof_predictions)
        run.current_epoch = experiment.current_epoch
        run.phase = experiment.phase
        run.used_item_ids = list(experiment.used_item_ids)
        run.fold_assignments = dict(experiment.fold_assignments)

    def request_training_stop(self, experiment_id: str, attempt: int, reason: str) -> None:
        """stop_request.json を原子的に保存する。"""
        if reason not in {"user_stop", "app_exit"}:
            raise ValueError("中断理由は user_stop または app_exit です")
        directory = self.root / experiment_id / "runs" / f"attempt_{attempt:03d}"
        atomic_write_json(
            directory / "stop_request.json", {"reason": reason, "requested_at": self._now()}
        )

    def _valid_result(self, run_dir: Path) -> bool:
        try:
            result = read_json(run_dir / "result.json")
            selected_epoch = result.get("selected_epoch")
            oof = result.get("oof", result.get("oof_evaluation"))
            metric = result.get("metric")
            artifacts = result.get("artifacts")
            if type(selected_epoch) is not int or selected_epoch < 1:
                return False
            if (
                not isinstance(oof, dict)
                or not isinstance(oof.get("ap"), (int, float))
                or not isinstance(oof.get("per_class"), dict)
                or type(oof.get("n_images")) is not int
            ):
                return False
            if not isinstance(metric, dict) or not {
                "id",
                "thresholds",
                "aggregation",
                "empty_rule",
            }.issubset(metric):
                return False
            if not isinstance(artifacts, list) or not artifacts:
                return False
            root = run_dir.resolve()
            final_found = False
            checkpoint_artifacts: dict[str, dict[str, Any]] = {}
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    return False
                relative = artifact.get("path")
                size = artifact.get("size")
                sha256 = artifact.get("sha256")
                if not isinstance(relative, str) or type(size) is not int or size < 0:
                    return False
                if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
                    return False
                posix_path = PurePosixPath(relative)
                windows_path = PureWindowsPath(relative)
                if (
                    posix_path.is_absolute()
                    or windows_path.is_absolute()
                    or windows_path.drive
                    or ".." in posix_path.parts
                    or ".." in windows_path.parts
                ):
                    return False
                path = (root / Path(*posix_path.parts)).resolve()
                try:
                    path.relative_to(root)
                except ValueError:
                    return False
                if not path.is_file() or path.stat().st_size != size:
                    return False
                if windows_path.as_posix() == "checkpoints/final.pt":
                    final_found = True
                if windows_path.suffix.lower() == ".pt":
                    checkpoint_artifacts[windows_path.as_posix()] = artifact
            if not final_found:
                return False
            spec = read_run_spec(run_dir)
            for relative, artifact in checkpoint_artifacts.items():
                checkpoint_path = root / Path(*PurePosixPath(relative).parts)
                sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
                if not sidecar.is_file():
                    return False
                metadata = read_json(sidecar)
                if not isinstance(metadata, dict) or not {
                    "model_type",
                    "fold",
                    "epoch",
                    "kind",
                    "run_id",
                    "sha256",
                }.issubset(metadata):
                    return False
                if (
                    metadata["model_type"] != spec.get("config", {}).get("model", {}).get("type")
                    or metadata["run_id"] != spec.get("run_id")
                    or type(metadata["epoch"]) is not int
                    or metadata["epoch"] < 1
                    or not isinstance(metadata["kind"], str)
                    or not isinstance(metadata["sha256"], str)
                    or re.fullmatch(r"[0-9a-fA-F]{64}", metadata["sha256"]) is None
                    or metadata["sha256"].lower() != artifact["sha256"].lower()
                ):
                    return False
                if metadata["kind"] == "final":
                    if metadata["fold"] is not None or relative != "checkpoints/final.pt":
                        return False
                elif metadata["kind"] in {"periodic", "selected"}:
                    if type(metadata["fold"]) is not int or metadata["fold"] < 1:
                        return False
                else:
                    return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _conclude(
        self, experiment_id: str, attempt: int, job_exit: JobExit | None = None
    ) -> TrainingOutcome:
        directory = self.root / experiment_id / "runs" / f"attempt_{attempt:03d}"
        spec = read_run_spec(directory)
        status_path = directory / "status.json"
        had_status = status_path.exists()
        if had_status:
            state = read_json(status_path)
            outcome = TrainingOutcome(
                experiment_id,
                attempt,
                spec.get("queue_id"),
                state["status"],
                state.get("reason"),
                state.get("message", ""),
            )
        elif (directory / "stop_request.json").exists():
            request = read_json(directory / "stop_request.json")
            outcome = TrainingOutcome(
                experiment_id, attempt, spec.get("queue_id"), "stopped", request["reason"]
            )
        elif self._valid_result(directory):
            outcome = TrainingOutcome(experiment_id, attempt, spec.get("queue_id"), "completed")
        elif (directory / "error.json").exists() or (job_exit and job_exit.start_failed):
            reason = "start_failed" if job_exit and job_exit.start_failed else "error"
            message = (
                job_exit.message
                if reason == "start_failed"
                else read_json(directory / "error.json").get("message", "")
            )
            outcome = TrainingOutcome(
                experiment_id, attempt, spec.get("queue_id"), "failed", reason, message
            )
        elif not (job_exit and job_exit.process_alive) and not self.process_alive(
            read_json(directory / "process.json") if (directory / "process.json").exists() else {}
        ):
            outcome = TrainingOutcome(
                experiment_id, attempt, spec.get("queue_id"), "stopped", "interrupted"
            )
        else:
            outcome = TrainingOutcome(experiment_id, attempt, spec.get("queue_id"), "running")
            return outcome
        if not had_status:
            write_status(
                directory,
                {
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "message": outcome.message,
                    "concluded_at": self._now(),
                    "returncode": job_exit.returncode if job_exit else None,
                },
            )
        experiment = self.experiments[experiment_id]
        run = next((item for item in experiment.runs if item.attempt == attempt), None)
        if run:
            run.result = outcome.status
            run.finished_at = datetime.now().astimezone()
        latest_attempt = max((item.attempt for item in experiment.runs), default=attempt)
        if attempt == latest_attempt:
            experiment.status = outcome.status
        if outcome.queue_id:
            row = next(
                (item for item in self.queue["rows"] if item["queue_id"] == outcome.queue_id), None
            )
            if row:
                row.update(state=outcome.status, attempt=attempt)
                self._save_queue()
        return outcome

    def conclude_training_run(
        self, experiment_id: str, attempt: int, job_exit: JobExit | None = None
    ) -> TrainingOutcome:
        return self._conclude(experiment_id, attempt, job_exit)

    def recover(self) -> list[TrainingOutcome]:
        """起動復旧を行い、キュー行・一時領域・試行状態を照合する。"""
        outcomes = []
        specs_by_queue: dict[str, tuple[int, str]] = {}
        for spec_path in sorted(self.root.glob("exp_*/runs/attempt_*/run_spec.json")):
            run_dir = spec_path.parent
            experiment_id = run_dir.parents[1].name
            spec = read_json(spec_path)
            attempt = int(spec["attempt"])
            if not (run_dir / "status.json").exists():
                process_file = run_dir / "process.json"
                if process_file.exists():
                    process = read_json(process_file)
                    if self.process_alive(process):
                        self.process_terminator(process)
                        deadline = time.monotonic() + 5.0
                        while self.process_alive(process) and time.monotonic() < deadline:
                            time.sleep(0.1)
                        if self.process_alive(process):
                            raise RuntimeError(
                                f"学習プロセスを終了できませんでした: {spec['run_id']}"
                            )
                outcome = self._conclude(experiment_id, attempt)
                outcomes.append(outcome)
            if spec.get("queue_id"):
                status_file = run_dir / "status.json"
                specs_by_queue[spec["queue_id"]] = (
                    attempt,
                    read_json(status_file).get("status", "running")
                    if status_file.exists()
                    else "running",
                )
        for row in self.queue.get("rows", []):
            found = specs_by_queue.get(row["queue_id"])
            if found:
                row["attempt"], row["state"] = found
            elif row["state"] == "running":
                row.update(state="queued", attempt=None)
        for pruned_path in sorted(self.root.glob("exp_*/runs/attempt_*/pruned.json")):
            try:
                self._finish_pruning(pruned_path.parent)
            except (OSError, ValueError):
                logger.exception("成果物の整理を再開できませんでした: %s", pruned_path)
        for path in self.root.glob("exp_*/runs/.preparing_*"):
            shutil.rmtree(path, ignore_errors=True)
        for storage_root in (self.root, self.workspace_root / "augmentation"):
            if storage_root.exists():
                for path in storage_root.rglob("*.tmp"):
                    path.unlink(missing_ok=True)
        self._save_queue()
        self._replay_history()
        return outcomes

    def _replay_attempt(self, experiment_id: str, attempt: int) -> None:
        """指定試行のイベントと結果ファイルから表示用履歴を再構築する。"""
        experiment = self.experiments[experiment_id]
        run_dir = self.root / experiment_id / "runs" / f"attempt_{attempt:03d}"
        run_id = read_run_spec(run_dir).get("run_id")
        latest_attempt = max((run.attempt for run in experiment.runs), default=attempt)
        state_fields = (
            "current_epoch",
            "history",
            "fold_histories",
            "oof_history",
            "final_history",
            "checkpoints",
            "selected_epoch",
            "oof_evaluation",
            "oof_predictions",
            "phase",
            "used_item_ids",
            "fold_assignments",
        )
        previous = {name: copy.deepcopy(getattr(experiment, name)) for name in state_fields}
        self._reset_experiment_history(experiment)
        spec = read_run_spec(run_dir)
        experiment.used_item_ids = list(spec["used_item_ids"])
        experiment.fold_assignments = dict(spec["fold_assignments"])
        if run_dir.joinpath("events.jsonl").exists():
            for event in read_events(run_dir / "events.jsonl", expected_run_id=run_id):
                self._apply_event_unsequenced(experiment_id, event)
                self._last_event_seq[(experiment_id, attempt)] = event["seq"]
        for name in ("result.partial.json", "result.json"):
            result_path = run_dir / name
            if not result_path.exists():
                continue
            result = read_json(result_path)
            per_image = result.get("per_image", result.get("oof_predictions", {}))
            if isinstance(per_image, dict):
                experiment.oof_predictions = {
                    str(key): float(value)
                    for key, value in per_image.items()
                    if isinstance(value, (int, float))
                }
            oof = result.get("oof", result.get("oof_evaluation"))
            if isinstance(oof, dict) and oof.get("ap") is not None:
                from foam_cell_analysis.services.models import Evaluation

                per_class = {}
                for key, value in oof.get("per_class", {}).items():
                    if isinstance(value, dict):
                        score = value.get("ap")
                        count = value.get("n_images", 0)
                    elif isinstance(value, list | tuple) and len(value) == 2:
                        score, count = value
                    else:
                        continue
                    if score is not None:
                        per_class[str(key)] = (float(score), int(count))
                experiment.oof_evaluation = Evaluation(float(oof["ap"]), per_class)
            if result.get("selected_epoch") is not None:
                experiment.selected_epoch = int(result["selected_epoch"])
        self._capture_event_run(experiment, run_id)
        if attempt != latest_attempt:
            for name, value in previous.items():
                setattr(experiment, name, value)

    def _replay_history(self) -> None:
        """イベントと結果ファイルから全試行の表示用履歴を再構築する。"""
        self._last_event_seq.clear()
        for spec_path in sorted(self.root.glob("exp_*/runs/attempt_*/run_spec.json")):
            run_dir = spec_path.parent
            experiment_id = run_dir.parents[1].name
            if experiment_id in self.experiments:
                self._replay_attempt(experiment_id, int(run_dir.name[-3:]))

    def create_candidate_snapshot(
        self, experiment_id: str, *, attempt: int | None = None
    ) -> CandidateSnapshot:
        """完了した試行の final.pt・実測 OOF・設定を固定して束ねる。

        attempt を省くと最新の完了試行を使う（従来の呼び出し元との互換のため）。
        """
        experiment = self.experiments[experiment_id]
        if attempt is None:
            completed_runs = [run for run in experiment.runs if run.result == "completed"]
            attempt = max((run.attempt for run in completed_runs), default=0)
            if attempt < 1:
                raise ValueError("完了した試行がありません")
            if experiment.status != "completed":
                raise ValueError("完了した実験のみ比較候補へ送れます")
        else:
            run = next((item for item in experiment.runs if item.attempt == attempt), None)
            if run is None:
                raise ValueError(f"試行 {attempt} がありません")
            if run.result != "completed":
                raise ValueError(f"試行 {attempt} は完了していません")
        run_id = f"{experiment_id}/attempt_{attempt:03d}"
        run_dir = self._experiment_dir(experiment_id) / "runs" / f"attempt_{attempt:03d}"
        status_path = run_dir / "status.json"
        if not status_path.is_file() or read_json(status_path).get("status") != "completed":
            raise ValueError(f"試行 {attempt} は完了していません")
        checkpoint = "checkpoints/final.pt"
        if checkpoint in self.pruned_paths(experiment_id, attempt):
            raise ValueError("最終学習モデルは成果物の整理で削除されています")
        try:
            result = read_json(run_dir / "result.json")
        except (OSError, ValueError) as error:
            raise ValueError("完了試行の result.json を読めません") from error
        selected_epoch = result.get("selected_epoch")
        if type(selected_epoch) is not int or selected_epoch < 1:
            raise ValueError("完了試行の result.json に選択エポックがありません")
        oof = result.get("oof")
        if not isinstance(oof, dict) or oof.get("ap") is None:
            raise ValueError("完了試行に実測 OOF 評価がありません")
        artifact = next(
            (
                item
                for item in result.get("artifacts") or []
                if isinstance(item, dict) and item.get("path") == checkpoint
            ),
            None,
        )
        if (
            artifact is None
            or type(artifact.get("size")) is not int
            or not isinstance(artifact.get("sha256"), str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", artifact["sha256"]) is None
        ):
            raise ValueError("完了試行の result.json に final.pt の記録がありません")
        weights = run_dir / checkpoint
        if not self._inside_run_dir(run_dir, weights):
            raise ValueError("完了試行に final.pt がありません")
        if weights.stat().st_size != artifact["size"]:
            raise ValueError("final.pt の大きさが result.json の記録と一致しません")
        spec = read_run_spec(run_dir)
        preprocessing = spec.get("preprocessing")
        dataset = spec.get("dataset")
        return CandidateSnapshot(
            experiment_id=experiment_id,
            attempt=attempt,
            selected_epoch=selected_epoch,
            run_id=run_id,
            checkpoint_path=checkpoint,
            oof_evaluation=copy.deepcopy(oof),
            experiment_config=copy.deepcopy(spec.get("config") or experiment.config.values),
            weights_size=artifact["size"],
            weights_sha256=artifact["sha256"].lower(),
            # 学習時の評価パラメータがない古い試行は空のまま返し、補完は比較側が記録付きで行う
            training_eval_params=copy.deepcopy(spec.get("eval_params") or {}),
            preprocessing=copy.deepcopy(preprocessing) if isinstance(preprocessing, dict) else {},
            training_dataset=copy.deepcopy(dataset) if isinstance(dataset, dict) else {},
        )

    # ---- 成果物の整理（比較・評価設計 19 章） ----

    ARTIFACT_CATEGORIES = ("fold_periodic", "fold_selected", "final", "temporary")

    @staticmethod
    def _inside_run_dir(run_dir: Path, path: Path) -> bool:
        """リンクをたどらずに run_dir の中の実在するファイルかを確かめる。"""
        try:
            relative = path.relative_to(run_dir)
        except ValueError:
            return False
        if ".." in relative.parts:
            return False
        probe = run_dir
        for part in relative.parts:
            probe = probe / part
            if probe.is_symlink() or probe.is_junction():
                return False
        if not path.is_file():
            return False
        try:
            path.resolve().relative_to(run_dir.resolve())
        except (OSError, ValueError):
            return False
        return True

    @staticmethod
    def _safe_relative(run_dir: Path, relative: Any) -> Path | None:
        """記録された相対パスを検査し、run_dir の中のパスへ変換する。"""
        if not isinstance(relative, str) or not relative:
            return None
        posix_path = PurePosixPath(relative)
        windows_path = PureWindowsPath(relative)
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in posix_path.parts
            or ".." in windows_path.parts
        ):
            return None
        return run_dir / Path(*windows_path.parts)

    def _artifact_files(self, run_dir: Path, category: str) -> list[Path]:
        """種類に当たる run_dir 内の実在ファイルを返す（*.pt.json は含めない）。"""
        if category == "fold_periodic":
            candidates = sorted(run_dir.glob("checkpoints/fold_*/epoch_*.pt"))
        elif category == "fold_selected":
            candidates = sorted(run_dir.glob("checkpoints/fold_*/selected.pt"))
        elif category == "final":
            candidates = [run_dir / "checkpoints" / "final.pt"]
        elif category == "temporary":
            candidates = []
            for name in ("tmp_pred", "tmp_ckpt"):
                top = run_dir / name
                if top.is_symlink() or top.is_junction() or not top.is_dir():
                    continue
                for directory, _dirnames, filenames in os.walk(top, followlinks=False):
                    candidates.extend(Path(directory) / filename for filename in filenames)
            candidates.sort()
        else:
            raise ValueError(f"成果物の種類が不正です: {category}")
        return [path for path in candidates if self._inside_run_dir(run_dir, path)]

    def _attempt_dirs(self, experiment_id: str) -> list[tuple[int, Path]]:
        """run_spec.json のある試行フォルダを試行番号順に返す。"""
        runs_root = self._experiment_dir(experiment_id) / "runs"
        result = []
        for path in sorted(runs_root.glob("attempt_*")):
            match = re.fullmatch(r"attempt_(\d+)", path.name)
            if (
                match is None
                or path.is_symlink()
                or path.is_junction()
                or not (path / "run_spec.json").is_file()
            ):
                continue
            result.append((int(match.group(1)), path))
        return result

    def _experiment_busy_reason(self, experiment_id: str) -> str:
        """学習中・待機中の実験なら整理できない理由を返す。"""
        experiment = self.experiments.get(experiment_id)
        states = {
            row["state"] for row in self.queue["rows"] if row["experiment_id"] == experiment_id
        }
        if (experiment is not None and experiment.status == "running") or "running" in states:
            return "学習中の実験の成果物は整理できません"
        if (experiment is not None and experiment.status == "queued") or "queued" in states:
            return "学習キューで待機中の実験の成果物は整理できません"
        return ""

    def artifact_cleanup_plan(
        self,
        experiment_ids: list[str],
        *,
        protected_final: Callable[[str, int], str] | None = None,
    ) -> list[ArtifactGroup]:
        """試行・種類ごとに、消せるファイルの数・容量と可否を返す。"""
        groups = []
        for experiment_id in experiment_ids:
            if experiment_id not in self.experiments:
                raise ValueError(f"実験がありません: {experiment_id}")
            busy = self._experiment_busy_reason(experiment_id)
            for attempt, run_dir in self._attempt_dirs(experiment_id):
                status_path = run_dir / "status.json"
                status = read_json(status_path).get("status") if status_path.is_file() else None
                for category in self.ARTIFACT_CATEGORIES:
                    files = self._artifact_files(run_dir, category)
                    if not files:
                        continue
                    reason = busy
                    if not reason and status is None:
                        reason = "試行が確定していないため整理できません"
                    if (
                        not reason
                        and category == "temporary"
                        and status not in {"stopped", "failed"}
                    ):
                        reason = "一時ファイルは中断・失敗した試行だけ整理できます"
                    if not reason and category == "final" and protected_final is not None:
                        reason = protected_final(experiment_id, attempt) or ""
                    groups.append(
                        ArtifactGroup(
                            experiment_id,
                            attempt,
                            category,
                            len(files),
                            sum(path.stat().st_size for path in files),
                            not reason,
                            reason,
                        )
                    )
        return groups

    @staticmethod
    def _read_pruned(run_dir: Path) -> dict[str, Any]:
        path = run_dir / "pruned.json"
        if not path.is_file():
            return {"schema": 1, "entries": []}
        record = read_json(path)
        if not isinstance(record.get("entries"), list):
            record["entries"] = []
        return record

    def pruned_paths(self, experiment_id: str, attempt: int) -> set[str]:
        """成果物の整理で削除済み（または削除中）の run_dir 相対パスを返す。"""
        run_dir = self._experiment_dir(experiment_id) / "runs" / f"attempt_{attempt:03d}"
        try:
            record = self._read_pruned(run_dir)
        except (OSError, ValueError):
            logger.warning("pruned.json を読めません: %s", run_dir)
            return set()
        return {
            entry["path"]
            for entry in record["entries"]
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }

    @staticmethod
    def _recorded_sha256(run_dir: Path, relative: str, path: Path) -> str | None:
        """result.json の成果物一覧か、横のメタ情報から sha256 を探す。"""
        try:
            result = read_json(run_dir / "result.json")
            for artifact in result.get("artifacts") or []:
                if isinstance(artifact, dict) and artifact.get("path") == relative:
                    value = artifact.get("sha256")
                    if isinstance(value, str):
                        return value.lower()
        except (OSError, ValueError):
            pass
        sidecar = path.with_name(path.name + ".json")
        try:
            value = read_json(sidecar).get("sha256") if sidecar.is_file() else None
        except (OSError, ValueError):
            value = None
        return value.lower() if isinstance(value, str) else None

    @staticmethod
    def _unlink(path: Path) -> int | None:
        """ファイルを消し、消した大きさを返す。すでにないときは None。"""
        for trial in range(5):
            try:
                size = path.stat().st_size
                path.unlink()
                return size
            except FileNotFoundError:
                return None
            except PermissionError:
                if trial == 4:
                    raise
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
                time.sleep(0.1)
        return None

    def _finish_pruning(self, run_dir: Path) -> tuple[int, int]:
        """pruned.json の deleting 項目のファイルを消し、deleted にする。"""
        record = self._read_pruned(run_dir)
        pending = [
            entry
            for entry in record["entries"]
            if isinstance(entry, dict) and entry.get("state") == "deleting"
        ]
        freed = count = 0
        for entry in pending:
            path = self._safe_relative(run_dir, entry.get("path"))
            if path is None:
                logger.warning("pruned.json の不正なパスを無視します: %s", entry.get("path"))
                continue
            if self._inside_run_dir(run_dir, path):
                size = self._unlink(path)
                if size is not None:
                    freed += size
                    count += 1
            elif path.exists() or path.is_symlink():
                # リンクや run_dir の外を指すものは消さず、削除中のまま残す
                logger.warning("run_dir の外を指すため削除しません: %s", path)
                continue
            entry["state"] = "deleted"
            entry["deleted_at"] = self._now()
        if pending:
            atomic_write_json(run_dir / "pruned.json", record)
        if not any(entry.get("category") == "temporary" for entry in pending):
            return freed, count
        # 空になった一時フォルダを片付ける（中身が残るフォルダは消さない）
        for name in ("tmp_pred", "tmp_ckpt"):
            top = run_dir / name
            if top.is_dir() and not (top.is_symlink() or top.is_junction()):
                for directory, _dirnames, _filenames in os.walk(top, topdown=False):
                    try:
                        Path(directory).rmdir()
                    except OSError:
                        pass
        return freed, count

    def prune_artifacts(
        self,
        experiment_ids: list[str],
        categories: list[str],
        *,
        protected_final: Callable[[str, int], str] | None = None,
    ) -> PruneResult:
        """選んだ種類の成果物を消す。条件を満たさないまとまりは消さずに skipped へ入れる。"""
        unknown = [item for item in categories if item not in self.ARTIFACT_CATEGORIES]
        if unknown:
            raise ValueError(f"成果物の種類が不正です: {', '.join(unknown)}")
        plan = self.artifact_cleanup_plan(experiment_ids, protected_final=protected_final)
        selected = [group for group in plan if group.category in categories]
        skipped = [group for group in selected if not group.deletable]
        targets: dict[tuple[str, int], list[str]] = {}
        for group in selected:
            if group.deletable:
                targets.setdefault((group.experiment_id, group.attempt), []).append(group.category)
        freed = count = 0
        for (experiment_id, attempt), attempt_categories in targets.items():
            run_dir = self._experiment_dir(experiment_id) / "runs" / f"attempt_{attempt:03d}"
            record = self._read_pruned(run_dir)
            by_path = {
                entry.get("path"): entry for entry in record["entries"] if isinstance(entry, dict)
            }
            requested_at = self._now()
            for category in attempt_categories:
                for path in self._artifact_files(run_dir, category):
                    relative = path.relative_to(run_dir).as_posix()
                    entry = {
                        "path": relative,
                        "category": category,
                        "size": path.stat().st_size,
                        "sha256": self._recorded_sha256(run_dir, relative, path),
                        "state": "deleting",
                        "requested_at": requested_at,
                        "deleted_at": None,
                    }
                    if relative in by_path:
                        by_path[relative].update(entry)
                    else:
                        record["entries"].append(entry)
                        by_path[relative] = entry
            atomic_write_json(run_dir / "pruned.json", record)
            attempt_freed, attempt_count = self._finish_pruning(run_dir)
            freed += attempt_freed
            count += attempt_count
        return PruneResult(freed, count, skipped)

    def _experiment_dir(self, experiment_id: str) -> Path:
        """実験フォルダを返す。experiments 配下の直下以外は扱わない。"""
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", experiment_id) or experiment_id in {".", ".."}:
            raise ValueError(f"実験識別子が不正です: {experiment_id}")
        directory = (self.root / experiment_id).resolve()
        if directory.parent != self.root.resolve():
            raise ValueError(f"実験識別子が不正です: {experiment_id}")
        return directory

    def experiment_deletion_info(
        self, experiment_id: str, *, measure_size: bool = True
    ) -> ExperimentDeletionInfo:
        """削除の可否と、削除で消える試行数・容量を返す。"""
        experiment = self.experiments[experiment_id]
        attempts = len(experiment.runs)
        reason = ""
        running_row = any(
            row["experiment_id"] == experiment_id and row["state"] == "running"
            for row in self.queue["rows"]
        )
        if experiment.status == "running" or running_row:
            reason = "学習中の実験は削除できません。学習を停止してから削除してください"
        elif experiment.status == "queued":
            reason = "キューで待機中の実験は、学習キューの表から削除してください"
        elif experiment.status not in {"draft", "stopped", "failed", "completed"}:
            reason = "この状態の実験は削除できません"
        size = None
        if measure_size:
            directory = self._experiment_dir(experiment_id)
            size = (
                sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
                if directory.is_dir()
                else 0
            )
        return ExperimentDeletionInfo(experiment_id, not reason, reason, attempts, size)

    def delete_experiment(self, experiment_id: str) -> None:
        """実験フォルダ全体と、その実験のキュー行を削除する。"""
        info = self.experiment_deletion_info(experiment_id, measure_size=False)
        if not info.allowed:
            raise ValueError(info.reason)
        directory = self._experiment_dir(experiment_id)

        def make_writable_and_retry(function, path, _error):
            os.chmod(path, 0o700)
            function(path)

        if directory.exists():
            shutil.rmtree(directory, onexc=make_writable_and_retry)
        self.experiments.pop(experiment_id, None)
        if experiment_id.startswith("exp_") and experiment_id[-4:].isdigit():
            self.queue["max_deleted_experiment_number"] = max(
                int(self.queue.get("max_deleted_experiment_number", 0)), int(experiment_id[-4:])
            )
        self.queue["rows"] = [
            row for row in self.queue["rows"] if row["experiment_id"] != experiment_id
        ]
        self._save_queue()
        for key in [key for key in self._last_event_seq if key[0] == experiment_id]:
            del self._last_event_seq[key]

    def list_experiments(self) -> list[Experiment]:
        return [self.experiments[key] for key in sorted(self.experiments)]

    def get_experiment(self, experiment_id: str) -> Experiment:
        return self.experiments[experiment_id]
