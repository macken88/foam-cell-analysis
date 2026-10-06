"""学習実験、試行、キュー、設定の Qt/torch 非依存な永続化。"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
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
from foam_cell_analysis.jobs.lifecycle import (
    decide_terminal_state,
    process_alive,
    process_created_at,
    settle_process,
    settle_unrecorded_worker,
    terminate_process,
    windows_process_handle,
)
from foam_cell_analysis.jobs.protocol import classify_seq
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
from foam_cell_analysis.training.config_rules import validate_numeric_config
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
from foam_cell_analysis.utils.backend_contracts import training_eval_params

logger = logging.getLogger(__name__)


# 旧名の互換（プロセス照合は jobs/lifecycle.py へ移した）
_process_created_at = process_created_at
_windows_process_handle = windows_process_handle
_default_process_alive = process_alive
_default_terminate_process = terminate_process


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
        self.recovery_issues: dict[str, tuple[str, str]] = {}
        self.recovery_blockers: list[str] = []
        self._queue_broken = False
        self.recovery_issues.update(
            {
                f"profile:{profile_id}": ("unrecoverable", reason)
                for profile_id, reason in self.profile_store.recovery_issues.items()
            }
        )
        self.queue: dict[str, Any] = {"schema": 1, "retry_seq": 0, "rows": []}
        self._last_event_seq: dict[tuple[str, int], int] = {}
        self._load()

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat()

    def _queue_path(self) -> Path:
        return self.root / "queue.json"

    def _guard_experiment(self, experiment_id: str) -> None:
        issue = self.recovery_issues.get(experiment_id)
        if issue is not None:
            raise ValueError(issue[1])

    def _guard_queue(self) -> None:
        if self._queue_broken:
            raise ValueError("学習キューの記録が破損しているため変更できません")

    def _save_queue(self) -> None:
        self._guard_queue()
        atomic_write_json(self._queue_path(), self.queue)

    def _load(self) -> None:
        if self._queue_path().exists():
            try:
                loaded_queue = read_json(self._queue_path())
                if (
                    not isinstance(loaded_queue, dict)
                    or not isinstance(loaded_queue.get("rows"), list)
                    or any(
                        not isinstance(row, dict)
                        or not isinstance(row.get("queue_id"), str)
                        or not row.get("queue_id")
                        or not isinstance(row.get("experiment_id"), str)
                        or not row.get("experiment_id")
                        or re.fullmatch(r"exp_\d{4,}", row.get("experiment_id", "")) is None
                        or not isinstance(row.get("state"), str)
                        or row.get("state")
                        not in {"queued", "running", "completed", "failed", "stopped"}
                        or row.get("kind") not in {"new", "retry"}
                        or not (
                            row.get("attempt") is None
                            or (type(row.get("attempt")) is int and row["attempt"] > 0)
                        )
                        for row in loaded_queue["rows"]
                    )
                ):
                    raise ValueError("キュー形式が不正です")
                self.queue = loaded_queue
            except (OSError, ValueError, TypeError) as error:
                self._queue_broken = True
                self.recovery_issues["queue"] = ("unrecoverable", "学習キューを読めません")
                logger.warning("学習キューを読めません: %s", error)
        for path in sorted(self.root.glob("exp_*/experiment.json")):
            expid = path.parent.name
            experiment = None
            try:
                record = read_json(path)
                if not isinstance(record, dict):
                    raise ValueError("実験記録の形式が不正です")
                if record.get("experiment_id") != expid:
                    raise ValueError("実験 ID がフォルダ名と一致しません")
                if not isinstance(record.get("model_type"), str):
                    raise ValueError("モデル種類の形式が不正です")
                if not isinstance(record.get("study_id", "foam_study"), str) or not isinstance(
                    record.get("description", ""), str
                ):
                    raise ValueError("実験の表示項目の形式が不正です")
                config_path = path.parent / "config.yaml"
                config = (
                    yaml.safe_load(config_path.read_text(encoding="utf-8"))
                    if config_path.exists()
                    else None
                )
                if config is not None and not isinstance(config, dict):
                    raise ValueError("config.yaml の形式が不正です")
                experiment = Experiment(
                    expid,
                    record.get("study_id", "foam_study"),
                    record.get("description", ""),
                    record["model_type"],
                    ExperimentConfig(config or {}),
                    "draft" if record.get("is_draft") else "queued",
                    created_at=datetime.fromisoformat(record["created_at"]),
                )
                experiment.total_epochs = self._total_epochs(config)
                for run_dir in sorted((path.parent / "runs").glob("attempt_*")):
                    spec_file = run_dir / "run_spec.json"
                    if not spec_file.exists():
                        continue
                    spec = read_json(spec_file)
                    if not experiment.config.values and isinstance(spec.get("config"), dict):
                        experiment.config = ExperimentConfig(spec["config"])
                        experiment.total_epochs = self._total_epochs(spec["config"])
                    status_path = run_dir / "status.json"
                    status_record = read_json(status_path) if status_path.exists() else None
                    status = status_record.get("status") if status_record is not None else "running"
                    if not isinstance(status, str):
                        raise ValueError("試行状態の形式が不正です")
                    experiment.runs.append(
                        RunAttempt(
                            spec["attempt"],
                            datetime.fromisoformat(spec["created_at"]),
                            result=status,
                        )
                    )
                if experiment.runs:
                    experiment.status = experiment.runs[-1].result
                self.experiments[expid] = experiment
            except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
                self.recovery_issues[expid] = ("unrecoverable", "実験の記録を読めません")
                if experiment is None:
                    config = {}
                    config_path = path.parent / "config.yaml"
                    try:
                        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                        if isinstance(value, dict):
                            config = value
                    except (OSError, yaml.YAMLError):
                        pass
                    if not config:
                        for spec_path in sorted(
                            (path.parent / "runs").glob("attempt_*/run_spec.json"),
                            reverse=True,
                        ):
                            try:
                                spec_config = read_json(spec_path).get("config")
                            except (OSError, ValueError, TypeError, KeyError):
                                continue
                            if isinstance(spec_config, dict) and spec_config:
                                config = spec_config
                                break
                    model_config = config.get("model")
                    experiment_config = config.get("experiment")
                    model_type = (
                        model_config.get("type", "") if isinstance(model_config, dict) else ""
                    )
                    experiment = Experiment(
                        expid,
                        experiment_config.get("study_id", "")
                        if isinstance(experiment_config, dict)
                        else "",
                        experiment_config.get("description", "")
                        if isinstance(experiment_config, dict)
                        else "",
                        model_type,
                        ExperimentConfig(config),
                        "unrecoverable",
                    )
                experiment.recovery_state = "unrecoverable"
                experiment.recovery_reason = "実験の記録を読めません"
                self.experiments[expid] = experiment
                logger.warning("実験を読めません (%s): %s", path.parent, error)
        for directory in sorted(self.root.glob("exp_*")):
            if not directory.is_dir() or (directory / "experiment.json").exists():
                continue
            experiment_id = directory.name
            reason = "実験の記録を読めません"
            self.recovery_issues[experiment_id] = ("unrecoverable", reason)
            config: dict[str, Any] = {}
            config_path = directory / "config.yaml"
            try:
                loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    config = loaded
            except (OSError, yaml.YAMLError):
                pass
            if not config:
                for spec_path in sorted(
                    (directory / "runs").glob("attempt_*/run_spec.json"), reverse=True
                ):
                    try:
                        loaded = read_json(spec_path).get("config")
                    except (OSError, ValueError, TypeError, KeyError, AttributeError):
                        continue
                    if isinstance(loaded, dict) and loaded:
                        config = loaded
                        break
            experiment_config = config.get("experiment")
            model_config = config.get("model")
            study_id = (
                experiment_config.get("study_id", "") if isinstance(experiment_config, dict) else ""
            )
            description = (
                experiment_config.get("description", "")
                if isinstance(experiment_config, dict)
                else ""
            )
            model_type = model_config.get("type", "") if isinstance(model_config, dict) else ""
            self.experiments[experiment_id] = Experiment(
                experiment_id,
                study_id if isinstance(study_id, str) else "",
                description if isinstance(description, str) else "",
                model_type if isinstance(model_type, str) else "",
                ExperimentConfig(config),
                "unrecoverable",
                recovery_state="unrecoverable",
                recovery_reason=reason,
            )
        for row in self.queue["rows"]:
            experiment_id = row["experiment_id"]
            if experiment_id in self.experiments:
                continue
            reason = "実験フォルダが見つかりません"
            self.recovery_issues[experiment_id] = ("unrecoverable", reason)
            self.experiments[experiment_id] = Experiment(
                experiment_id,
                "",
                "",
                "",
                ExperimentConfig({}),
                "unrecoverable",
                recovery_state="unrecoverable",
                recovery_reason=reason,
            )

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
        numbers = [
            int(match.group(1))
            for key in self.experiments
            if (match := re.fullmatch(r"exp_(\d+)", key))
        ]
        # 破損記録を含むディレクトリ名も予約済みとして扱う。
        for folder in self.root.glob("exp_*"):
            match = re.fullmatch(r"exp_(\d+)", folder.name)
            if match:
                numbers.append(int(match.group(1)))
        for row in self.queue["rows"]:
            match = re.fullmatch(r"exp_(\d+)", row["experiment_id"])
            if match:
                numbers.append(int(match.group(1)))
        # 削除した実験の番号は再利用しない（古い比較候補などが別の実験を指さないように）
        deleted_number = self.queue.get("max_deleted_experiment_number", 0)
        if type(deleted_number) is int and deleted_number >= 0:
            numbers.append(deleted_number)
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
        issues.extend(
            {"level": "error", "key": key, "message": message}
            for key, message in validate_numeric_config(config)
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
        total_epochs = self._total_epochs(config)
        if existing:
            existing.config = ExperimentConfig(config)
            existing.status = status
            existing.description = exp.get("description", "")
            existing.total_epochs = total_epochs
            return existing
        return Experiment(
            experiment_id,
            exp.get("study_id", "foam_study"),
            exp.get("description", ""),
            model_type,
            ExperimentConfig(config),
            status,
            total_epochs=total_epochs,
        )

    @staticmethod
    def _total_epochs(config: Any) -> int:
        """不正な旧設定でも読込を失敗させず、表示用の既定値を返す。"""
        try:
            value = config.get("training", {}).get("epochs", 40)
            return value if type(value) is int and value > 0 else 40
        except (AttributeError, TypeError):
            return 40

    def save_experiment_draft(
        self, config: dict[str, Any], experiment_id: str | None = None
    ) -> Experiment:
        migrated, _ = self.migrate_experiment_config(config)
        expid = (
            experiment_id or migrated.get("experiment", {}).get("id") or self.next_experiment_id()
        )
        if not isinstance(expid, str) or not re.fullmatch(r"exp_\d{4,}", expid):
            raise ValueError("実験 ID が不正です")
        self._guard_experiment(expid)
        if (self.root / expid).exists() and expid not in self.experiments:
            raise ValueError("既存の実験記録を上書きできません")
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
        self._guard_queue()
        if self.recovery_blockers:
            raise ValueError("前回のプロセスを確認できないため、新しい学習を開始できません")
        experiment = self.save_experiment_draft(config, experiment_id)
        experiment.status = "running"
        self._write_experiment(experiment, False)
        return experiment

    def retry_experiment(self, experiment_id: str) -> Experiment:
        """既存実験を再実行待ちに戻す。試行の作成は行わない。"""
        self._guard_experiment(experiment_id)
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
        self._guard_experiment(experiment_id)
        directory = self.root / experiment_id / "runs" / f"attempt_{attempt:03d}"
        # hello 受信のプロセス同一性を先に残し、process.json 作成前の異常終了も照合可能にする。
        atomic_write_json(
            directory / "hello.json",
            {"pid": int(pid), "creation_time": float(creation_time)},
        )
        atomic_write_json(
            directory / "process.json",
            {"pid": int(pid), "creation_time": float(creation_time)},
        )

    def fail_training_preparation(self, experiment_id: str) -> None:
        """試行ディレクトリを作る前の準備失敗を実験へ記録する。"""
        self._guard_experiment(experiment_id)
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
        self._guard_queue()
        migrated, _ = self.migrate_experiment_config(config)
        expid = migrated.get("experiment", {}).get("id") or self.next_experiment_id()
        if not isinstance(expid, str) or re.fullmatch(r"exp_\d{4,}", expid) is None:
            raise ValueError("実験 ID の形式が不正です")
        self._guard_experiment(expid)
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
        self._guard_queue()
        self._guard_experiment(experiment_id)
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
            issue = self.recovery_issues.get(row["experiment_id"])
            if issue is not None:
                view.status, view.recovery_reason = issue
                view.recovery_state = issue[0]
            else:
                view.status = row["state"]
            view.queue_id = row["queue_id"]
            view.queue_is_retry = row["kind"] == "retry"
            view.queue_retry_attempt = row["attempt"]
            result.append(view)
        return result

    def update_training_queue_item(self, experiment_id: str, config: dict[str, Any]) -> Experiment:
        self._guard_queue()
        self._guard_experiment(experiment_id)
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
        self._guard_queue()
        for row in self.queue["rows"]:
            if row["queue_id"] in experiment_ids:
                self._guard_experiment(row["experiment_id"])
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
        self._guard_queue()
        for row in self.queue["rows"]:
            if row["queue_id"] in experiment_ids:
                self._guard_experiment(row["experiment_id"])
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
        self._guard_queue()
        for row in self.queue["rows"]:
            if row["queue_id"] in experiment_ids:
                self._guard_experiment(row["experiment_id"])
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
        self._guard_queue()
        self.queue["rows"] = [
            row
            for row in self.queue["rows"]
            if row["experiment_id"] in self.recovery_issues
            or row["state"] not in {"completed", "failed", "stopped"}
        ]
        self._save_queue()

    def take_next_training_queue_item(self) -> Experiment | None:
        self._guard_queue()
        if self.recovery_blockers:
            raise ValueError("前回のプロセスを確認できないため、新しい学習を開始できません")
        row = next(
            (
                item
                for item in self.queue["rows"]
                if item["state"] == "queued"
                and item["experiment_id"] not in self.recovery_issues
                and item["experiment_id"] in self.experiments
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
        self._guard_queue()
        row = next((item for item in self.queue["rows"] if item["queue_id"] == queue_id), None)
        if row:
            self._guard_experiment(row["experiment_id"])
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
        self._guard_experiment(experiment_id)
        self._guard_queue()
        if self.recovery_blockers:
            raise ValueError("前回のプロセスを確認できないため、新しい学習を開始できません")
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
            "eval_params": training_eval_params()[config["model"]["type"]],
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
        self._guard_experiment(experiment_id)
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
        seq_state = classify_seq(last_seq, event["seq"])
        if seq_state == "duplicate":
            return self.experiments[experiment_id]
        if seq_state == "gap":
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
        self._guard_experiment(experiment_id)
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
        self._guard_experiment(experiment_id)
        spec = read_run_spec(directory)
        status_path = directory / "status.json"
        stop_path = directory / "stop_request.json"
        error_path = directory / "error.json"
        process_path = directory / "process.json"
        protocol_error = bool(job_exit and job_exit.protocol_error)
        decision = decide_terminal_state(
            existing_status=read_json(status_path) if status_path.exists() else None,
            stop_request=read_json(stop_path) if stop_path.exists() else None,
            result_valid=lambda: self._valid_result(directory),
            error_present=lambda: error_path.exists() or protocol_error,
            start_failed=bool(job_exit and job_exit.start_failed),
            process_alive=lambda: (
                bool(job_exit and job_exit.process_alive)
                or self.process_alive(read_json(process_path) if process_path.exists() else {})
            ),
        )
        queue_id = spec.get("queue_id")
        if decision is None:
            return TrainingOutcome(experiment_id, attempt, queue_id, "running")
        had_status = decision.source == "existing"
        message = decision.message
        if decision.source == "start_failed" or (
            decision.source == "error" and not error_path.exists()
        ):
            message = job_exit.message
        elif decision.source == "error":
            message = read_json(error_path).get("message", "")
        outcome = TrainingOutcome(
            experiment_id, attempt, queue_id, decision.status, decision.reason, message
        )
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
        self._guard_experiment(experiment_id)
        return self._conclude(experiment_id, attempt, job_exit)

    def recover(self) -> list[TrainingOutcome]:
        """起動復旧を行い、キュー行・一時領域・試行状態を照合する。"""
        self.recovery_blockers = []
        for experiment_id, (state, _reason) in list(self.recovery_issues.items()):
            if state == "unconfirmed":
                self.recovery_issues.pop(experiment_id, None)
                experiment = self.experiments.get(experiment_id)
                if experiment is not None and experiment.recovery_state == "unconfirmed":
                    experiment.recovery_state = ""
                    experiment.recovery_reason = ""
        # 先に復旧が触れる全入力を読む。履歴の破損を見つける前に status や
        # queue を書き換えないよう、対象実験単位で保護する。
        protected: set[str] = set(self.recovery_issues)
        preflight_history: dict[str, Experiment] = {}
        for spec_path in sorted(self.root.glob("exp_*/runs/attempt_*/run_spec.json")):
            run_dir = spec_path.parent
            experiment_id = run_dir.parents[1].name
            try:
                spec = read_json(spec_path)
                if (
                    not isinstance(spec.get("run_id"), str)
                    or type(spec.get("attempt")) is not int
                    or spec["attempt"] < 1
                ):
                    raise ValueError("run_spec の必須項目が不正です")
                for name in (
                    "process.json",
                    "hello.json",
                    "status.json",
                    "stop_request.json",
                    "error.json",
                    "result.json",
                    "result.partial.json",
                ):
                    candidate = run_dir / name
                    if candidate.is_file():
                        value = read_json(candidate)
                        if name == "status.json" and (
                            not isinstance(value.get("status"), str)
                            or not value.get("status")
                            or (
                                value.get("reason") is not None
                                and not isinstance(value.get("reason"), str)
                            )
                        ):
                            raise ValueError("status.json の形式が不正です")
                        if name == "stop_request.json" and (
                            value.get("reason") is not None
                            and not isinstance(value.get("reason"), str)
                        ):
                            raise ValueError("stop_request.json の形式が不正です")
                events = run_dir / "events.jsonl"
                if events.is_file():
                    read_events(events, expected_run_id=spec.get("run_id"))
                original = copy.deepcopy(self.experiments.get(experiment_id))
                seq_key = (experiment_id, int(spec["attempt"]))
                prior_seq = self._last_event_seq.get(seq_key)
                try:
                    if original is not None:
                        self._replay_attempt(experiment_id, int(spec["attempt"]))
                        preflight_history[experiment_id] = copy.deepcopy(
                            self.experiments[experiment_id]
                        )
                finally:
                    if original is not None:
                        self.experiments[experiment_id] = original
                    if prior_seq is None:
                        self._last_event_seq.pop(seq_key, None)
                    else:
                        self._last_event_seq[seq_key] = prior_seq
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                protected.add(experiment_id)
                process_file = run_dir / "process.json"
                process_broken = False
                try:
                    if process_file.exists():
                        read_json(process_file)
                except (OSError, ValueError, TypeError, KeyError):
                    process_broken = True
                # ここでは記録の保護だけ決め、計算全体の blocker は後続の
                # PID/作成時刻照合結果が unconfirmed の場合にだけ設定する。
                reason = (
                    "前回の学習プロセスを確認できません。終了後に再起動してください。"
                    if process_broken
                    else "学習履歴を読めません"
                )
                self.recovery_issues[experiment_id] = ("unrecoverable", reason)
                experiment = self.experiments.get(experiment_id)
                if experiment is not None:
                    experiment.recovery_state = "unrecoverable"
                    experiment.recovery_reason = reason
                logger.warning("学習履歴を読めません (%s): %s", run_dir, error)
        outcomes = []
        specs_by_queue: dict[str, tuple[int, str]] = {}
        for spec_path in sorted(self.root.glob("exp_*/runs/attempt_*/run_spec.json")):
            run_dir = spec_path.parent
            experiment_id = run_dir.parents[1].name
            if experiment_id in protected:
                # 状態ファイルの有無を問わず、保護対象の spec は再読込せず、
                # 各試行のプロセス照合だけを行う。
                settled = self._settle_run_identity(run_dir)
                if settled == "missing":
                    settled = settle_unrecorded_worker(
                        run_dir,
                        "foam_cell_analysis.training.run",
                        self.process_alive,
                        self.process_terminator,
                    )
                if settled == "unconfirmed":
                    reason = "前回の学習プロセスを確認できません。終了後に再起動してください。"
                    if reason not in self.recovery_blockers:
                        self.recovery_blockers.append(reason)
                    self.recovery_issues[experiment_id] = ("unconfirmed", reason)
                    experiment = self.experiments.get(experiment_id)
                    if experiment is not None:
                        experiment.recovery_state = "unconfirmed"
                        experiment.recovery_reason = reason
                continue
            spec = read_json(spec_path)
            attempt = int(spec["attempt"])
            if not (run_dir / "status.json").exists():
                settled = self._settle_run_identity(run_dir)
                if settled == "missing":
                    settled = settle_unrecorded_worker(
                        run_dir,
                        "foam_cell_analysis.training.run",
                        self.process_alive,
                        self.process_terminator,
                    )
                if settled == "unconfirmed":
                    reason = "前回の学習プロセスを確認できません。終了後に再起動してください。"
                    self.recovery_blockers.append(reason)
                    self.recovery_issues[experiment_id] = ("unconfirmed", reason)
                    experiment = self.experiments.get(experiment_id)
                    if experiment is not None:
                        experiment.recovery_state = "unconfirmed"
                        experiment.recovery_reason = reason
                    protected.add(experiment_id)
                    continue
                # hello 前に process.json がまだない場合は停止済みと断定しない。
                if settled == "missing":
                    has_terminal_evidence = any(
                        (run_dir / name).is_file()
                        for name in (
                            "status.json",
                            "stop_request.json",
                            "error.json",
                            "result.json",
                        )
                    )
                    queued_before_launch = any(
                        row.get("experiment_id") == experiment_id
                        and row.get("state") == "queued"
                        and row.get("attempt") is None
                        for row in self.queue.get("rows", [])
                    )
                    hello_seen = (run_dir / "hello.json").is_file()
                    if hello_seen or (not has_terminal_evidence and not queued_before_launch):
                        reason = "前回の学習プロセスを確認できません。終了後に再起動してください。"
                        self.recovery_blockers.append(reason)
                        self.recovery_issues[experiment_id] = ("unconfirmed", reason)
                        experiment = self.experiments.get(experiment_id)
                        if experiment is not None:
                            experiment.recovery_state = "unconfirmed"
                            experiment.recovery_reason = reason
                        protected.add(experiment_id)
                        continue
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
        for row in self.queue.get("rows", []) if not self._queue_broken else []:
            if row.get("experiment_id") in protected:
                continue
            found = specs_by_queue.get(row["queue_id"])
            if found:
                row["attempt"], row["state"] = found
            elif row["state"] == "running":
                row.update(state="queued", attempt=None)
        if not self._queue_broken:
            for experiment_id, experiment in self.experiments.items():
                if (
                    experiment_id in protected
                    or experiment_id in self.recovery_issues
                    or experiment.status == "draft"
                    or self._has_attempts(experiment_id)
                ):
                    continue
                has_queued_new_row = any(
                    row.get("experiment_id") == experiment_id
                    and row.get("kind") == "new"
                    and row.get("state") == "queued"
                    for row in self.queue.get("rows", [])
                )
                experiment.status = "queued" if has_queued_new_row else "failed"
        for experiment_dir in sorted(self.root.glob("exp_*")):
            if experiment_dir.name in protected:
                continue
            # リンク（ジャンクション含む）の試行フォルダは _attempt_dirs が除外する
            try:
                attempt_dirs = self._attempt_dirs(experiment_dir.name)
            except ValueError:
                continue
            for _attempt, attempt_dir in attempt_dirs:
                pruned_path = attempt_dir / "pruned.json"
                if not pruned_path.is_file():
                    continue
                try:
                    self._finish_pruning(attempt_dir)
                except (OSError, ValueError):
                    logger.exception("成果物の整理を再開できませんでした: %s", pruned_path)
        for path in self.root.glob("exp_*/runs/.preparing_*"):
            if path.parents[1].name not in protected:
                shutil.rmtree(path, ignore_errors=True)
        for storage_root in (self.root, self.workspace_root / "augmentation"):
            if storage_root.exists():
                for path in storage_root.rglob("*.tmp"):
                    if (
                        storage_root == self.root
                        and self._queue_broken
                        and path.name == "queue.json.tmp"
                    ):
                        continue
                    if storage_root == self.root and any(part in protected for part in path.parts):
                        continue
                    if storage_root.name == "augmentation" and any(
                        key.startswith("profile:") for key in self.recovery_issues
                    ):
                        continue
                    path.unlink(missing_ok=True)
        if not self._queue_broken and not protected:
            self._save_queue()
        try:
            self._replay_history(protected)
            # 保護対象の閲覧用履歴は事前読取中に得たメモリ上の値を使い、
            # プロセス照合後に run_spec/events を再読込しない。
            for experiment_id, experiment in preflight_history.items():
                if experiment_id not in protected:
                    continue
                issue = self.recovery_issues.get(experiment_id)
                if issue is not None:
                    experiment.recovery_state, experiment.recovery_reason = issue
                self.experiments[experiment_id] = experiment
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning("履歴の再構築を完了できません: %s", error)
        return outcomes

    def _settle_run_identity(self, run_dir: Path) -> str:
        """process.json が壊れていても独立した hello.json で照合を試す。"""
        process_file = run_dir / "process.json"
        hello_file = run_dir / "hello.json"
        primary = process_file if process_file.exists() else hello_file
        settled = settle_process(primary, self.process_alive, self.process_terminator)
        if settled == "unconfirmed" and primary == process_file and hello_file.exists():
            fallback = settle_process(hello_file, self.process_alive, self.process_terminator)
            if fallback in {"dead", "terminated"}:
                return fallback
        return settled

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

    def _replay_history(self, protected: set[str] | None = None) -> None:
        """イベントと結果ファイルから全試行の表示用履歴を再構築する。"""
        self._last_event_seq.clear()
        for spec_path in sorted(self.root.glob("exp_*/runs/attempt_*/run_spec.json")):
            run_dir = spec_path.parent
            experiment_id = run_dir.parents[1].name
            if experiment_id in self.experiments and experiment_id not in (protected or set()):
                try:
                    self._replay_attempt(experiment_id, int(run_dir.name[-3:]))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.recovery_issues[experiment_id] = ("unrecoverable", "学習履歴を読めません")
                    experiment = self.experiments[experiment_id]
                    experiment.recovery_state = "unrecoverable"
                    experiment.recovery_reason = "学習履歴を読めません"
                    logger.warning("学習履歴を再構築できません (%s): %s", run_dir, error)

    def create_candidate_snapshot(
        self, experiment_id: str, *, attempt: int | None = None
    ) -> CandidateSnapshot:
        """完了した試行の final.pt・実測 OOF・設定を固定して束ねる。

        attempt を省くと最新の完了試行を使う（従来の呼び出し元との互換のため）。
        """
        self._guard_experiment(experiment_id)
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
        # run_dir 自身と、実験フォルダまでの祖先がリンクなら外部を指している可能性がある
        for ancestor in (run_dir, run_dir.parent, run_dir.parent.parent):
            if ancestor.is_symlink() or ancestor.is_junction():
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
        experiment_dir = self._experiment_dir(experiment_id)
        # 解決前のパスでリンクを判定する（解決後は別実験への junction が通ってしまう）
        unresolved = self.root / experiment_id
        runs_root = experiment_dir / "runs"
        result = []
        if any(
            item.is_symlink() or item.is_junction()
            for item in (unresolved, unresolved / "runs", experiment_dir, runs_root)
        ):
            return result
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

    def _attempt_inventory(
        self, run_dir: Path
    ) -> tuple[
        dict[str, dict[tuple[int, int], int]], dict[tuple[int, int], tuple[int, int, set[str]]]
    ]:
        """種類ごとの実体（st_dev, st_ino）と、実体ごとの大きさ・リンク数・種類を返す。

        ハードリンクで同じ実体を指すファイルは 1 つの実体として扱う。
        戻り値の 1 つ目は 種類 → {実体: その種類内のリンク数}、
        2 つ目は 実体 → (大きさ, st_nlink, 実体を含む種類の集合)。
        """
        by_category: dict[str, dict[tuple[int, int], int]] = {}
        identities: dict[tuple[int, int], tuple[int, int, set[str]]] = {}
        for category in self.ARTIFACT_CATEGORIES:
            for path in self._artifact_files(run_dir, category):
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino)
                links = by_category.setdefault(category, {})
                links[key] = links.get(key, 0) + 1
                entry = identities.setdefault(key, (stat.st_size, stat.st_nlink, set()))
                entry[2].add(category)
        return by_category, identities

    @staticmethod
    def _freed_by(
        categories: set[str],
        by_category: dict[str, dict[tuple[int, int], int]],
        identities: dict[tuple[int, int], tuple[int, int, set[str]]],
    ) -> int:
        """選んだ種類をすべて消したときに実際に空く容量を返す。"""
        freed = 0
        for key, (size, nlink, owners) in identities.items():
            links = sum(by_category.get(category, {}).get(key, 0) for category in owners)
            # 実体の全リンクが選んだ種類に含まれるときだけ容量が空く
            if owners <= categories and links >= nlink:
                freed += size
        return freed

    def artifact_cleanup_plan(
        self,
        experiment_ids: list[str],
        *,
        protected_final: Callable[[str, int], str] | None = None,
    ) -> list[ArtifactGroup]:
        """試行・種類ごとに、消せるファイルの数・容量と可否を返す。

        size_bytes は同じ実体（ハードリンク）を 1 回だけ数えた合計。
        shared_bytes はそのうち他の種類や試行の外と実体を共有していて、単独では空かない分。
        """
        groups = []
        for experiment_id in experiment_ids:
            if experiment_id not in self.experiments:
                raise ValueError(f"実験がありません: {experiment_id}")
            busy = self._experiment_busy_reason(experiment_id)
            for attempt, run_dir in self._attempt_dirs(experiment_id):
                status_path = run_dir / "status.json"
                status = read_json(status_path).get("status") if status_path.is_file() else None
                by_category, identities = self._attempt_inventory(run_dir)
                for category in self.ARTIFACT_CATEGORIES:
                    links = by_category.get(category)
                    if not links:
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
                    size = sum(identities[key][0] for key in links)
                    freed = self._freed_by({category}, by_category, identities)
                    groups.append(
                        ArtifactGroup(
                            experiment_id,
                            attempt,
                            category,
                            sum(links.values()),
                            size,
                            not reason,
                            reason,
                            size - freed,
                        )
                    )
        return groups

    def estimate_freed_bytes(
        self,
        experiment_ids: list[str],
        categories: list[str],
        *,
        protected_final: Callable[[str, int], str] | None = None,
    ) -> int:
        """選んだ種類の組み合わせを整理したときに実際に空く容量を見積もる。

        消せないまとまりは除き、同じ実体の全リンクが選択に含まれるときだけ数える。
        """
        unknown = [item for item in categories if item not in self.ARTIFACT_CATEGORIES]
        if unknown:
            raise ValueError(f"成果物の種類が不正です: {', '.join(unknown)}")
        plan = self.artifact_cleanup_plan(experiment_ids, protected_final=protected_final)
        selected: dict[tuple[str, int], set[str]] = {}
        for group in plan:
            if group.deletable and group.category in categories:
                selected.setdefault((group.experiment_id, group.attempt), set()).add(group.category)
        total = 0
        for (experiment_id, attempt), chosen in selected.items():
            run_dir = self._experiment_dir(experiment_id) / "runs" / f"attempt_{attempt:03d}"
            by_category, identities = self._attempt_inventory(run_dir)
            total += self._freed_by(chosen, by_category, identities)
        return total

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
        """ファイルを消し、空いた大きさを返す。すでにないときは None。

        他のハードリンクが残る（削除前の st_nlink が 2 以上）ときは容量が空かないので 0。
        """
        for trial in range(5):
            try:
                stat = path.stat()
                path.unlink()
                return stat.st_size if stat.st_nlink <= 1 else 0
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
        for experiment_id in experiment_ids:
            self._guard_experiment(experiment_id)
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
        self._guard_experiment(experiment_id)
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
        self._guard_experiment(experiment_id)
        self._guard_queue()
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
