"""メモリ内で動作する画面確認用 Backend。"""

from __future__ import annotations

import copy
import hashlib
import math
from datetime import datetime, timedelta
from typing import Any, ClassVar

import numpy as np

from ..models import (
    AugmentationProfile,
    Candidate,
    Checkpoint,
    CheckResult,
    DataItem,
    DatasetVersion,
    EpochMetrics,
    Evaluation,
    Experiment,
    ExperimentConfig,
    ExternalResult,
    ImportCandidate,
    InferenceConfig,
    ReleasedModel,
    RoutingHistory,
    RunAttempt,
    TransformSetting,
    ValidationIssue,
    ValidationReport,
    WorkingDataset,
)
from .synthetic import make_sample, make_thumbnail, predict_like


class MockBackend:
    """GUI API を網羅した Qt 非依存のインメモリ実装。"""

    classifications: ClassVar[list[str]] = ["分類A", "分類B", "分類C"]
    check_names: ClassVar[list[str]] = [
        "識別子の重複なし",
        "原画像が存在",
        "必要なマスクが存在",
        "原画像とマスクの対応",
        "必須メタデータ入力済み",
        "原画像とマスクの画像サイズ一致",
        "ファイルハッシュ一致",
        "参照先が一意に確定",
    ]

    def __init__(self) -> None:
        self.working: dict[str, WorkingDataset] = {}
        self.versions: list[DatasetVersion] = []
        self.experiments: dict[str, Experiment] = {}
        self.profiles: dict[str, AugmentationProfile] = {}
        self.inference_configs: dict[str, InferenceConfig] = {}
        self.candidates: dict[str, Candidate] = {}
        self.released: dict[str, ReleasedModel] = {}
        self.routing: dict[str, str | None] = {name: None for name in self.classifications}
        self.routing_history: list[RoutingHistory] = []
        self._seed()
        items_by_id = {item.item_id: item for item in self.working["all"].items}
        self._version_items = {
            version.version: [
                copy.deepcopy(items_by_id[item_id])
                for item_id in version.item_ids
                if item_id in items_by_id
            ]
            for version in self.versions
        }

    @staticmethod
    def _now() -> datetime:
        """ローカル timezone 付き現在時刻を返す。"""
        return datetime.now().astimezone()

    @staticmethod
    def _digest(value: str) -> str:
        """安定した SHA-256 文字列を返す。"""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _items(self, purpose: str, count: int, offset: int = 0) -> list[DataItem]:
        """決定的な作業データ項目を生成する。"""
        items = []
        for index in range(count):
            number = offset + index + 1
            revisions = ["rev_001", "rev_002"] if index % 5 == 0 else ["rev_001"]
            items.append(
                DataItem(
                    item_id=f"item_{number:06d}",
                    source_filename=f"foam_{number:04d}.tif",
                    source_relpath=f"{purpose}/images/foam_{number:04d}.tif",
                    channels=["A", "B", "C"],
                    classification=self.classifications[index % 3],
                    quality=["良", "可", "不良"][index % 3],
                    mask_revisions=revisions,
                    selected_mask_revision="rev_001",
                    usage=purpose,
                    sha256=self._digest(f"{purpose}:{number}"),
                    seed=number,
                )
            )
        return items

    def _seed(self) -> None:
        """設計書の画面確認用データを作る。"""
        now = self._now()
        for purpose, prefix, count, offset in (
            ("train", "train", 60, 0),
            ("val", "val", 30, 1000),
        ):
            versions = [f"{prefix}_v{index:03d}" for index in range(1, 4)]
            for index, version in enumerate(versions, start=1):
                created = now - timedelta(days=(4 - index) * 14 + (0 if purpose == "train" else 2))
                archive = {1: "NOT_CREATED", 2: "FAILED", 3: "COMPLETED"}[index]
                self.versions.append(
                    DatasetVersion(
                        version=version,
                        purpose=purpose,
                        parent_version=f"{prefix}_v{index - 1:03d}" if index > 1 else None,
                        created_at=created,
                        item_ids=[
                            f"item_{item_id:06d}"
                            for item_id in range(offset + 1, offset + count + 1)
                        ],
                        n_images=count,
                        archive_status=archive,
                        archive_sha256=self._digest(version) if archive == "COMPLETED" else None,
                    )
                )
            items = self._items(purpose, count, offset)
            if purpose == "train":
                items[0].classification = None
                items[1].quality = None
                items[2].change = "added"
                items[3].change = "changed"
                items[4].included = False
                items[4].change = "excluded"
            self.working[purpose] = WorkingDataset(purpose, versions[-1], items)

        train = self.working["train"]
        val = self.working["val"]
        unassigned = self._items("unassigned", 23, 2000)
        for item in unassigned[:2]:
            item.classification = None
            item.quality = None
        master = WorkingDataset(
            "all",
            train.base_version,
            train.items + val.items + unassigned,
            base_train_version=train.base_version,
            base_val_version=val.base_version,
        )
        self.working = {"train": master, "val": master, "all": master}
        validation_by_version = {
            version.version.rsplit("_", 1)[-1]: version.version
            for version in self.versions
            if version.purpose == "val"
        }
        for version in self.versions:
            if version.purpose == "train":
                key = version.version.rsplit("_", 1)[-1]
                version.base_validation_version = validation_by_version.get(key)

        self._seed_profiles()
        exp42 = self._seed_experiment("exp_0042", "mask_rcnn", "completed", 100)
        exp43 = self._seed_experiment("exp_0043", "cellpose", "completed", 100)
        exp44 = self._seed_experiment("exp_0044", "mask_rcnn", "stopped", 12)
        exp44.runs = [
            RunAttempt(
                1,
                now - timedelta(days=3),
                now - timedelta(days=3),
                "メモリ不足で失敗",
                {"GPU": "Mock GPU"},
            ),
            RunAttempt(
                2, now - timedelta(days=2), now - timedelta(days=2), "中断", {"GPU": "Mock GPU"}
            ),
        ]
        self._seed_experiment("exp_0045", "mask_rcnn", "draft", 0)
        self._seed_inference_configs()
        self._seed_candidates_and_releases(exp42, exp43)
        self._sync_profile_usage()

    def _seed_profiles(self) -> None:
        """14種の変換を持つ独立したプロファイルを作る。"""
        definitions = [
            ("horizontal_flip", "左右反転", True, 0.5, None, None),
            ("vertical_flip", "上下反転", True, 0.5, None, None),
            ("rotation", "回転", True, 0.5, -180, 180),
            ("scale", "拡大縮小", True, 0.3, 0.8, 1.2),
            ("translate", "平行移動", False, 0.2, -0.1, 0.1),
            ("crop", "切り出し", False, 0.2, 0.7, 1.0),
            ("elastic", "弾性変形", False, 0.1, 0.0, 0.05),
            ("brightness", "明るさ", True, 0.3, -0.1, 0.1),
            ("contrast", "コントラスト", True, 0.3, 0.9, 1.1),
            ("gamma", "ガンマ", False, 0.2, 0.8, 1.2),
            ("blur", "ぼかし", True, 0.2, 0.0, 1.0),
            ("noise", "ノイズ", True, 0.2, 0.0, 0.05),
            ("channel_dropout", "チャンネル欠落", False, 0.1, None, None),
            ("channel_intensity", "チャンネル強度変動", False, 0.2, 0.8, 1.2),
        ]
        base_transforms = [TransformSetting(*definition) for definition in definitions]
        for index in range(1, 4):
            profile_id = f"aug_v{index:03d}"
            self.profiles[profile_id] = AugmentationProfile(
                profile_id=profile_id,
                name=f"foam_cell_aug_v{index:03d}",
                base_profile=f"aug_v{index - 1:03d}" if index > 1 else None,
                transforms=copy.deepcopy(base_transforms),
                order=[setting.key for setting in base_transforms],
            )

    def _seed_experiment(self, expid: str, model_type: str, status: str, epochs: int) -> Experiment:
        """指定エポック数の学習履歴とチェックポイントを作る。"""
        config = self.default_experiment_config(model_type)
        config["experiment"]["description"] = {
            "exp_0042": "分類A / 入力画像サイズ比較",
            "exp_0043": "分類B / Cellpose 基準モデル",
            "exp_0044": "入力解像度のメモリ使用量確認",
            "exp_0045": "学習設定の下書き",
        }[expid]
        config["experiment"]["id"] = expid
        config["training"]["epochs"] = 100 if status == "completed" else max(1, epochs)
        config["augmentation"]["profile"] = (
            "aug_v003" if expid in {"exp_0042", "exp_0045"} else "aug_v002"
        )
        config["data"]["used_item_ids"] = (
            [item.item_id for item in self.working["train"].items[:40]] if status != "draft" else []
        )
        history = self._make_history(expid, epochs, status)
        checkpoints: list[Checkpoint] = []
        if status == "completed":
            checkpoints.extend(
                Checkpoint(
                    name=f"epoch_{epoch:03d}{'.pt' if model_type == 'mask_rcnn' else ''}",
                    epoch=epoch,
                    map=self._map_at(history, epoch),
                    saved_at=self._now() - timedelta(days=1, minutes=100 - epoch),
                )
                for epoch in range(10, 101, 10)
            )
            best_epoch, best_map = max(
                ((metric.epoch, metric.map) for metric in history if metric.map is not None),
                key=lambda pair: pair[1],
            )
            checkpoints.extend(
                [
                    Checkpoint(
                        "best.pt" if model_type == "mask_rcnn" else "best", best_epoch, best_map
                    ),
                    Checkpoint(
                        "last.pt" if model_type == "mask_rcnn" else "last", 100, history[-1].map
                    ),
                ]
            )
        runs = []
        if status != "draft":
            now = self._now()
            runs.append(
                RunAttempt(
                    1, now - timedelta(days=1), now, "完走" if status == "completed" else "失敗"
                )
            )
        used = config["data"]["used_item_ids"]
        exp = Experiment(
            experiment_id=expid,
            study_id="foam_study",
            description=config["experiment"]["description"],
            model_type=model_type,
            config=ExperimentConfig(config),
            status=status,
            current_epoch=100 if status == "completed" else (epochs if status != "draft" else 0),
            total_epochs=config["training"]["epochs"],
            history=history,
            checkpoints=checkpoints,
            runs=runs,
            used_item_ids=used,
            created_at=self._now() - timedelta(days=int(expid[-2:])),
        )
        self.experiments[expid] = exp
        return exp

    @staticmethod
    def _make_history(expid: str, epochs: int, status: str) -> list[EpochMetrics]:
        """減少損失と飽和傾向 mAP を持つ学習曲線を生成する。"""
        if status == "draft":
            return []
        count = 100 if status == "completed" else epochs
        seed = int(expid[-4:])
        rng = np.random.default_rng(seed)
        metrics = []
        for epoch in range(1, count + 1):
            loss = 1.4 * math.exp(-epoch / 30) + 0.04 + float(rng.normal(0, 0.008))
            map_value = None
            if epoch % 5 == 0:
                map_value = min(0.97, 0.45 + 0.52 * (1 - math.exp(-epoch / 32)))
                map_value += float(rng.normal(0, 0.012))
                map_value = max(0.0, min(0.99, map_value))
            metrics.append(EpochMetrics(epoch, max(0.01, loss), map_value))
        return metrics

    @staticmethod
    def _map_at(history: list[EpochMetrics], epoch: int) -> float | None:
        """指定エポックの直近 mAP を返す。"""
        return next((metric.map for metric in reversed(history) if metric.epoch == epoch), None)

    def _seed_inference_configs(self) -> None:
        """仕様書のキーを使う推論設定を作る。"""
        self.inference_configs["infer_v005"] = InferenceConfig(
            "infer_v005",
            "mask_rcnn",
            {"box_score_thresh": 0.5, "box_nms_thresh": 0.5, "box_detections_per_img": 100},
        )
        self.inference_configs["infer_v006"] = InferenceConfig(
            "infer_v006",
            "cellpose",
            {"cellprob_threshold": 0.0, "flow_threshold": 0.4},
        )

    def _seed_candidates_and_releases(self, exp42: Experiment, exp43: Experiment) -> None:
        """候補・リリース・振り分けの参照関係を作る。"""
        self.candidates["RC-001"] = Candidate(
            "RC-001",
            exp42.experiment_id,
            "best.pt",
            "infer_v005",
            evaluations={"val_v003": self._evaluation(0.91)},
        )
        self.candidates["RC-002"] = Candidate(
            "RC-002",
            exp43.experiment_id,
            "best",
            "infer_v006",
            evaluations={"val_v003": self._evaluation(0.89)},
        )
        self.candidates["RC-003"] = Candidate(
            "RC-003",
            exp42.experiment_id,
            "epoch_080.pt",
            "infer_v005",
        )
        now = self._now()
        for model_id, candidate_id in (("model_007", "RC-001"), ("model_012", "RC-002")):
            candidate = self.candidates[candidate_id]
            experiment = self.experiments[candidate.experiment_id]
            inference = self.inference_configs[candidate.inference_config_id]
            evaluation = candidate.evaluations["val_v003"]
            self.released[model_id] = ReleasedModel(
                model_id=model_id,
                candidate_id=candidate_id,
                experiment_id=experiment.experiment_id,
                checkpoint=candidate.checkpoint,
                preprocessing_config=copy.deepcopy(experiment.config.values["model"]),
                inference_config=copy.deepcopy(inference.params),
                validation_dataset="val_v003",
                evaluation_result=evaluation,
                released_at=now - timedelta(days=30 if model_id == "model_007" else 8),
            )
        self.routing.update({"分類A": "model_007", "分類B": "model_012", "分類C": "model_007"})

    def _sync_profile_usage(self) -> None:
        """実験の設定を基準にプロファイル使用先を構築する。"""
        for profile in self.profiles.values():
            profile.used_by_experiments = [
                experiment.experiment_id
                for experiment in self.experiments.values()
                if experiment.status != "draft"
                and experiment.config.values["augmentation"]["profile"] == profile.profile_id
            ]

    @staticmethod
    def _evaluation(score: float) -> Evaluation:
        """決定的な分類別モック評価を返す。"""
        classes = MockBackend.classifications
        return Evaluation(
            score,
            {name: (score - index * 0.03, 10 + index) for index, name in enumerate(classes)},
        )

    def get_working_dataset(self, purpose: str = "train") -> WorkingDataset:
        """学習・検証側へ互換の用途別表示を返す。"""
        master = self.working["all"]
        if purpose == "all":
            return master
        return WorkingDataset(
            purpose,
            master.base_train_version if purpose == "train" else master.base_val_version or "",
            [item for item in master.items if item.usage == purpose],
            master.state,
            master.last_saved_at,
            master.validation,
            master.base_train_version,
            master.base_val_version,
        )

    def get_working_items(self) -> list[DataItem]:
        """用途を問わず全ての作業項目を返す。"""
        return self.working["all"].items

    def update_items(self, item_ids: list[str], **changes: Any) -> None:
        """複数項目をまとめて更新する。"""
        self.bulk_update_items("all", item_ids, **changes)

    def set_usage(self, item_ids: list[str], usage: str) -> None:
        """複数項目へ同じ用途を設定する。"""
        if usage not in {"unassigned", "train", "val", "excluded"}:
            raise ValueError("用途の値が正しくありません")
        self.update_items(item_ids, usage=usage)

    def scan_import_folders(
        self, paths: dict[str, str], mask_rule: Any = None, channel_rule: Any = None
    ) -> list[ImportCandidate]:
        """複数チャンネルの取り込み元を走査する。"""
        return self.scan_import_source(paths, str(mask_rule or ""))

    def import_folders(
        self, scans: list[ImportCandidate], uniform_values: dict[str, dict[str, Any]]
    ) -> list[DataItem]:
        """フォルダごとの設定を各取り込み画像へ適用する。"""
        grouped: dict[str, list[ImportCandidate]] = {}
        for candidate in scans:
            folder = candidate.source_relpath.rsplit("/", 1)[0]
            grouped.setdefault(folder, []).append(candidate)
        imported: list[DataItem] = []
        for folder, candidates in grouped.items():
            values = dict(uniform_values.get(folder, {}))
            classification = values.pop("classification", None)
            group = self.import_items("all", candidates, classification)
            if values:
                self.update_items([item.item_id for item in group], **values)
            imported.extend(group)
        return imported

    def preview_auto_split(
        self, settings: dict[str, Any], item_ids: list[str] | None = None
    ) -> dict[str, int]:
        """自動振り分けの実行後件数を返す。"""
        assignments = self.preview_auto_split_assignments(settings, item_ids)
        return {
            usage: sum(value == usage for value in assignments.values())
            for usage in ("train", "val", "excluded")
        }

    def preview_auto_split_assignments(
        self, settings: dict[str, Any], item_ids: list[str] | None = None
    ) -> dict[str, str]:
        """変更を保存せず対象ごとの実行後用途を返す。"""
        allowed = set(item_ids) if item_ids is not None else None
        include_assigned = bool(settings.get("include_assigned", False))
        item_ids_to_update = [
            item.item_id
            for item in self.working["all"].items
            if (include_assigned or item.usage == "unassigned")
            and (allowed is None or item.item_id in allowed)
        ]
        settings = dict(settings)
        if "ratio" in settings:
            settings["validation_ratio"] = settings.pop("ratio")
        if "unit" in settings:
            settings["by_folder"] = settings.pop("unit") == "source_folder"
        clone = copy.copy(self)
        clone.working = {"all": copy.deepcopy(self.working["all"])}
        clone.working.update({"train": clone.working["all"], "val": clone.working["all"]})
        clone.apply_auto_split(settings, item_ids)
        return {item_id: clone._get_item("all", item_id).usage for item_id in item_ids_to_update}

    def apply_auto_split(
        self, settings: dict[str, Any], item_ids: list[str] | None = None
    ) -> dict[str, int]:
        """自動振り分けを作業データへ適用する。"""
        values = dict(settings)
        if "ratio" in values:
            values["validation_ratio"] = values.pop("ratio")
        if "unit" in values:
            values["by_folder"] = values.pop("unit") == "source_folder"
        return self.apply_auto_triage(values, item_ids)

    def validate_items(self, item_ids: list[str] | None = None) -> ValidationReport:
        """全件または指定項目を検査する。"""
        if item_ids is None:
            return self.validate_all_working_items()
        dataset = self.working["all"]
        selected = set(item_ids)
        targets = [item for item in dataset.items if item.item_id in selected]
        errors = [
            ValidationIssue("エラー", item.item_id, "必須メタデータ", "分類または品質が未設定です")
            for item in targets
            if item.usage in {"train", "val"}
            and (not item.classification or not item.quality or not item.mask_revisions)
        ]
        checks = dataset.validation.checks if dataset.validation else []
        report = ValidationReport(checks, errors)
        return report

    def summarize_finalize(self) -> dict[str, dict[str, int | str]]:
        """学習用・検証用の一括確定内容を集計する。"""
        return {purpose: self.summarize_working_changes(purpose) for purpose in ("train", "val")}

    def finalize_working(
        self, comment: str = "", create_archive: bool = False
    ) -> list[DatasetVersion]:
        """学習・検証の変更版を同時に確定する。"""
        return self.finalize_working_dataset(comment, create_archive)

    def validate_all_working_items(self) -> ValidationReport:
        """学習・検証に使う項目の必須メタデータを自動検査する。"""
        dataset = self.working["all"]
        errors = [
            ValidationIssue("エラー", item.item_id, "必須メタデータ", "分類または品質が未設定です")
            for item in dataset.items
            if item.usage in {"train", "val"}
            and (not item.classification or not item.quality or not item.mask_revisions)
        ]
        dataset.validation = ValidationReport(
            [
                CheckResult(
                    "metadata", "必須メタデータ入力済み", "NG" if errors else "OK", len(errors)
                )
            ],
            errors,
        )
        return dataset.validation

    def apply_auto_triage(
        self, settings: dict[str, Any], item_ids: list[str] | None = None
    ) -> dict[str, int]:
        """未振り分け画像を分類別に安定した疑似乱数で振り分ける。"""
        dataset = self.working["all"]
        allowed = set(item_ids) if item_ids is not None else None
        include_assigned = bool(settings.get("include_assigned", False))
        candidates = [
            item
            for item in dataset.items
            if (include_assigned or item.usage == "unassigned")
            and (allowed is None or item.item_id in allowed)
        ]
        ratio = max(0, min(100, int(settings.get("validation_ratio", 20))))
        seed = int(settings.get("seed", 42))
        by_folder = bool(settings.get("by_folder", False))
        stratify = bool(settings.get("stratify", True))
        bad_to_excluded = bool(settings.get("bad_quality_to_excluded", False))
        buckets: dict[str, list[DataItem]] = {}
        folder_classes: dict[str, str] = {}
        if by_folder and stratify:
            folder_counts: dict[str, dict[str, int]] = {}
            for item in candidates:
                classes = folder_counts.setdefault(item.source_folder, {})
                label = item.classification or "未設定"
                classes[label] = classes.get(label, 0) + 1
            folder_classes = {
                folder: max(classes, key=classes.get) for folder, classes in folder_counts.items()
            }
        for item in candidates:
            bucket = (
                folder_classes[item.source_folder]
                if by_folder and stratify
                else (item.classification or "未設定")
                if stratify
                else "all"
            )
            buckets.setdefault(bucket, []).append(item)
        counts = {"train": 0, "val": 0, "excluded": 0}
        for key, values in buckets.items():
            if by_folder:
                folders: dict[str, list[DataItem]] = {}
                for item in values:
                    folders.setdefault(item.source_folder, []).append(item)
                ordered_folders = sorted(
                    folders, key=lambda folder: self._digest(f"{seed}:{key}:{folder}")
                )
                target = round(len(values) * ratio / 100)
                validation_folders: set[str] = set()
                validation_count = 0
                for folder in ordered_folders:
                    size = len(folders[folder])
                    if abs(validation_count + size - target) <= abs(validation_count - target):
                        validation_folders.add(folder)
                        validation_count += size
                assignments = [
                    (item, "val" if item.source_folder in validation_folders else "train")
                    for item in values
                ]
            else:
                ordered = sorted(
                    values, key=lambda item: self._digest(f"{seed}:{key}:{item.item_id}")
                )
                n_val = round(len(ordered) * ratio / 100)
                assignments = [
                    (item, "val" if index < n_val else "train")
                    for index, item in enumerate(ordered)
                ]
            for item, usage in assignments:
                if bad_to_excluded and item.quality == "不良":
                    usage = "excluded"
                item.usage = usage
                item.change = "added" if item.change == "added" else "changed"
                counts[usage] += 1
        dataset.last_saved_at = self._now()
        self.validate_all_working_items()
        return counts

    def finalize_working_dataset(
        self, comment: str = "", create_archive: bool = False
    ) -> list[DatasetVersion]:
        """変更のある用途だけを一回の操作で確定する。"""
        report = self.validate_all_working_items()
        if report.errors:
            raise ValueError("整合性エラーを解消してください")
        dataset = self.working["all"]
        created: list[DatasetVersion] = []
        for purpose, base in (
            ("train", dataset.base_train_version),
            ("val", dataset.base_val_version),
        ):
            items = [item for item in dataset.items if item.usage == purpose]
            changed = [item for item in dataset.items if item.change and item.usage == purpose]
            latest = next((v for v in reversed(self.versions) if v.purpose == purpose), None)
            item_ids = [item.item_id for item in items]
            if latest and latest.item_ids == item_ids and not changed:
                continue
            version = DatasetVersion(
                self.next_dataset_version(purpose),
                purpose,
                base,
                self._now(),
                item_ids,
                len(items),
                comment=comment,
                base_validation_version=None,
            )
            self.versions.append(version)
            self._version_items[version.version] = copy.deepcopy(items)
            created.append(version)
            if create_archive:
                self.record_archive_result(version.version, "mock_archive")
            if purpose == "train":
                dataset.base_train_version = version.version
            else:
                dataset.base_val_version = version.version
        for item in dataset.items:
            item.change = None
            item.previous_change = None
        dataset.base_version = dataset.base_train_version or dataset.base_val_version or ""
        linked_val = next(
            (version.version for version in reversed(created) if version.purpose == "val"),
            next(
                (
                    version.version
                    for version in reversed(self.versions)
                    if version.purpose == "val"
                ),
                None,
            ),
        )
        for version in created:
            if version.purpose == "train":
                version.base_validation_version = linked_val
        dataset.last_saved_at = self._now()
        return created

    def update_item(self, purpose: str, item_id: str, **changes: Any) -> DataItem:
        """項目を更新し、変更状態・自動保存時刻を更新する。"""
        dataset = self.working["all"]
        item = next(item for item in dataset.items if item.item_id == item_id)
        revision = changes.get("selected_mask_revision")
        if revision and revision not in item.mask_revisions:
            raise ValueError(f"{item_id} にマスク版 {revision} はありません")
        was_included = item.included
        original_change = item.previous_change if item.change == "excluded" else item.change
        legacy_included = "included" in changes
        if legacy_included:
            changes["usage"] = "train" if changes.pop("included") else "excluded"
        for key, value in changes.items():
            if not hasattr(item, key):
                raise AttributeError(key)
            setattr(item, key, value)
        if "usage" in changes and item.usage == "excluded":
            item.previous_change = original_change
            item.change = "excluded"
        elif "usage" in changes and item.included and not was_included:
            if original_change == "added":
                item.change = "added"
            elif any(
                key in changes
                for key in (
                    ("classification", "quality", "selected_mask_revision")
                    if legacy_included
                    else ("usage", "classification", "quality", "selected_mask_revision")
                )
            ):
                item.change = "changed"
            else:
                item.change = original_change
        elif not item.included:
            item.change = "excluded"
        elif original_change == "added":
            item.change = "added"
        elif any(
            key in changes for key in ("classification", "quality", "selected_mask_revision")
        ) or ("usage" in changes and not legacy_included):
            item.change = "changed"
        if "usage" in changes and item.included:
            item.previous_change = None
        dataset.state = "WORKING"
        dataset.validation = None
        dataset.last_saved_at = self._now()
        return item

    def bulk_update_items(self, purpose: str, item_ids: list[str], **changes: Any) -> None:
        """複数項目を一括更新する。"""
        for item_id in item_ids:
            self.update_item(purpose, item_id, **changes)

    def add_imported_items(self, purpose: str, items: list[DataItem]) -> list[DataItem]:
        """作業中データセットへ項目を追加する。"""
        dataset = self.working["all"]
        for item in items:
            item.change = "added"
            dataset.items.append(item)
        dataset.state = "WORKING"
        dataset.last_saved_at = self._now()
        return items

    def add_mask_revision(self, purpose: str, item_id: str) -> str:
        """上書きせずに次のマスク版を追加する。"""
        item = next(item for item in self.working[purpose].items if item.item_id == item_id)
        revision = f"rev_{len(item.mask_revisions) + 1:03d}"
        item.mask_revisions.append(revision)
        self.update_item(purpose, item_id, selected_mask_revision=revision)
        return revision

    def validate_working_dataset(self, purpose: str) -> ValidationReport:
        """8項目の整合性確認を行い結果を保存する。"""
        dataset = self.working["all"]
        checks = [
            CheckResult(f"check_{index:02d}", name)
            for index, name in enumerate(self.check_names, 1)
        ]
        issues = [
            ValidationIssue(
                "エラー", item.item_id, self.check_names[4], "画像分類または品質が未設定です"
            )
            for item in dataset.items
            if item.usage in {"train", "val"} and (not item.classification or not item.quality)
        ]
        for check in checks:
            if check.key == "check_05":
                check.status = "NG" if issues else "OK"
                check.count = len(issues)
            else:
                check.status = "OK"
        report = ValidationReport(checks, issues)
        dataset.validation = report
        dataset.state = "WORKING" if issues else "VALIDATED"
        dataset.last_saved_at = self._now()
        return report

    def next_dataset_version(self, purpose: str) -> str:
        """用途別の次版名を返す。"""
        prefix = "train" if purpose == "train" else "val"
        versions = [
            int(version.version[-3:]) for version in self.versions if version.purpose == purpose
        ]
        return f"{prefix}_v{max(versions, default=0) + 1:03d}"

    def summarize_working_changes(self, purpose: str) -> dict[str, int | str]:
        """確定ダイアログ用に作業差分と件数を集計する。"""
        dataset = self.working["all"]
        items = [item for item in dataset.items if item.usage == purpose]
        latest = next(
            (version for version in reversed(self.versions) if version.purpose == purpose), None
        )
        item_ids = [item.item_id for item in items]
        has_changes = bool(
            (latest is None and items)
            or (latest is not None and latest.item_ids != item_ids)
            or any(item.change for item in items)
        )
        return {
            "added": sum(item.change == "added" for item in items),
            "removed": sum(item.change == "excluded" for item in items),
            "changed": sum(item.change == "changed" for item in items),
            "n_images": len(items),
            "n_masks": sum(bool(item.mask_revisions) for item in items),
            "missing_metadata": sum(
                (not item.classification or not item.quality) for item in items
            ),
            "next_version": self.next_dataset_version(purpose),
            "has_changes": has_changes,
        }

    def check_dataset_duplicates(
        self, purpose: str, validation_version: str | None = None
    ) -> list[tuple[str, str]]:
        """学習用データと基準検証版の識別子・ハッシュ重複を返す。"""
        if purpose != "train" or validation_version is None:
            return []
        train_items = [item for item in self.working["all"].items if item.usage == "train"]
        validation_items = self._items_for_version(validation_version)
        val_ids = {item.item_id for item in validation_items}
        val_hashes = {item.sha256 for item in validation_items}
        return [
            (item.item_id, "識別子" if item.item_id in val_ids else "ハッシュ")
            for item in train_items
            if item.item_id in val_ids or item.sha256 in val_hashes
        ]

    def finalize_dataset(
        self,
        purpose: str,
        comment: str = "",
        base_validation_version: str | None = None,
    ) -> DatasetVersion:
        """検証済み作業データを新しい版として確定する。"""
        dataset = self.working["all"]
        if dataset.state != "VALIDATED":
            raise ValueError("整合性確認が完了していません")
        duplicates = self.check_dataset_duplicates(purpose, base_validation_version)
        if duplicates:
            raise ValueError("基準検証用データセットと重複するデータがあります")
        version = DatasetVersion(
            version=self.next_dataset_version(purpose),
            purpose=purpose,
            parent_version=(
                dataset.base_train_version if purpose == "train" else dataset.base_val_version
            ),
            created_at=self._now(),
            item_ids=[item.item_id for item in dataset.items if item.usage == purpose],
            n_images=sum(item.usage == purpose for item in dataset.items),
            comment=comment,
            base_validation_version=base_validation_version,
        )
        self.versions.append(version)
        self._version_items[version.version] = copy.deepcopy(
            [item for item in dataset.items if item.usage == purpose]
        )
        if purpose == "train":
            dataset.base_train_version = version.version
        else:
            dataset.base_val_version = version.version
        dataset.base_version = version.version
        dataset.state = "WORKING"
        for item in dataset.items:
            if item.usage != purpose:
                continue
            item.change = None
            item.previous_change = None
        return version

    def list_dataset_versions(self, purpose: str | None = None) -> list[DatasetVersion]:
        """確定済みデータセット版を返す。"""
        return [
            version for version in self.versions if purpose is None or version.purpose == purpose
        ]

    def get_dataset_version_items(self, version: str) -> list[DataItem]:
        """指定版の確定時スナップショットを返す。"""
        return copy.deepcopy(self._version_items.get(version, []))

    def list_validation_versions(self) -> list[DatasetVersion]:
        """検証用確定データセット版を返す。"""
        return self.list_dataset_versions("val")

    def record_archive_result(
        self, version: str, output_path: str, ok: bool = True
    ) -> DatasetVersion:
        """アーカイブ結果を記録する。"""
        record = next(item for item in self.versions if item.version == version)
        record.archive_status = "FAILED" if "fail" in output_path.lower() or not ok else "COMPLETED"
        record.archive_sha256 = (
            self._digest(version) if record.archive_status == "COMPLETED" else None
        )
        return record

    def get_last_saved_at(self, purpose: str = "train") -> datetime:
        """作業データセットの自動保存時刻を返す。"""
        return self.working["all"].last_saved_at

    def scan_import_source(
        self, image_dirs: dict[str, str], mask_dir: str
    ) -> list[ImportCandidate]:
        """取り込み元から決定的な架空ファイル候補を返す。"""
        source_key = (
            "|".join(f"{key}:{image_dirs[key]}" for key in sorted(image_dirs)) + f"|{mask_dir}"
        )
        if "A" not in image_dirs:
            raise ValueError("チャンネルAの画像フォルダは必須です")
        count = 5 + int(self._digest(source_key)[:2], 16) % 6
        candidates = []
        for index in range(count):
            filename = f"import_{index + 1:03d}.tif"
            candidates.append(
                ImportCandidate(
                    source_filename=filename,
                    source_relpath=f"{source_key}/{filename}",
                    channels=sorted(image_dirs),
                    mask_available=bool(mask_dir),
                    seed=int(self._digest(f"{source_key}:{index}")[:8], 16),
                    sha256=self._digest(f"{source_key}:{filename}"),
                )
            )
        return candidates

    def import_items(
        self,
        purpose: str,
        candidates: list[ImportCandidate],
        classification: str | None,
    ) -> list[DataItem]:
        """取り込み候補を DataItem に変換して追加する。"""
        dataset = self.working[purpose]
        start = max((int(item.item_id[-6:]) for item in dataset.items), default=0)
        items = [
            DataItem(
                item_id=f"item_{start + index:06d}",
                source_filename=candidate.source_filename,
                source_relpath=candidate.source_relpath,
                channels=candidate.channels.copy(),
                classification=classification,
                quality=None,
                mask_revisions=["rev_001"] if candidate.mask_available else [],
                selected_mask_revision="rev_001" if candidate.mask_available else "",
                usage="unassigned",
                change="added",
                sha256=candidate.sha256,
                seed=candidate.seed,
            )
            for index, candidate in enumerate(candidates, 1)
        ]
        return self.add_imported_items(purpose, items)

    def get_item_image(self, purpose: str, item_id: str, channel: str) -> np.ndarray:
        """合成原画像の指定チャンネルを返す。"""
        item = self._get_item(purpose, item_id)
        images, _ = make_sample(item.seed, channels=(channel,))
        return images[channel].copy()

    def get_item_thumbnail(self, item_id: str, size: tuple[int, int]) -> np.ndarray:
        """小さな合成画像を直接生成してサムネイルに返す。"""
        item = self._get_item("all", item_id)
        return make_thumbnail(item.seed, size)

    def get_item_mask(self, purpose: str, item_id: str, revision: str) -> np.ndarray:
        """指定マスク版の合成ラベルを返す。"""
        item = self._get_item(purpose, item_id)
        _, labels = make_sample(item.seed, channels=("A",))
        if revision not in item.mask_revisions:
            raise KeyError(revision)
        if revision == "rev_001":
            return labels.copy()
        return predict_like(labels, item.seed + self._seed_number(revision)).copy()

    def get_candidate_prediction(self, candidate_id: str, item_id: str) -> np.ndarray:
        """候補別に決定的な予測ラベルを返す。"""
        if candidate_id not in self.candidates:
            raise KeyError(candidate_id)
        _, labels = make_sample(self._get_item("val", item_id).seed, channels=("A",))
        return predict_like(
            labels, self._seed_number(candidate_id) + self._get_item("val", item_id).seed
        )

    def get_inference_result(self, filename: str, model_id: str) -> tuple[np.ndarray, np.ndarray]:
        """ファイル名とモデルから決定的な合成画像・予測ラベルを返す。"""
        model = self.released[model_id]
        seed = self._seed_number(filename + model.model_id)
        images, labels = make_sample(seed, channels=("A",))
        return images["A"].copy(), predict_like(labels, seed + 7)

    @staticmethod
    def _seed_number(value: str) -> int:
        """文字列を安定した整数シードへ変換する。"""
        return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)

    def _get_item(self, purpose: str, item_id: str) -> DataItem:
        """用途・識別子から作業項目を取得する。"""
        return next(item for item in self.working[purpose].items if item.item_id == item_id)

    def _items_for_version(self, version: str) -> list[DataItem]:
        """版に含まれる画像項目を返す。"""
        record = next(item for item in self.versions if item.version == version)
        working = self.working[record.purpose].items
        by_id = {item.item_id: item for item in working}
        return [by_id[item_id] for item_id in record.item_ids if item_id in by_id]

    def list_training_options(self) -> dict[str, list[str]]:
        """学習フォームで使用する選択肢を返す。"""
        return {
            "datasets": [
                version.version for version in self.versions if version.purpose == "train"
            ],
            "splits": ["split_001", "split_002"],
            "classifications": self.classifications.copy(),
            "qualities": ["良", "可", "不良"],
            "channels": ["A", "B", "C"],
            "backbones": ["resnet50_fpn_v2", "resnet101_fpn"],
            "pretrained": ["coco", "imagenet"],
            "cellpose_models": ["cyto3", "nuclei", "cpsam"],
            "optimizers": ["Adam", "SGD"],
            "profiles": list(self.profiles),
        }

    def default_experiment_config(self, model_type: str) -> dict[str, Any]:
        """内部キー仕様に沿ったモデル別既定設定を返す。"""
        common = {
            "experiment": {"id": None, "study_id": "foam_study", "description": ""},
            "data": {
                "dataset_version": "train_v003",
                "split_id": "split_001",
                "seed": 42,
                "classification": "all",
                "quality_filter": "all",
                "input_channels": ["A", "B", "C"],
                "used_item_ids": [],
            },
            "training": {
                "epochs": 100,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "early_stopping": {"enabled": True, "patience": 10},
            },
            "augmentation": {"profile": "aug_v003"},
            "checkpoint": {
                "save_every": 10,
                "save_last": True,
                "save_best": True,
                "best_metric": "instance_map",
                "best_mode": "max",
                "validation_interval": 5,
            },
        }
        if model_type == "mask_rcnn":
            model = {
                "type": "mask_rcnn",
                "pretrained_weights": "coco",
                "backbone": "resnet50_fpn_v2",
                "trainable_backbone_layers": 3,
                "num_classes": 2,
                "input": {
                    "min_size": 800,
                    "max_size": 1333,
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225],
                },
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
            model = {
                "type": "cellpose",
                "pretrained_model": "cyto3",
                "optimizer": "Adam",
                "normalize": True,
                "scale_range": 0.5,
                "rescale": None,
                "bsize": 256,
                "nimg_per_epoch": 100,
                "min_train_masks": 0,
                "class_weights": [1.0],
            }
        else:
            raise ValueError(f"未対応のモデル種類です: {model_type}")
        common["model"] = model
        return common

    def validate_experiment_config(self, config: dict[str, Any]) -> list[dict[str, str]]:
        """入れ子設定を検証し、同一設定も警告する。"""
        results = []
        data = config.get("data", {})
        training = config.get("training", {})
        if not data.get("dataset_version"):
            results.append({"level": "error", "message": "データセット版を選択してください"})
        if not data.get("input_channels"):
            results.append({"level": "error", "message": "入力チャンネルを選択してください"})
        if int(training.get("batch_size", 1)) > 16:
            results.append(
                {"level": "warning", "message": "GPU メモリ使用量が大きくなる可能性があります"}
            )
        comparable = copy.deepcopy(config)
        comparable.get("experiment", {}).pop("id", None)
        comparable.get("experiment", {}).pop("description", None)
        for experiment in self.experiments.values():
            existing = copy.deepcopy(experiment.config.values)
            existing.get("experiment", {}).pop("id", None)
            existing.get("experiment", {}).pop("description", None)
            if comparable == existing:
                results.append(
                    {
                        "level": "warning",
                        "message": f"同一設定の実験 {experiment.experiment_id} があります",
                    }
                )
                break
        return results

    def estimate_training_items(
        self,
        dataset_version: str | None = None,
        classification: str = "all",
        quality_filter: str = "all",
        **filters: Any,
    ) -> tuple[int, int]:
        """指定条件を適用した件数を80/20で分割して返す。"""
        if dataset_version is None:
            items = self.working["train"].items
        else:
            items = self._items_for_version(dataset_version)
        filtered = (
            [item for item in items if item.usage == "train"]
            if dataset_version is None
            else [item for item in items if item.included]
        )
        if classification != "all":
            filtered = [item for item in filtered if item.classification == classification]
        if quality_filter == "good_only":
            filtered = [item for item in filtered if item.quality == "良"]
        elif quality_filter == "good_and_acceptable":
            filtered = [item for item in filtered if item.quality in {"良", "可"}]
        total = len(filtered)
        train_count = int(total * 0.8)
        return train_count, total - train_count

    def next_experiment_id(self) -> str:
        """次の実験識別子を返す。"""
        numbers = [int(key[-4:]) for key in self.experiments]
        return f"exp_{max(numbers, default=0) + 1:04d}"

    def _save_experiment(
        self, config: dict[str, Any], experiment_id: str | None, status: str
    ) -> Experiment:
        """実験設定を新規作成または更新する。"""
        saved_config = copy.deepcopy(config)
        experiment_config = saved_config.setdefault("experiment", {})
        expid = experiment_id or experiment_config.get("id") or self.next_experiment_id()
        experiment_config["id"] = expid
        data = saved_config.setdefault("data", {})
        data["used_item_ids"] = data.get("used_item_ids") or [
            item.item_id for item in self.working["train"].items[:40]
        ]
        model_type = saved_config["model"]["type"]
        existing = self.experiments.get(expid)
        now = self._now()
        if existing is None:
            experiment = Experiment(
                experiment_id=expid,
                study_id=experiment_config.get("study_id", "foam_study"),
                description=experiment_config.get("description", ""),
                model_type=model_type,
                config=ExperimentConfig(saved_config),
                status=status,
                total_epochs=int(saved_config["training"]["epochs"]),
                used_item_ids=list(data["used_item_ids"]),
                created_at=now,
            )
        else:
            experiment = existing
            experiment.study_id = experiment_config.get("study_id", "foam_study")
            experiment.description = experiment_config.get("description", "")
            experiment.model_type = model_type
            experiment.config = ExperimentConfig(saved_config)
            experiment.status = status
            experiment.total_epochs = int(saved_config["training"]["epochs"])
            experiment.used_item_ids = list(data["used_item_ids"])
        self.experiments[expid] = experiment
        self._sync_profile_usage()
        return experiment

    def save_experiment_draft(
        self, config: dict[str, Any], experiment_id: str | None = None
    ) -> Experiment:
        """下書きを新規保存または更新する。"""
        return self._save_experiment(config, experiment_id, "draft")

    def start_training(
        self, config: dict[str, Any], experiment_id: str | None = None
    ) -> Experiment:
        """実行試行を追加して学習を開始状態にする。"""
        experiment = self._save_experiment(config, experiment_id, "running")
        experiment.runs.append(RunAttempt(len(experiment.runs) + 1, self._now()))
        return experiment

    def record_epoch(
        self, experiment_id: str, epoch: int, loss: float, map_value: float | None = None
    ) -> Experiment:
        """学習履歴と設定間隔に基づくチェックポイントを記録する。"""
        experiment = self.get_experiment(experiment_id)
        experiment.current_epoch = epoch
        experiment.history.append(EpochMetrics(epoch, loss, map_value))
        checkpoint_config = experiment.config.values["checkpoint"]
        interval = max(1, int(checkpoint_config["save_every"]))
        suffix = ".pt" if experiment.model_type == "mask_rcnn" else ""
        if epoch % interval == 0:
            name = f"epoch_{epoch:03d}{suffix}"
            experiment.checkpoints = [item for item in experiment.checkpoints if item.name != name]
            experiment.checkpoints.append(Checkpoint(name, epoch, map_value, self._now()))
        best_mode = checkpoint_config["best_mode"]
        best_metrics = [item.map for item in experiment.history if item.map is not None]
        improved = map_value is not None and (
            not best_metrics[:-1]
            or (
                map_value >= max(best_metrics[:-1])
                if best_mode == "max"
                else map_value <= min(best_metrics[:-1])
            )
        )
        if checkpoint_config["save_best"] and improved:
            best_name = "best.pt" if experiment.model_type == "mask_rcnn" else "best"
            experiment.checkpoints = [
                item for item in experiment.checkpoints if item.name != best_name
            ]
            experiment.checkpoints.append(Checkpoint(best_name, epoch, map_value, self._now()))
        return experiment

    def finish_training(self, experiment_id: str, status: str = "completed") -> Experiment:
        """学習を完了・失敗・中断状態にする。"""
        experiment = self.get_experiment(experiment_id)
        experiment.status = status
        config = experiment.config.values["checkpoint"]
        if status == "completed" and config["save_last"]:
            name = "last.pt" if experiment.model_type == "mask_rcnn" else "last"
            value = experiment.history[-1].map if experiment.history else None
            experiment.checkpoints.append(
                Checkpoint(name, experiment.current_epoch, value, self._now())
            )
        if experiment.runs:
            experiment.runs[-1].finished_at = self._now()
            experiment.runs[-1].result = status
        return experiment

    def retry_experiment(self, experiment_id: str) -> Experiment:
        """同一実験へ新しい試行を追加する。"""
        experiment = self.get_experiment(experiment_id)
        experiment.status = "running"
        experiment.runs.append(RunAttempt(len(experiment.runs) + 1, self._now()))
        return experiment

    def list_experiments(self) -> list[Experiment]:
        """実験一覧を識別子順で返す。"""
        return sorted(self.experiments.values(), key=lambda experiment: experiment.experiment_id)

    def get_experiment(self, experiment_id: str) -> Experiment:
        """実験を識別子で返す。"""
        return self.experiments[experiment_id]

    def list_augmentation_profiles(self) -> list[AugmentationProfile]:
        """データ拡張プロファイルを返す。"""
        return list(self.profiles.values())

    def get_augmentation_profile(self, profile_id: str) -> AugmentationProfile:
        """データ拡張プロファイルを返す。"""
        return self.profiles[profile_id]

    def save_augmentation_profile(self, profile: AugmentationProfile) -> AugmentationProfile:
        """呼び出し元と分離した新しいプロファイル版を登録する。"""
        saved = copy.deepcopy(profile)
        number = max((int(key[-3:]) for key in self.profiles), default=0) + 1
        saved.profile_id = f"aug_v{number:03d}"
        saved.base_profile = saved.base_profile or max(self.profiles)
        saved.used_by_experiments = []
        self.profiles[saved.profile_id] = saved
        return saved

    def list_candidates(self) -> list[Candidate]:
        """比較候補一覧を返す。"""
        return list(self.candidates.values())

    def get_candidate(self, candidate_id: str) -> Candidate:
        """比較候補を取得する。"""
        return self.candidates[candidate_id]

    def list_inference_configs(self, model_type: str | None = None) -> list[InferenceConfig]:
        """推論設定一覧を返す。"""
        return [
            config
            for config in self.inference_configs.values()
            if model_type is None or config.model_type == model_type
        ]

    def create_inference_config(self, model_type: str, params: dict[str, Any]) -> InferenceConfig:
        """推論設定を採番して保存する。"""
        number = max((int(key[-3:]) for key in self.inference_configs), default=0) + 1
        config = InferenceConfig(f"infer_v{number:03d}", model_type, copy.deepcopy(params))
        self.inference_configs[config.config_id] = config
        return config

    def add_candidate(
        self, experiment_id: str, checkpoint: str, inference_config_id: str, comment: str = ""
    ) -> Candidate:
        """重複を確認して比較候補を追加する。"""
        experiment = self.get_experiment(experiment_id)
        if experiment.status != "completed":
            raise ValueError("完了した実験のみ候補に追加できます")
        if not any(item.name == checkpoint for item in experiment.checkpoints):
            raise ValueError("実験に存在しないチェックポイントです")
        for candidate in self.candidates.values():
            if (
                candidate.experiment_id == experiment_id
                and candidate.checkpoint == checkpoint
                and candidate.inference_config_id == inference_config_id
            ):
                raise ValueError(f"同じ設定は {candidate.candidate_id} として登録済みです")
        number = max((int(key[-3:]) for key in self.candidates), default=0) + 1
        candidate = Candidate(
            f"RC-{number:03d}", experiment_id, checkpoint, inference_config_id, comment=comment
        )
        self.candidates[candidate.candidate_id] = candidate
        return candidate

    def start_evaluation(
        self, candidate_ids: list[str], validation_version: str
    ) -> list[Candidate]:
        """評価対象候補を評価中状態にする。"""
        candidates = [self.candidates[candidate_id] for candidate_id in candidate_ids]
        for candidate in candidates:
            if candidate.status not in {"candidate", "evaluating"}:
                raise ValueError("候補状態のモデルのみ評価できます")
            candidate.status = "evaluating"
        return candidates

    def evaluate_candidate(
        self, candidate_id: str, validation_version: str = "val_v003"
    ) -> Evaluation:
        """全体・分類別評価値を記録して候補状態へ戻す。"""
        candidate = self.candidates[candidate_id]
        number = int(candidate_id[-3:])
        evaluation = self._evaluation(0.86 + number % 10 / 100)
        candidate.evaluations[validation_version] = evaluation
        candidate.status = "candidate"
        return evaluation

    def save_external_results(
        self,
        candidate_id: str,
        results: list[ExternalResult | dict[str, Any]],
        software: str = "",
        date: str = "",
        comment: str = "",
    ) -> Candidate:
        """外部解析値とコメントを保存する。"""
        candidate = self.candidates[candidate_id]
        candidate.external_results = [
            result if isinstance(result, ExternalResult) else ExternalResult(**result)
            for result in results
        ]
        candidate.external_software = software
        candidate.external_date = date
        candidate.comment = comment
        return candidate

    def reject_candidate(self, candidate_id: str) -> Candidate:
        """候補を非採用にする。"""
        candidate = self.candidates[candidate_id]
        candidate.status = "rejected"
        return candidate

    def release_candidate(
        self, candidate_id: str, comment: str = "", validation_version: str = "val_v003"
    ) -> ReleasedModel:
        """評価済み候補を変更不可のリリースとして登録する。"""
        candidate = self.candidates[candidate_id]
        if validation_version not in candidate.evaluations:
            raise ValueError("評価済み候補のみリリースできます")
        number = max((int(key[-3:]) for key in self.released), default=0) + 1
        experiment = self.get_experiment(candidate.experiment_id)
        inference = self.inference_configs[candidate.inference_config_id]
        model = ReleasedModel(
            model_id=f"model_{number:03d}",
            candidate_id=candidate_id,
            experiment_id=candidate.experiment_id,
            checkpoint=candidate.checkpoint,
            preprocessing_config=copy.deepcopy(experiment.config.values["model"]),
            inference_config=copy.deepcopy(inference.params),
            validation_dataset=validation_version,
            evaluation_result=candidate.evaluations[validation_version],
            released_at=self._now(),
            comment=comment or candidate.comment,
        )
        self.released[model.model_id] = model
        candidate.status = "released"
        return model

    def list_validation_items(
        self, validation_version: str = "val_v003", classification: str | None = None
    ) -> list[DataItem]:
        """検証用版の画像を分類条件付きで返す。"""
        items = self._items_for_version(validation_version)
        return [
            item
            for item in items
            if classification is None or item.classification == classification
        ]

    def list_released_models(self) -> list[ReleasedModel]:
        """リリース済みモデルを返す。"""
        return list(self.released.values())

    def get_routing(self) -> dict[str, str | None]:
        """分類ごとの現在の振り分けを返す。"""
        return self.routing.copy()

    def apply_routing(self, changes: dict[str, str | None]) -> list[RoutingHistory]:
        """振り分け変更を検証・適用し履歴を返す。"""
        records = []
        for classification, model_id in changes.items():
            if model_id is not None and model_id not in self.released:
                raise ValueError(f"未登録のリリースモデルです: {model_id}")
            before = self.routing.get(classification)
            if before != model_id:
                records.append(RoutingHistory(self._now(), classification, before, model_id))
                self.routing[classification] = model_id
        self.routing_history.extend(records)
        return records

    def list_routing_history(self) -> list[RoutingHistory]:
        """振り分け変更履歴を返す。"""
        return self.routing_history.copy()

    def resolve_model(self, classification: str | None) -> ReleasedModel | None:
        """分類から有効なモデルを返す。"""
        model_id = self.routing.get(classification or "")
        return self.released.get(model_id) if model_id else None

    def validate_excel_import(self, filename: str) -> dict[str, Any]:
        """ファイル名に応じたモック Excel 取込チェックを返す。"""
        error = "error" in filename.lower()
        changes = (
            []
            if error
            else [{"item_id": "item_000001", "field": "quality", "before": "可", "after": "良"}]
        )
        return {
            "ok": not error,
            "errors": ["必須列不足", "識別子が不明"] if error else [],
            "changes": changes,
        }

    def preview_excel_import(self, filename: str) -> dict[str, Any]:
        """Excel 取込前検査と変更プレビューを返す。"""
        return self.validate_excel_import(filename)

    def apply_excel_import(self, purpose: str, changes: list[dict[str, Any]]) -> None:
        """検証済み Excel 変更を反映する。"""
        for change in changes:
            self.update_item(
                purpose,
                change["item_id"],
                **{change["field"]: change["after"]},
            )
