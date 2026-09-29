"""メモリ内で動作する画面確認用 Backend。"""

from __future__ import annotations

import copy
import hashlib
import math
from datetime import datetime, timedelta
from typing import Any, ClassVar

import numpy as np

from ...training.folds import assign_folds
from ..backend import normalization_for_weights
from ..models import (
    AugmentationProfile,
    Candidate,
    CandidateSnapshot,
    Checkpoint,
    CheckResult,
    DataItem,
    DatasetVersion,
    EpochMetrics,
    Evaluation,
    Experiment,
    ExperimentConfig,
    ExperimentDeletionInfo,
    ExternalResult,
    ImportCandidate,
    InferenceConfig,
    JobExit,
    PreparedRun,
    ReleasedModel,
    RoutingHistory,
    RunAttempt,
    TrainingOutcome,
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

    def __init__(self, seed_samples: bool = True) -> None:
        self.working: dict[str, WorkingDataset] = {}
        self.versions: list[DatasetVersion] = []
        self.experiments: dict[str, Experiment] = {}
        self.training_queue_ids: list[str] = []
        self.training_retry_reservations: dict[str, dict[str, Any]] = {}
        self._retry_reservation_seq = 0
        self.fail_training_ids: set[str] = set()
        self._prepared_queue_runs: dict[str, PreparedRun] = {}
        self._stop_requests: dict[tuple[str, int], str] = {}
        self._retry_prepared: set[str] = set()
        self.profiles: dict[str, AugmentationProfile] = {}
        self.inference_configs: dict[str, InferenceConfig] = {}
        self.candidates: dict[str, Candidate] = {}
        self.released: dict[str, ReleasedModel] = {}
        self.routing: dict[str, str | None] = {name: None for name in self.classifications}
        self.routing_history: list[RoutingHistory] = []
        if seed_samples:
            self._seed()
        else:
            empty = WorkingDataset("all", "", [])
            self.working = {purpose: empty for purpose in ("all", "train", "val")}
            self._seed_inference_configs()
        empty_dataset = WorkingDataset("all", "", [])
        items_by_id = {item.item_id: item for item in self.working.get("all", empty_dataset).items}
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
                    source_relpath=(
                        f"{purpose}/images/source_{number % 20:02d}/foam_{number:04d}.tif"
                    ),
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
        self._capture_current_attempt(exp44)
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
            [
                item.item_id
                for item in self._items_for_version(config["data"]["dataset_version"])
                if item.usage == "train"
            ]
            if status != "draft"
            else []
        )
        history = self._make_history(expid, epochs, status)
        fold_histories = (
            {
                fold: self._fold_history(history, config["data"]["seed"], fold)
                for fold in range(1, int(config["data"]["cv"]["n_folds"]) + 1)
            }
            if status != "draft"
            else {}
        )
        checkpoints: list[Checkpoint] = []
        latest_cv_save = self._now() - timedelta(days=1)
        if status == "completed":
            for fold, points in fold_histories.items():
                checkpoints.extend(
                    Checkpoint(
                        name=f"epoch_{epoch:03d}.pt",
                        epoch=epoch,
                        map=self._map_at(points, epoch),
                        saved_at=self._now() - timedelta(days=1, minutes=100 - epoch),
                        fold=fold,
                    )
                    for epoch in range(10, 101, 10)
                )
            best_epoch, best_map = max(
                ((metric.epoch, metric.map) for metric in history if metric.map is not None),
                key=lambda pair: pair[1],
            )
            for fold, points in fold_histories.items():
                checkpoints.append(
                    Checkpoint(
                        "selected.pt",
                        best_epoch,
                        self._map_at(points, best_epoch),
                        latest_cv_save + timedelta(seconds=fold),
                        fold=fold,
                    )
                )
            checkpoints.append(
                Checkpoint(
                    "final.pt",
                    best_epoch,
                    None,
                    latest_cv_save + timedelta(seconds=len(fold_histories) + 1),
                )
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
            fold_assignments=self._assign_folds(config, used) if used else {},
            fold_histories=fold_histories,
            oof_history=copy.deepcopy(history),
            selected_epoch=best_epoch if status == "completed" else None,
            oof_evaluation=None,
            phase="cross_validation" if status != "completed" else "completed",
            created_at=self._now() - timedelta(days=int(expid[-2:])),
        )
        if status == "completed":
            exp.final_history = [
                EpochMetrics(point.epoch, point.loss * 0.78, None)
                for point in history
                if point.epoch <= best_epoch
            ]
            exp.oof_evaluation = self._oof_evaluation(exp, best_map)
            exp.oof_predictions = {item_id: float(best_map) for item_id in exp.used_item_ids}
        self._capture_current_attempt(exp)
        self.experiments[expid] = exp
        return exp

    @staticmethod
    def _fold_history(history: list[EpochMetrics], seed: int, fold: int) -> list[EpochMetrics]:
        """OOF 曲線の周囲に、フォールド固有の決定的な評価差を作る。"""
        rng = np.random.default_rng(seed + fold * 104729)
        direction = -1.0 if fold % 2 else 1.0
        return [
            EpochMetrics(
                point.epoch,
                point.loss,
                max(0.0, min(1.0, point.map + direction * float(rng.uniform(0.01, 0.03))))
                if point.map is not None
                else None,
            )
            for point in history
        ]

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
        self.inference_configs["infer_v007"] = InferenceConfig(
            "infer_v007",
            "mask_rcnn",
            {"box_score_thresh": 0.35, "box_nms_thresh": 0.55, "box_detections_per_img": 100},
        )

    def _seed_validation_data(self) -> None:
        """hybrid 比較画面の正式評価に使う模擬検証版だけを作る。"""
        if self.versions:
            return
        items = self._items("val", 30, 1000)
        version = DatasetVersion(
            version="val_v003",
            purpose="val",
            parent_version=None,
            created_at=self._now(),
            item_ids=[item.item_id for item in items],
            n_images=len(items),
        )
        self.versions.append(version)
        self.working["val"] = WorkingDataset("val", version.version, items)
        self._version_items[version.version] = copy.deepcopy(items)

    def _seed_candidates_and_releases(self, exp42: Experiment, exp43: Experiment) -> None:
        """候補・リリース・振り分けの参照関係を作る。"""
        self.candidates["RC-001"] = Candidate(
            "RC-001",
            exp42.experiment_id,
            "final.pt",
            "infer_v005",
            evaluations={
                "val_v003": self._evaluation_for_items(0.91, self._items_for_version("val_v003"))
            },
            oof_evaluation=self._candidate_oof_evaluation(exp42, "infer_v005"),
            oof_experiment_id=exp42.experiment_id,
            oof_epoch=exp42.selected_epoch,
        )
        self.candidates["RC-002"] = Candidate(
            "RC-002",
            exp43.experiment_id,
            "final.pt",
            "infer_v006",
            oof_evaluation=self._candidate_oof_evaluation(exp43, "infer_v006"),
            oof_experiment_id=exp43.experiment_id,
            oof_epoch=exp43.selected_epoch,
            evaluations={
                "val_v003": self._evaluation_for_items(0.89, self._items_for_version("val_v003"))
            },
        )
        self.candidates["RC-003"] = Candidate(
            "RC-003",
            exp42.experiment_id,
            "final.pt",
            "infer_v007",
            oof_evaluation=self._candidate_oof_evaluation(exp42, "infer_v007"),
            oof_experiment_id=exp42.experiment_id,
            oof_epoch=exp42.selected_epoch,
        )
        for candidate in self.candidates.values():
            source = self.experiments[candidate.experiment_id]
            candidate.source_attempt_number = len(source.runs)
            candidate.checkpoint_reference = (
                f"試行 {candidate.source_attempt_number}/{candidate.checkpoint}"
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
                oof_evaluation=copy.deepcopy(candidate.oof_evaluation),
                released_at=now - timedelta(days=30 if model_id == "model_007" else 8),
                source_attempt_number=candidate.source_attempt_number,
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

    def _evaluation_for_items(self, score: float, items: list[DataItem]) -> Evaluation:
        """指定画像を数え、未分類を含む分類別評価を返す。"""
        labels = [*self.classifications, "未分類"]
        counts = {label: 0 for label in labels}
        for item in items:
            label = item.classification if item.classification in counts else "未分類"
            counts[label] += 1
        return Evaluation(
            score,
            {
                label: (max(0.0, score - index * 0.03), count)
                for index, (label, count) in enumerate(counts.items())
                if count or label != "未分類"
            },
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

    def get_dataset_item_image(self, version: str, item_id: str, channel: str) -> np.ndarray:
        """確定版の指定画像を返す。"""
        item = next(item for item in self._items_for_version(version) if item.item_id == item_id)
        if channel not in item.channels:
            raise ValueError(f"指定チャンネルがありません: {channel}")
        images, _ = make_sample(item.seed, channels=(channel,))
        return images[channel].copy()

    def get_dataset_item_mask(self, version: str, item_id: str, revision: str) -> np.ndarray:
        """確定版の指定ラベル画像を返す。"""
        purpose = next(
            (dataset.purpose for dataset in self.versions if dataset.version == version), "train"
        )
        return self.get_item_mask(purpose, item_id, revision)

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
        record = next((item for item in self.versions if item.version == version), None)
        if record is None:
            raise ValueError(f"データセット版 {version} がありません")
        working = self.working[record.purpose].items
        by_id = {item.item_id: item for item in working}
        return [by_id[item_id] for item_id in record.item_ids if item_id in by_id]

    def list_training_options(self) -> dict[str, list[str]]:
        """学習フォームで使用する選択肢を返す。"""
        return {
            "datasets": [
                version.version for version in self.versions if version.purpose == "train"
            ],
            "classifications": self.classifications.copy(),
            "qualities": ["良", "可", "不良"],
            "channels": ["A", "B", "C"],
            "backbones": ["resnet50_fpn_v2", "resnet101_fpn"],
            "pretrained": ["coco", "imagenet"],
            "cellpose_models": ["cpsam", "cpsam_v2"],
            "optimizers": ["SGD", "AdamW"],
            "profiles": list(self.profiles),
        }

    def default_experiment_config(self, model_type: str) -> dict[str, Any]:
        """内部キー仕様に沿ったモデル別既定設定を返す。"""
        common = {
            "experiment": {"id": None, "study_id": "foam_study", "description": ""},
            "data": {
                "dataset_version": "train_v003",
                "cv": {
                    "n_folds": 5,
                    "stratify_by_classification": True,
                    "group_by_source_folder": True,
                },
                "seed": 42,
                "classification": "all",
                "quality_filter": "all",
                "input_channels": ["A", "B", "C"],
                "used_item_ids": [],
            },
            "training": {
                "epochs": 40,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "early_stopping": {"enabled": True, "patience": 10},
            },
            "augmentation": {"profile": "aug_v003"},
            "checkpoint": {
                "save_every": 10,
                "best_metric": "oof_instance_map",
                "validation_interval": 5,
                "save_fold_models": True,
            },
        }
        if model_type == "mask_rcnn":
            model = {
                "type": "mask_rcnn",
                "pretrained_weights": "coco",
                "backbone": "resnet50_fpn_v2",
                "trainable_backbone_layers": 3,
                "num_classes": 2,
                "optimizer": "SGD",
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
            common["training"].update(
                {"batch_size": 1, "learning_rate": 0.00001, "weight_decay": 0.1}
            )
            model = {
                "type": "cellpose",
                "pretrained_model": "cpsam",
                "scale_range": 0.5,
                "bsize": 256,
                "nimg_per_epoch": None,
                "min_train_masks": 5,
                "input_channels": ["A"],
                "input": {
                    "normalization": {
                        "method": "percentile",
                        "low_percentile": 1.0,
                        "high_percentile": 99.0,
                    }
                },
            }
        else:
            raise ValueError(f"未対応のモデル種類です: {model_type}")
        common["model"] = model
        return common

    def migrate_experiment_config(self, config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """旧設定キーを除き、現行既定値と固定の OOF 選択指標を適用する。"""
        migrated = copy.deepcopy(config)
        changed = False
        data = migrated.setdefault("data", {})
        if "split_id" in data:
            del data["split_id"]
            changed = True
        model_type = migrated.get("model", {}).get("type", "mask_rcnn")
        default_cv = self.default_experiment_config(model_type)["data"]["cv"]
        current_cv = data.setdefault("cv", {})
        for key, value in default_cv.items():
            if key not in current_cv:
                current_cv[key] = copy.deepcopy(value)
                changed = True
        checkpoint = migrated.setdefault("checkpoint", {})
        for key in ("save_best", "save_last", "best_mode"):
            if key in checkpoint:
                del checkpoint[key]
                changed = True
        if "save_fold_models" not in checkpoint:
            checkpoint["save_fold_models"] = True
            changed = True
        if checkpoint.get("best_metric") != "oof_instance_map":
            checkpoint["best_metric"] = "oof_instance_map"
            changed = True
        return migrated, changed

    def validate_experiment_config(self, config: dict[str, Any]) -> list[dict[str, str]]:
        """入れ子設定を検証し、同一設定も警告する。"""
        results = []
        data = config.get("data", {})
        training = config.get("training", {})
        dataset_version = data.get("dataset_version")
        if not dataset_version:
            results.append({"level": "error", "message": "データセット版を選択してください"})
        elif not any(item.version == dataset_version for item in self.versions):
            results.append(
                {
                    "level": "error",
                    "message": f"データセット版 {dataset_version} がありません",
                }
            )
        if not data.get("input_channels"):
            results.append({"level": "error", "message": "入力チャンネルを選択してください"})
        cv = data.get("cv", {})
        folds = int(cv.get("n_folds", 5))
        if not 2 <= folds <= 10:
            results.append({"level": "error", "message": "分割数は 2〜10 にしてください"})
        if dataset_version and any(item.version == dataset_version for item in self.versions):
            try:
                total, _per_fold = self.estimate_training_items(
                    dataset_version,
                    data.get("classification", "all"),
                    data.get("quality_filter", "all"),
                    folds,
                )
                if total < folds:
                    results.append(
                        {"level": "error", "message": f"交差検証には画像が最低 {folds} 件必要です"}
                    )
                elif cv.get("group_by_source_folder", True):
                    groups = {item.source_folder for item in self._filtered_training_items(config)}
                    if len(groups) < folds:
                        results.append(
                            {
                                "level": "error",
                                "message": (
                                    f"フォルダ単位の分割には取込元フォルダが最低 {folds} 個必要です"
                                ),
                            }
                        )
            except (KeyError, ValueError) as error:
                results.append({"level": "error", "message": str(error)})
        if int(training.get("batch_size", 1)) > 16:
            results.append(
                {"level": "warning", "message": "GPU メモリ使用量が大きくなる可能性があります"}
            )
        if int(training.get("epochs", 0)) < 1:
            results.append({"level": "error", "message": "エポック数は 1 以上にしてください"})
        if float(training.get("learning_rate", 0)) <= 0:
            results.append({"level": "error", "message": "学習率は 0 より大きくしてください"})
        if float(training.get("weight_decay", 0)) < 0:
            results.append({"level": "error", "message": "重み減衰は 0 以上にしてください"})
        model = config.get("model", {})
        if model.get("type") == "mask_rcnn":
            expected_mean, expected_std = normalization_for_weights(
                model.get("pretrained_weights", "coco")
            )
            image_config = model.get("input", {})
            if (
                image_config.get("image_mean") != expected_mean
                or image_config.get("image_std") != expected_std
            ):
                results.append(
                    {
                        "level": "warning",
                        "message": "画像平均・標準偏差が事前学習済み重みの値と異なります",
                    }
                )
        comparable = copy.deepcopy(config)
        comparable.get("experiment", {}).pop("id", None)
        comparable.get("experiment", {}).pop("description", None)
        for experiment in self.experiments.values():
            if experiment.experiment_id == config.get("experiment", {}).get("id"):
                continue
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
        n_folds: int = 5,
        **filters: Any,
    ) -> tuple[int, int]:
        """指定条件を適用した学習総数と各フォールドの概算件数を返す。"""
        if dataset_version is None:
            items = self.working["train"].items
        else:
            items = self._items_for_version(dataset_version)
        filtered = (
            [item for item in items if item.usage == "train"]
            if dataset_version is None
            else [item for item in items if item.usage == "train"]
        )
        if classification != "all":
            filtered = [item for item in filtered if item.classification == classification]
        if quality_filter == "good_only":
            filtered = [item for item in filtered if item.quality == "良"]
        elif quality_filter == "good_and_acceptable":
            filtered = [item for item in filtered if item.quality in {"良", "可"}]
        total = len(filtered)
        folds = max(2, min(10, int(n_folds)))
        return total, (total + folds - 1) // folds

    def next_experiment_id(self) -> str:
        """次の実験識別子を返す。"""
        numbers = [int(key[-4:]) for key in self.experiments]
        # 削除した実験の番号は再利用しない
        numbers.append(getattr(self, "_max_deleted_experiment_number", 0))
        return f"exp_{max(numbers) + 1:04d}"

    def add_training_queue_item(self, config: dict[str, Any]) -> Experiment:
        saved = copy.deepcopy(config)
        expid = saved.setdefault("experiment", {}).get("id") or self.next_experiment_id()
        existing = self.experiments.get(expid)
        # 試行のない下書きは同じ識別子のままキューへ移す
        if existing is not None and (existing.status != "draft" or existing.runs):
            saved["experiment"]["id"] = self.next_experiment_id()
        item = self._save_experiment(saved, None, "queued")
        if item.experiment_id not in self.training_queue_ids:
            self.training_queue_ids.append(item.experiment_id)
        return item

    def add_training_retry_reservation(self, experiment_id: str) -> Experiment:
        """既存実験の次の試行を、記録を変えずにキューへ予約する。"""
        experiment = self.get_experiment(experiment_id)
        migrated_config, migrated = self.migrate_experiment_config(experiment.config.values)
        if migrated:
            raise ValueError(
                "この実験は旧形式の設定で記録されているため再試行できません。"
                "『設定を複製して新規実験』で、現在の形式に移した設定から始めてください。"
            )
        # 予約の識別子は通し番号で一意にする（試行番号は取り消しで詰まるため使わない）
        self._retry_reservation_seq += 1
        queue_id = f"retry:{experiment_id}:{self._retry_reservation_seq}"
        self.training_retry_reservations[queue_id] = {
            "experiment_id": experiment_id,
            "attempt": None,
            "status": "queued",
        }
        self.training_queue_ids.append(queue_id)
        return self.list_training_queue()[-1]

    def _expected_retry_attempt(self, queue_id: str) -> int:
        """予約の試行番号を返す。待機中はキューの並びから見込みの番号を求める。"""
        reservation = self.training_retry_reservations[queue_id]
        if reservation["attempt"] is not None:
            return reservation["attempt"]
        experiment_id = reservation["experiment_id"]
        ahead = 0
        for key in self.training_queue_ids:
            if key == queue_id:
                break
            other = self.training_retry_reservations.get(key)
            if other and other["experiment_id"] == experiment_id and other["status"] == "queued":
                ahead += 1
        return len(self.experiments[experiment_id].runs) + ahead + 1

    def list_training_queue(self) -> list[Experiment]:
        entries = []
        for key in self.training_queue_ids:
            if key in self.training_retry_reservations:
                reservation = self.training_retry_reservations[key]
                entry = copy.deepcopy(self.experiments[reservation["experiment_id"]])
                entry.status = reservation["status"]
                entry.queue_id = key
                entry.queue_is_retry = True
                entry.queue_retry_attempt = self._expected_retry_attempt(key)
                entries.append(entry)
            elif key in self.experiments:
                entry = self.experiments[key]
                entry.queue_id = key
                entries.append(entry)
        return entries

    def update_training_queue_item(self, experiment_id: str, config: dict[str, Any]) -> Experiment:
        if experiment_id in self.training_retry_reservations:
            raise ValueError("再試行の予約は編集できません")
        if experiment_id not in self.training_queue_ids:
            raise ValueError("キュー項目がありません")
        if self.experiments[experiment_id].status != "queued":
            raise ValueError("待機中のキュー項目のみ編集できます")
        saved = copy.deepcopy(config)
        saved.setdefault("experiment", {})["id"] = experiment_id
        return self._save_experiment(saved, experiment_id, "queued")

    def reorder_training_queue(self, experiment_ids: list[str]) -> list[Experiment]:
        rest = [key for key in self.training_queue_ids if key not in experiment_ids]
        ordered = [key for key in experiment_ids if key in self.training_queue_ids]
        self.training_queue_ids = ordered + rest
        return self.list_training_queue()

    def duplicate_training_queue_items(self, experiment_ids: list[str]) -> list[Experiment]:
        result = []
        for key in experiment_ids:
            source_id = self.training_retry_reservations.get(key, {}).get("experiment_id", key)
            source = self.experiments[source_id]
            config = copy.deepcopy(source.config.values)
            config["experiment"]["id"] = self.next_experiment_id()
            duplicate = self._save_experiment(config, None, "queued")
            insertion = (
                self.training_queue_ids.index(key) + 1
                if key in self.training_queue_ids
                else len(self.training_queue_ids)
            )
            self.training_queue_ids.insert(insertion, duplicate.experiment_id)
            result.append(duplicate)
        return result

    def delete_training_queue_items(self, experiment_ids: list[str]) -> None:
        for key in experiment_ids:
            if key in self.training_retry_reservations:
                reservation = self.training_retry_reservations[key]
                if reservation["status"] == "queued":
                    self.training_queue_ids.remove(key)
                    del self.training_retry_reservations[key]
                continue
            item = self.experiments.get(key)
            if item is not None and item.status == "queued":
                self.training_queue_ids.remove(key)
                del self.experiments[key]

    def clear_finished_training_queue_items(self) -> None:
        finished = {"completed", "failed", "stopped"}
        retained = []
        for key in self.training_queue_ids:
            if key in self.training_retry_reservations:
                if self.training_retry_reservations[key]["status"] in finished:
                    del self.training_retry_reservations[key]
                else:
                    retained.append(key)
            elif key in self.experiments and self.experiments[key].status not in finished:
                retained.append(key)
        self.training_queue_ids = retained

    def take_next_training_queue_item(self) -> Experiment | None:
        for key in self.training_queue_ids:
            if key in self.training_retry_reservations:
                reservation = self.training_retry_reservations[key]
                if reservation["status"] != "queued":
                    continue
                experiment = self.get_experiment(reservation["experiment_id"])
                reservation["status"] = "running"
                reservation["attempt"] = len(experiment.runs) + 1
                queue_item = copy.deepcopy(experiment)
                queue_item.queue_id = key
                queue_item.queue_is_retry = True
                queue_item.queue_retry_attempt = reservation["attempt"]
                return queue_item
            experiment = self.experiments[key]
            if experiment.status == "queued" and not any(
                issue["level"] == "error"
                for issue in self.validate_experiment_config(experiment.config.values)
            ):
                experiment.status = "running"
                return experiment
        return None

    def finish_training_queue_item(self, queue_id: str | None, status: str) -> None:
        """予約行の表示状態を実行結果へ進める。"""
        reservation = self.training_retry_reservations.get(queue_id or "")
        if reservation is not None:
            reservation["status"] = status

    def _save_experiment(
        self, config: dict[str, Any], experiment_id: str | None, status: str
    ) -> Experiment:
        """実験設定を新規作成または更新する。"""
        saved_config, _migrated = self.migrate_experiment_config(config)
        experiment_config = saved_config.setdefault("experiment", {})
        expid = experiment_id or experiment_config.get("id") or self.next_experiment_id()
        experiment_config["id"] = expid
        data = saved_config.setdefault("data", {})
        if status == "running":
            used_item_ids = [item.item_id for item in self._filtered_training_items(saved_config)]
            fold_assignments = self._assign_folds(saved_config, used_item_ids)
        else:
            used_item_ids = []
            fold_assignments = {}
        data["used_item_ids"] = used_item_ids
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
                used_item_ids=list(used_item_ids),
                fold_assignments=fold_assignments,
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
            experiment.used_item_ids = list(used_item_ids)
            experiment.fold_assignments = fold_assignments
        self.experiments[expid] = experiment
        self._sync_profile_usage()
        return experiment

    def save_experiment_draft(
        self, config: dict[str, Any], experiment_id: str | None = None
    ) -> Experiment:
        """下書きを新規保存または更新する。"""
        if experiment_id in self.experiments and self.experiments[experiment_id].runs:
            raise ValueError("試行がある実験の設定は上書きできません")
        return self._save_experiment(config, experiment_id, "draft")

    def start_training(
        self, config: dict[str, Any], experiment_id: str | None = None
    ) -> Experiment:
        """設定を保存して実行待ち状態にする。試行は prepare で作る。"""
        return self._save_experiment(config, experiment_id, "running")

    def fail_training_preparation(self, experiment_id: str) -> None:
        """試行作成前の失敗を実験へ反映する。"""
        experiment = self.experiments.get(experiment_id)
        if experiment is not None and not experiment.runs:
            experiment.status = "failed"

    def record_training_process(
        self, experiment_id: str, attempt: int, pid: int, creation_time: float
    ) -> None:
        """メモリ内モックではプロセス識別情報を保持しない。"""
        del experiment_id, attempt, pid, creation_time

    def prepare_training_run(
        self, experiment_id: str, queue_id: str | None = None, retry: bool = False
    ) -> PreparedRun:
        """新契約で試行を準備し、同じキュー要求には同じ fake job を返す。"""
        if queue_id and queue_id in self._prepared_queue_runs:
            return self._prepared_queue_runs[queue_id]
        experiment = self.get_experiment(experiment_id)
        config, legacy = self.migrate_experiment_config(experiment.config.values)
        if retry and legacy:
            raise ValueError("旧形式設定は再試行できません")
        errors = [
            item["message"]
            for item in self.validate_experiment_config(config)
            if item["level"] == "error"
        ]
        if errors:
            raise ValueError("設定エラー: " + "、".join(errors))
        if not experiment.runs:
            started = self._save_experiment(config, experiment_id, "running")
            experiment = started
        elif retry:
            if experiment_id in self._retry_prepared:
                self._retry_prepared.remove(experiment_id)
            else:
                experiment = self.retry_experiment(experiment_id)
                self._retry_prepared.discard(experiment_id)
        else:
            raise ValueError("この実験にはすでに試行があります")
        attempt = len(experiment.runs) + 1
        experiment.runs.append(RunAttempt(attempt, self._now()))
        run_id = f"{experiment_id}/attempt_{attempt:03d}"
        run = PreparedRun(run_id, f"mock://{run_id}", "mock", [], {}, fake=True)
        if queue_id:
            self._prepared_queue_runs[queue_id] = run
            self.finish_training_queue_item(queue_id, "running")
        return run

    def apply_training_event(self, experiment_id: str, event: dict[str, Any]) -> Experiment:
        """イベントをモック実験の履歴へ反映する。"""
        experiment = self.get_experiment(experiment_id)
        kind = event["type"]
        if kind == "phase":
            experiment.phase = "cross_validation" if event["phase"] == "cv" else "final_training"
            experiment.current_epoch = 0
        elif kind == "epoch":
            point = EpochMetrics(event["epoch"], float(event["loss"]), None)
            if event["phase"] == "cv":
                experiment.fold_histories.setdefault(int(event["fold"]), []).append(point)
            else:
                experiment.final_history.append(point)
                experiment.phase = "final_training"
            experiment.current_epoch = event["epoch"]
        elif kind == "val":
            points = experiment.fold_histories.setdefault(int(event["fold"]), [])
            point = next((value for value in points if value.epoch == event["epoch"]), None)
            if point:
                point.map = float(event["ap"])
        elif kind == "oof":
            losses = []
            for fold_key in event.get("per_fold", {}):
                fold = int(fold_key)
                point = next(
                    (
                        value
                        for value in experiment.fold_histories.get(fold, [])
                        if value.epoch == event["epoch"]
                    ),
                    None,
                )
                if point:
                    losses.append(point.loss)
            loss = sum(losses) / len(losses) if losses else 0.0
            experiment.oof_history.append(
                EpochMetrics(int(event["epoch"]), loss, float(event["ap"]))
            )
            experiment.history = experiment.oof_history
        elif kind == "selected":
            per_class = {
                label: (value[0], int(value[1])) for label, value in event["per_class"].items()
            }
            experiment.selected_epoch = int(event["epoch"])
            experiment.oof_evaluation = Evaluation(float(event["ap"]), per_class)
        elif kind == "checkpoint":
            name = event["path"].replace("\\", "/").rsplit("/", maxsplit=1)[-1]
            fold = event.get("fold")
            epoch = int(event["epoch"])
            score = (
                next(
                    (
                        value.map
                        for value in experiment.fold_histories.get(int(fold), [])
                        if value.epoch == epoch
                    ),
                    None,
                )
                if fold is not None
                else None
            )
            if not any(item.name == name and item.fold == fold for item in experiment.checkpoints):
                experiment.checkpoints.append(
                    Checkpoint(name, epoch, score, self._now(), fold=fold)
                )
        self._capture_current_attempt(experiment, detach=False)
        return experiment

    def request_training_stop(self, experiment_id: str, attempt: int, reason: str) -> None:
        """中断要求をモック内部に記録する。"""
        if reason not in {"user_stop", "app_exit"}:
            raise ValueError("中断理由は user_stop または app_exit です")
        self._stop_requests[(experiment_id, attempt)] = reason

    def conclude_training_run(
        self, experiment_id: str, attempt: int, job_exit: JobExit | None = None
    ) -> TrainingOutcome:
        """モックの失敗注入と停止要求を試行の終端状態にする。"""
        experiment = self.get_experiment(experiment_id)
        reason = self._stop_requests.get((experiment_id, attempt))
        if reason:
            status = "stopped"
        elif experiment_id in self.fail_training_ids or (job_exit and job_exit.start_failed):
            status, reason = (
                "failed",
                "start_failed" if job_exit and job_exit.start_failed else "error",
            )
        else:
            status = "completed"
        experiment.status = status
        if experiment.runs and len(experiment.runs) >= attempt:
            experiment.runs[attempt - 1].result = status
            experiment.runs[attempt - 1].finished_at = self._now()
        queue_id = next(
            (
                key
                for key, run in self._prepared_queue_runs.items()
                if run.run_id == f"{experiment_id}/attempt_{attempt:03d}"
            ),
            None,
        )
        if queue_id:
            self.finish_training_queue_item(queue_id, status)
        return TrainingOutcome(experiment_id, attempt, queue_id, status, reason)

    def _capture_current_attempt(self, experiment: Experiment, *, detach: bool = True) -> None:
        """実験の現在結果を最新の実行試行へスナップショットする。"""
        if not experiment.runs:
            return
        run = experiment.runs[-1]
        snapshot = copy.deepcopy if detach else lambda value: value
        run.history = snapshot(experiment.history)
        run.fold_histories = snapshot(experiment.fold_histories)
        run.oof_history = snapshot(experiment.oof_history)
        run.final_history = snapshot(experiment.final_history)
        run.checkpoints = snapshot(experiment.checkpoints)
        run.used_item_ids = list(experiment.used_item_ids)
        run.fold_assignments = dict(experiment.fold_assignments)
        run.selected_epoch = experiment.selected_epoch
        run.oof_evaluation = snapshot(experiment.oof_evaluation)
        run.oof_predictions = dict(experiment.oof_predictions)
        run.total_epochs = experiment.total_epochs
        if run.selected_epoch is None and run.oof_history:
            selected = max(run.oof_history, key=lambda point: point.map or 0.0)
            run.selected_epoch = selected.epoch
        if run.oof_evaluation is None and run.selected_epoch is not None:
            selected = next(
                (point for point in run.oof_history if point.epoch == run.selected_epoch), None
            )
            if selected is not None and selected.map is not None:
                run.oof_evaluation = self._oof_evaluation(experiment, selected.map)

    def _assign_folds(self, config: dict[str, Any], item_ids: list[str]) -> dict[str, int]:
        """共通の group_greedy_v1 実装へ fold 割り当てを委譲する。"""
        cv = config.get("data", {}).get("cv", {})
        by_id = {
            item.item_id: item
            for item in self._items_for_version(config["data"]["dataset_version"])
        }
        items = [by_id[item_id] for item_id in sorted(item_ids) if item_id in by_id]
        return assign_folds(
            items,
            int(cv.get("n_folds", 5)),
            int(config.get("data", {}).get("seed", 42)),
            group_by_source_folder=cv.get("group_by_source_folder", True),
            stratify_by_classification=cv.get("stratify_by_classification", True),
        )

    def _filtered_training_items(self, config: dict[str, Any]) -> list[DataItem]:
        """学習版、分類、品質条件を適用した画像を返す。"""
        data = config.get("data", {})
        items = [
            item
            for item in self._items_for_version(data["dataset_version"])
            if item.usage == "train"
        ]
        if data.get("classification", "all") != "all":
            items = [item for item in items if item.classification == data["classification"]]
        quality = data.get("quality_filter", "all")
        if quality == "good_only":
            items = [item for item in items if item.quality == "良"]
        elif quality == "good_and_acceptable":
            items = [item for item in items if item.quality in {"良", "可"}]
        return items

    def finish_training(self, experiment_id: str, status: str = "completed") -> Experiment:
        """学習を完了・失敗・中断状態にする。"""
        experiment = self.get_experiment(experiment_id)
        experiment.status = status
        if status == "completed":
            if experiment.oof_history:
                selected = next(
                    (
                        point
                        for point in experiment.oof_history
                        if point.epoch == experiment.selected_epoch
                    ),
                    max(experiment.oof_history, key=lambda point: point.map or 0),
                )
                experiment.oof_evaluation = self._oof_evaluation(experiment, selected.map or 0.0)
                rng = np.random.default_rng(int(experiment.config.values["data"].get("seed", 42)))
                experiment.oof_predictions = {
                    item_id: max(
                        0.0,
                        min(1.0, (selected.map or 0.0) + float(rng.normal(0.0, 0.015))),
                    )
                    for item_id in experiment.used_item_ids
                }
                if experiment.config.values["checkpoint"].get("save_fold_models", True):
                    for fold, points in experiment.fold_histories.items():
                        metric = next(
                            (point.map for point in points if point.epoch == selected.epoch), None
                        )
                        if metric is not None:
                            experiment.checkpoints.append(
                                Checkpoint(
                                    "selected.pt",
                                    selected.epoch,
                                    metric,
                                    self._now(),
                                    fold=fold,
                                )
                            )
            name = "final.pt"
            value = None
            experiment.checkpoints.append(
                Checkpoint(
                    name, experiment.selected_epoch or experiment.current_epoch, value, self._now()
                )
            )
            experiment.phase = "completed"
        if experiment.runs:
            experiment.runs[-1].finished_at = self._now()
            experiment.runs[-1].result = status
            self._capture_current_attempt(experiment)
        return experiment

    def _oof_evaluation(self, experiment: Experiment, score: float) -> Evaluation:
        """実使用画像の分類別件数を持つ OOF 評価を返す。"""
        data_version = experiment.config.values["data"]["dataset_version"]
        items = {item.item_id: item for item in self._items_for_version(data_version)}
        labels = [*self.classifications, "未分類"]
        counts = {label: 0 for label in labels}
        for item_id in experiment.used_item_ids:
            item = items.get(item_id)
            classification = item.classification if item else None
            label = classification if classification in counts else "未分類"
            counts[label] += 1
        return Evaluation(
            score,
            {
                label: (max(0.0, score - index * 0.03), count)
                for index, (label, count) in enumerate(counts.items())
                if count or label != "未分類"
            },
        )

    def retry_experiment(self, experiment_id: str) -> Experiment:
        """同一実験へ新しい試行を追加する。"""
        experiment = self.get_experiment(experiment_id)
        migrated_config, migrated = self.migrate_experiment_config(experiment.config.values)
        if migrated:
            raise ValueError(
                "この実験は旧形式の設定で記録されているため再試行できません。"
                "『設定を複製して新規実験』で、現在の形式に移した設定から始めてください。"
            )
        self._capture_current_attempt(experiment)
        used_item_ids = [item.item_id for item in self._filtered_training_items(migrated_config)]
        fold_assignments = self._assign_folds(migrated_config, used_item_ids)
        experiment.used_item_ids = used_item_ids
        experiment.fold_assignments = fold_assignments
        experiment.total_epochs = int(migrated_config["training"]["epochs"])
        experiment.status = "running"
        experiment.phase = "cross_validation"
        experiment.current_epoch = 0
        experiment.fold_histories.clear()
        experiment.oof_history.clear()
        experiment.final_history.clear()
        experiment.history.clear()
        experiment.checkpoints.clear()
        experiment.selected_epoch = None
        experiment.oof_evaluation = None
        experiment.oof_predictions.clear()
        self._retry_prepared.add(experiment_id)
        return experiment

    def _experiment_reference_reason(self, experiment_id: str) -> str:
        """実験を参照するリリース済みモデル・比較候補があれば、削除できない理由を返す。"""
        for model in self.released.values():
            if model.experiment_id == experiment_id:
                return (
                    f"リリース済みモデル {model.model_id} がこの実験を参照しているため、"
                    "削除できません"
                )
        for candidate in self.candidates.values():
            referenced = candidate.experiment_id == experiment_id or (
                candidate.snapshot is not None and candidate.snapshot.experiment_id == experiment_id
            )
            if referenced and candidate.status != "rejected":
                return (
                    f"比較候補 {candidate.candidate_id} がこの実験を参照しています。"
                    "先に候補を非採用・削除してください"
                )
        return ""

    def experiment_deletion_info(
        self, experiment_id: str, *, measure_size: bool = True
    ) -> ExperimentDeletionInfo:
        """削除の可否を返す。メモリ内モックは容量を持たない。"""
        del measure_size
        experiment = self.get_experiment(experiment_id)
        running_reservation = any(
            reservation["experiment_id"] == experiment_id and reservation["status"] == "running"
            for reservation in self.training_retry_reservations.values()
        )
        if experiment.status == "running" or running_reservation:
            reason = "学習中の実験は削除できません。学習を停止してから削除してください"
        elif experiment.status == "queued":
            reason = "キューで待機中の実験は、学習キューの表から削除してください"
        elif experiment.status not in {"draft", "stopped", "failed", "completed"}:
            reason = "この状態の実験は削除できません"
        else:
            reason = self._experiment_reference_reason(experiment_id)
        return ExperimentDeletionInfo(experiment_id, not reason, reason, len(experiment.runs))

    def delete_experiment(self, experiment_id: str) -> None:
        """実験とそのキュー行をメモリから削除する。"""
        info = self.experiment_deletion_info(experiment_id)
        if not info.allowed:
            raise ValueError(info.reason)
        del self.experiments[experiment_id]
        self._max_deleted_experiment_number = max(
            getattr(self, "_max_deleted_experiment_number", 0), int(experiment_id[-4:])
        )
        for key, reservation in list(self.training_retry_reservations.items()):
            if reservation["experiment_id"] == experiment_id:
                del self.training_retry_reservations[key]
        self.training_queue_ids = [
            key
            for key in self.training_queue_ids
            if key in self.training_retry_reservations or key in self.experiments
        ]
        self._retry_prepared.discard(experiment_id)
        self._sync_profile_usage()

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
        if checkpoint != "final.pt":
            raise ValueError("比較候補には最終学習モデルのみ指定できます")
        if not any(item.name == checkpoint for item in experiment.checkpoints):
            raise ValueError("最終学習モデルが実験にありません")
        for candidate in self.candidates.values():
            if (
                candidate.experiment_id == experiment_id
                and candidate.checkpoint == checkpoint
                and candidate.source_attempt_number == len(experiment.runs)
                and candidate.inference_config_id == inference_config_id
            ):
                raise ValueError(f"同じ試行の設定は {candidate.candidate_id} として登録済みです")
        number = max((int(key[-3:]) for key in self.candidates), default=0) + 1
        oof = self._candidate_oof_evaluation(experiment, inference_config_id)
        candidate = Candidate(
            f"RC-{number:03d}",
            experiment_id,
            checkpoint,
            inference_config_id,
            oof_evaluation=oof,
            oof_experiment_id=experiment_id,
            oof_epoch=experiment.selected_epoch,
            comment=comment,
            source_attempt_number=len(experiment.runs),
            checkpoint_reference=f"試行 {len(experiment.runs)}/final.pt",
        )
        self.candidates[candidate.candidate_id] = candidate
        return candidate

    def add_candidate_from_snapshot(
        self, snapshot: CandidateSnapshot, inference_config_id: str, comment: str = ""
    ) -> Candidate:
        """実測 OOF と実験設定を保持したスナップショットを比較候補へ加える。"""
        inference = self.inference_configs.get(inference_config_id)
        model_config = snapshot.experiment_config.get("model", {})
        model_type = model_config.get("type")
        if inference is None or inference.model_type != model_type:
            raise ValueError("実験のモデル種類に合う推論設定を選択してください")
        if not snapshot.checkpoint_path or snapshot.checkpoint_path.startswith(("/", "\\")):
            raise ValueError("候補のチェックポイント参照が不正です")
        if any(
            candidate.experiment_id == snapshot.experiment_id
            and candidate.source_attempt_number == snapshot.attempt
            and candidate.inference_config_id == inference_config_id
            for candidate in self.candidates.values()
        ):
            raise ValueError("この試行と推論設定の候補は登録済みです")
        raw_oof = copy.deepcopy(snapshot.oof_evaluation)
        per_class = {
            str(name): (float(values[0]), int(values[1]))
            for name, values in raw_oof.get("per_class", {}).items()
        }
        evaluation = Evaluation(float(raw_oof["ap"]), per_class)
        number = max((int(key[-3:]) for key in self.candidates), default=0) + 1
        candidate = Candidate(
            candidate_id=f"RC-{number:03d}",
            experiment_id=snapshot.experiment_id,
            checkpoint="final.pt",
            inference_config_id=inference_config_id,
            oof_evaluation=evaluation,
            oof_experiment_id=snapshot.experiment_id,
            oof_epoch=snapshot.selected_epoch,
            comment=comment,
            source_attempt_number=snapshot.attempt,
            checkpoint_reference=f"試行 {snapshot.attempt}/final.pt",
            snapshot=copy.deepcopy(snapshot),
        )
        self.candidates[candidate.candidate_id] = candidate
        return candidate

    def _candidate_oof_evaluation(
        self, experiment: Experiment, inference_config_id: str
    ) -> Evaluation:
        """保存済み OOF 結果を推論設定に応じて決定的に再評価する。"""
        base_score = (
            experiment.oof_evaluation.overall_map
            if experiment.oof_evaluation
            else max((point.map or 0 for point in experiment.oof_history), default=0.0)
        )
        base = self._oof_evaluation(experiment, base_score)
        params = self.inference_configs[inference_config_id].params
        param_bytes = repr(sorted(params.items())).encode("utf-8")
        stable_jitter = (
            (sum((index + 1) * value for index, value in enumerate(param_bytes)) % 17) - 8
        ) * 0.003
        return Evaluation(
            max(0.0, min(1.0, base.overall_map + stable_jitter)),
            {
                label: (max(0.0, min(1.0, score + stable_jitter)), count)
                for label, (score, count) in base.per_class.items()
            },
        )

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
        items = self._items_for_version(validation_version)
        evaluation = self._evaluation_for_items(0.86 + number % 10 / 100, items)
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
        experiment = (
            None if candidate.snapshot is not None else self.get_experiment(candidate.experiment_id)
        )
        inference = self.inference_configs[candidate.inference_config_id]
        preprocessing = (
            candidate.snapshot.experiment_config["model"]
            if candidate.snapshot is not None
            else experiment.config.values["model"]
        )
        model = ReleasedModel(
            model_id=f"model_{number:03d}",
            candidate_id=candidate_id,
            experiment_id=candidate.experiment_id,
            checkpoint=candidate.checkpoint,
            preprocessing_config=copy.deepcopy(preprocessing),
            inference_config=copy.deepcopy(inference.params),
            validation_dataset=validation_version,
            evaluation_result=candidate.evaluations[validation_version],
            oof_evaluation=copy.deepcopy(candidate.oof_evaluation),
            released_at=self._now(),
            comment=comment or candidate.comment,
            source_attempt_number=candidate.source_attempt_number,
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
