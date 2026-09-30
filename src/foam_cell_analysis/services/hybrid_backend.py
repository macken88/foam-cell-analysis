"""永続的な学習・比較サービスと、既存の画面用モックをまとめる Backend。

委譲先（比較・評価設計 2.2）:
- 学習系 → TrainingService
- 推論設定・候補・評価・予測・外部解析・抽出結果出力・リリース・振り分け → ComparisonService
- 検証版の一覧・画像 → TrainingService.dataset_store
- 本番推論（get_inference_result）とデータ準備 → MockBackend

比較系の API は __getattr__ に頼らず、すべてこのクラスに明示的に定義する
（MockBackend に流れると、実データと模擬データが混ざるため）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from foam_cell_analysis.services.backend import MaskExportParams
from foam_cell_analysis.services.comparison_service import ComparisonService
from foam_cell_analysis.services.mock.backend import MockBackend
from foam_cell_analysis.services.models import (
    DatasetVersion,
)
from foam_cell_analysis.services.training_service import TrainingService

logger = logging.getLogger(__name__)


# MockBackend にあるが hybrid では使わせない名前（比較系の旧 API・模擬データの作成口）
_BLOCKED_MOCK_NAMES = frozenset(
    {
        "_experiment_reference_reason",
        "_seed_validation_data",
        "_seed_inference_configs",
        "_seed_candidates_and_releases",
    }
)


class HybridBackend:
    """学習系を TrainingService、比較系を ComparisonService、その他を MockBackend へ委譲する。"""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        process_alive=None,
        process_terminator=None,
        is_evaluation_active: Callable[[str], bool] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.training = TrainingService(
            self.workspace_root,
            process_alive=process_alive,
            process_terminator=process_terminator,
        )
        self._evaluation_active: Callable[[str], bool] = is_evaluation_active or (
            lambda _candidate_id: False
        )
        self.comparison = ComparisonService(
            self.workspace_root,
            self.training,
            is_evaluation_active=self._is_evaluation_active,
            process_alive=process_alive,
            process_terminator=process_terminator,
        )
        # データ準備と本番推論の画面だけに使う。比較候補・推論設定・リリース・振り分け・
        # 検証版の模擬データは作らない（2.2）
        self.mock = MockBackend(seed_samples=False)

    def __getattr__(self, name: str) -> Any:
        if name in _BLOCKED_MOCK_NAMES or name in {"training", "comparison", "mock"}:
            raise AttributeError(name)
        return getattr(self.mock, name)

    def __dir__(self):
        return sorted((set(super().__dir__()) | set(dir(self.mock))) - _BLOCKED_MOCK_NAMES)

    @property
    def is_hybrid(self) -> bool:
        return True

    # ---- 起動時の復旧（学習 5.4・比較 7.6・13.3） ----

    def recover(self):
        """学習、比較（一時ファイル・リリース）、評価の順に起動時の復旧を行う。"""
        outcomes = self.training.recover()
        self.comparison.recover()
        self.comparison.recover_evaluations()
        return outcomes

    @property
    def recovery_blockers(self) -> list[str]:
        """復旧で終了できなかったプロセスの説明（新しい計算を始める前に利用者へ示す）。"""
        return [
            *getattr(self.training, "recovery_blockers", []),
            *getattr(self.comparison, "recovery_blockers", []),
        ]

    # ---- 評価中かどうか（EvaluationRunner から受け取る） ----

    def set_evaluation_activity(self, is_evaluation_active: Callable[[str], bool]) -> None:
        """候補が評価中・評価待ちかを答える関数を差し替える。"""
        self._evaluation_active = is_evaluation_active

    def _is_evaluation_active(self, candidate_id: str) -> bool:
        return bool(self._evaluation_active(candidate_id))

    # ---- 学習フォームの選択肢 ----

    def list_training_options(self) -> dict[str, list[str]]:
        versions = self.training.dataset_store.list_versions()
        items = [
            item for version in versions for item in self.training.dataset_store.get_items(version)
        ]
        options = self.mock.list_training_options()
        options["datasets"] = versions
        options["classifications"] = sorted(
            {item.classification for item in items if item.classification}
        )
        options["channels"] = sorted({channel for item in items for channel in item.channels})
        options["cellpose_models"] = ["cpsam", "cpsam_v2"]
        options["optimizers"] = ["SGD", "AdamW"]
        options["profiles"] = [
            profile.profile_id for profile in self.training.list_augmentation_profiles()
        ]
        return options

    # ---- データセット版（学習・検証は DatasetStore。3 章） ----

    def _store_purpose(self, version: str) -> str | None:
        """DatasetStore にある版なら用途（train / val）を返す。"""
        store = self.training.dataset_store
        for purpose in ("train", "val"):
            if version in store.list_versions(purpose):
                return purpose
        return None

    def _store_versions(self, purpose: str) -> list[DatasetVersion]:
        store = self.training.dataset_store
        versions = []
        for version in store.list_versions(purpose):
            try:
                items = store.get_items(version, expected_purpose=purpose)
            except (OSError, ValueError, KeyError) as error:
                logger.warning("データセット %s を読めません: %s", version, error)
                continue
            info = self._dataset_info(version)
            created = self._parse_time(info.get("created_at"))
            base = info.get("base_validation_version")
            versions.append(
                DatasetVersion(
                    version,
                    purpose,
                    info.get("parent_version") or None,
                    created,
                    [item.item_id for item in items],
                    len(items),
                    comment=str(info.get("comment") or ""),
                    base_validation_version=base if isinstance(base, str) and base else None,
                )
            )
        return versions

    def _dataset_info(self, version: str) -> dict[str, Any]:
        path = self.training.dataset_store.datasets_root / version / "dataset_info.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _parse_time(value: Any) -> datetime:
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
                return parsed if parsed.tzinfo else parsed.astimezone()
            except ValueError:
                pass
        return datetime.now().astimezone()

    def list_dataset_versions(self, purpose=None):
        """学習・検証の版は DatasetStore、それ以外（データ準備のモック版）は MockBackend。"""
        if purpose in {"train", "val"}:
            return self._store_versions(purpose)
        others = [
            version
            for version in self.mock.list_dataset_versions(purpose)
            if version.purpose not in {"train", "val"}
        ]
        if purpose is None:
            return others + self._store_versions("train") + self._store_versions("val")
        return others

    def list_validation_versions(self):
        return self._store_versions("val")

    def get_dataset_version_items(self, version):
        purpose = self._store_purpose(version)
        if purpose is not None:
            return self.training.dataset_store.get_items(version, expected_purpose=purpose)
        return self.mock.get_dataset_version_items(version)

    def list_validation_items(self, validation_version, classification=None):
        """検証版の全画像（item_id 昇順）を分類条件付きで返す。"""
        if not validation_version:
            return []
        items = self.training.dataset_store.select_evaluation_items(validation_version)
        return [
            item
            for item in items
            if classification is None or item.classification == classification
        ]

    def get_dataset_item_image(self, version, item_id, channel=None):
        if self._store_purpose(version) is not None:
            return self.training.dataset_store.get_image(version, item_id, channel)
        return self.mock.get_dataset_item_image(version, item_id, channel)

    def get_dataset_item_mask(self, version, item_id, revision=None):
        if self._store_purpose(version) is not None:
            return self.training.dataset_store.get_mask(version, item_id, revision)
        return self.mock.get_dataset_item_mask(version, item_id, revision)

    # ---- 推論設定・候補（5 章） ----

    def list_candidates(self):
        return self.comparison.list_candidates()

    def get_candidate(self, candidate_id):
        return self.comparison.get_candidate(candidate_id)

    def list_inference_configs(self, model_type=None):
        return self.comparison.list_inference_configs(model_type)

    def default_inference_params(self, model_type):
        return self.comparison.default_inference_params(model_type)

    def create_inference_config(self, model_type, params):
        return self.comparison.create_inference_config(model_type, params)

    def add_candidate(self, experiment_id, attempt, inference_config_id, comment=""):
        """試行を明示して比較候補を作る（5.3）。"""
        return self.comparison.add_candidate(experiment_id, attempt, inference_config_id, comment)

    def reject_candidate(self, candidate_id):
        return self.comparison.reject_candidate(candidate_id)

    # ---- 評価（7 章） ----

    def prepare_evaluation_run(self, candidate_id):
        return self.comparison.prepare_evaluation_run(candidate_id)

    def record_evaluation_process(self, candidate_id, evaluation_id, pid, creation_time):
        return self.comparison.record_evaluation_process(
            candidate_id, evaluation_id, pid, creation_time
        )

    def request_evaluation_stop(self, candidate_id, evaluation_id, reason):
        return self.comparison.request_evaluation_stop(candidate_id, evaluation_id, reason)

    def apply_evaluation_event(self, candidate_id, evaluation_id, event):
        return self.comparison.apply_evaluation_event(candidate_id, evaluation_id, event)

    def conclude_evaluation_run(self, candidate_id, evaluation_id, job_exit=None):
        return self.comparison.conclude_evaluation_run(candidate_id, evaluation_id, job_exit)

    def get_evaluation_progress(self, candidate_id):
        return self.comparison.get_evaluation_progress(candidate_id)

    def get_candidate_evaluation(self, candidate_id, validation_version=None):
        return self.comparison.get_candidate_evaluation(candidate_id, validation_version)

    def list_candidate_evaluations(self, candidate_id, validation_version=None):
        return self.comparison.list_candidate_evaluations(candidate_id, validation_version)

    def get_candidate_prediction(self, candidate_id, evaluation_id, item_id):
        """評価で保存した予測を返す。"""
        return self.comparison.get_candidate_prediction(candidate_id, evaluation_id, item_id)

    # ---- 外部解析（9.4） ----

    def list_external_results(self, candidate_id):
        return self.comparison.list_external_results(candidate_id)

    def save_external_analysis(self, candidate_id, evaluation_id, values, **kwargs):
        """外部解析（画像ごとの円相当径の中央値）を評価 ID 付きで保存する。"""
        return self.comparison.save_external_analysis(candidate_id, evaluation_id, values, **kwargs)

    # ---- 抽出結果出力（12 章） ----

    def _export_source(self, candidate_id: str, evaluation_id: str, item_ids: list[str] | None):
        """評価の記録から ExportSource を作る。未完了・破損・対象外の画像は ValueError。"""
        from foam_cell_analysis.inference.mask_export import ExportSource
        from foam_cell_analysis.services.comparison_service import resolve_recorded_path
        from foam_cell_analysis.training.protocol import read_json

        record = None
        for version in self.training.dataset_store.list_versions("val"):
            for item in self.comparison.list_candidate_evaluations(candidate_id, version):
                if item.evaluation_id == evaluation_id:
                    record = item
        if record is None:
            raise ValueError(f"評価がありません: {candidate_id} / {evaluation_id}")
        if record.status != "completed":
            raise ValueError(f"{candidate_id} は評価が完了していないため出力できません")
        if record.broken:
            raise ValueError("評価結果のファイルが壊れています。再評価してください")
        run_dir = self.comparison.evaluation_run_dir(candidate_id, evaluation_id)
        try:
            spec = read_json(run_dir / "run_spec.json")
            result = read_json(run_dir / "result.json")
            targets = list(spec["validation"]["item_ids"])
            predictions = result["predictions"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError("評価結果のファイルが壊れています。再評価してください") from error
        if item_ids is not None:
            missing = [item_id for item_id in item_ids if item_id not in targets]
            if missing:
                raise ValueError(
                    f"{candidate_id} の評価に含まれない画像があります: " + "、".join(missing[:5])
                )
            targets = [item_id for item_id in targets if item_id in set(item_ids)]
        entries = []
        for item_id in targets:
            try:
                entry = predictions[item_id]
                path = resolve_recorded_path(run_dir, entry["path"])
                entries.append((item_id, path, int(entry["bytes"]), str(entry["sha256"]).lower()))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("評価結果のファイルが壊れています。再評価してください") from error
        return ExportSource(
            candidate_id,
            evaluation_id,
            record.validation_version,
            record.input_fingerprint or "",
            entries,
        )

    def export_particle_masks(self, request, progress, is_cancelled):
        """採用した評価の予測から抽出結果を出力する（ワーカースレッドから呼ぶ）。"""
        from foam_cell_analysis.inference.mask_export import ExportRequest, run_export

        params = MaskExportParams.from_value(request)
        if not params.selections:
            raise ValueError("出力する候補を選択してください")
        sources = [
            self._export_source(candidate_id, evaluation_id, params.item_ids)
            for candidate_id, evaluation_id in params.selections
        ]
        export_request = ExportRequest(
            Path(params.output_parent),
            sources,
            set(params.contents),
            params.file_format,
            self.comparison.app_version,
        )
        return run_export(export_request, progress, is_cancelled)

    # ---- リリース（13 章） ----

    def release_candidate(self, candidate_id, evaluation_id, comment=""):
        """指定した評価で候補をリリースする（13.1）。"""
        return self.comparison.release_candidate(candidate_id, evaluation_id, comment)

    def list_released_models(self):
        return self.comparison.list_released_models()

    # ---- 振り分け（14 章） ----

    def get_routing_state(self):
        return self.comparison.get_routing_state()

    def get_routing(self):
        return self.comparison.get_routing()

    def apply_routing(self, changes, *, expected_revision):
        """全変更を検証してから一括で適用する。"""
        return self.comparison.apply_routing(changes, expected_revision=expected_revision)

    def list_routing_history(self):
        return self.comparison.list_routing_history()

    def list_routing_classifications(self):
        return self.comparison.list_routing_classifications()

    def resolve_model(self, classification):
        return self.comparison.resolve_model(classification)

    # ---- 本番推論（2.4。合成値のまま） ----

    def get_inference_result(self, filename, model_id):
        return self.mock.get_inference_result(filename, model_id)

    # ---- 成果物の整理（19 章）。final の参照確認は ComparisonService ----

    def artifact_cleanup_plan(self, experiment_ids):
        return self.training.artifact_cleanup_plan(
            experiment_ids, protected_final=self.comparison.protected_final_reason
        )

    def estimate_freed_bytes(self, experiment_ids, categories):
        return self.training.estimate_freed_bytes(
            experiment_ids, categories, protected_final=self.comparison.protected_final_reason
        )

    def prune_artifacts(self, experiment_ids, categories):
        return self.training.prune_artifacts(
            experiment_ids, categories, protected_final=self.comparison.protected_final_reason
        )

    def pruned_paths(self, experiment_id, attempt):
        return self.training.pruned_paths(experiment_id, attempt)

    # ---- 実験の削除（13.5） ----

    def experiment_deletion_info(self, experiment_id: str, *, measure_size: bool = True):
        """学習記録の状態に加え、比較候補・リリースからの参照を確認する。"""
        info = self.training.experiment_deletion_info(experiment_id, measure_size=measure_size)
        if info.allowed:
            reason = self.comparison.experiment_reference_reason(experiment_id)
            if reason:
                info = replace(info, allowed=False, reason=reason)
        return info

    def delete_experiment(self, experiment_id: str) -> None:
        """参照がないことを確かめてから実験フォルダを削除する。"""
        reason = self.comparison.experiment_reference_reason(experiment_id)
        if reason:
            raise ValueError(reason)
        self.training.delete_experiment(experiment_id)


_TRAINING_METHODS = (
    "default_experiment_config",
    "migrate_experiment_config",
    "validate_experiment_config",
    "estimate_training_items",
    "next_experiment_id",
    "add_training_queue_item",
    "add_training_retry_reservation",
    "list_training_queue",
    "update_training_queue_item",
    "reorder_training_queue",
    "duplicate_training_queue_items",
    "delete_training_queue_items",
    "clear_finished_training_queue_items",
    "take_next_training_queue_item",
    "finish_training_queue_item",
    "save_experiment_draft",
    "start_training",
    "retry_experiment",
    "prepare_training_run",
    "record_training_process",
    "fail_training_preparation",
    "apply_training_event",
    "request_training_stop",
    "conclude_training_run",
    "list_experiments",
    "get_experiment",
    "list_augmentation_profiles",
    "get_augmentation_profile",
    "save_augmentation_profile",
    "create_candidate_snapshot",
)

# HybridBackend 自身に定義していなければならない比較・評価・出力・整理の API（2.2）
COMPARISON_METHODS = (
    "list_candidates",
    "get_candidate",
    "list_inference_configs",
    "default_inference_params",
    "create_inference_config",
    "add_candidate",
    "reject_candidate",
    "set_evaluation_activity",
    "prepare_evaluation_run",
    "record_evaluation_process",
    "request_evaluation_stop",
    "apply_evaluation_event",
    "conclude_evaluation_run",
    "get_evaluation_progress",
    "get_candidate_evaluation",
    "list_candidate_evaluations",
    "get_candidate_prediction",
    "list_external_results",
    "save_external_analysis",
    "export_particle_masks",
    "release_candidate",
    "list_released_models",
    "get_routing_state",
    "get_routing",
    "apply_routing",
    "list_routing_history",
    "list_routing_classifications",
    "resolve_model",
    "list_validation_versions",
    "list_validation_items",
    "list_dataset_versions",
    "get_dataset_item_image",
    "get_dataset_item_mask",
    "artifact_cleanup_plan",
    "estimate_freed_bytes",
    "prune_artifacts",
    "pruned_paths",
    "experiment_deletion_info",
    "delete_experiment",
    "recover",
)


def _delegate_to_training_service(method_name):
    def delegated(self, *args, **kwargs):
        return getattr(self.training, method_name)(*args, **kwargs)

    delegated.__name__ = method_name
    return delegated


for _method_name in _TRAINING_METHODS:
    setattr(HybridBackend, _method_name, _delegate_to_training_service(_method_name))
