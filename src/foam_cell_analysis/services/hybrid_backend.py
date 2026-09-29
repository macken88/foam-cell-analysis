"""永続的な学習サービスと既存の画面用モックをまとめる Backend。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from foam_cell_analysis.services.mock.backend import MockBackend
from foam_cell_analysis.services.training_service import TrainingService


class HybridBackend:
    """学習系 API を TrainingService、その他を MockBackend へ委譲する。"""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        process_alive=None,
        process_terminator=None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.training = TrainingService(
            self.workspace_root,
            process_alive=process_alive,
            process_terminator=process_terminator,
        )
        self.mock = MockBackend(seed_samples=False)
        self.mock._seed_validation_data()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.mock, name)

    @property
    def is_hybrid(self) -> bool:
        return True

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

    def list_dataset_versions(self, purpose=None):
        if purpose in {None, "train"}:
            return [
                version
                for version in self.mock.list_dataset_versions(purpose)
                if version.purpose != "train"
            ] + self._training_dataset_versions()
        return self.mock.list_dataset_versions(purpose)

    def _training_dataset_versions(self):
        from datetime import datetime

        from foam_cell_analysis.services.models import DatasetVersion

        versions = []
        for version in self.training.dataset_store.list_versions():
            items = self.training.dataset_store.get_items(version)
            versions.append(
                DatasetVersion(
                    version,
                    "train",
                    None,
                    datetime.now().astimezone(),
                    [item.item_id for item in items],
                    len(items),
                )
            )
        return versions

    def get_dataset_version_items(self, version):
        if version in self.training.dataset_store.list_versions():
            return self.training.dataset_store.get_items(version)
        return self.mock.get_dataset_version_items(version)

    def get_dataset_item_image(self, version, item_id, channel=None):
        if version in self.training.dataset_store.list_versions():
            return self.training.dataset_store.get_image(version, item_id, channel)
        return self.mock.get_dataset_item_image(version, item_id, channel)

    def get_dataset_item_mask(self, version, item_id, revision=None):
        if version in self.training.dataset_store.list_versions():
            return self.training.dataset_store.get_mask(version, item_id, revision)
        return self.mock.get_dataset_item_mask(version, item_id, revision)

    def add_candidate(
        self, experiment_id: str, checkpoint: str, inference_config_id: str, comment: str = ""
    ):
        """完了実験の試行スナップショットを MockBackend の比較へ登録する。"""
        if checkpoint != "final.pt":
            raise ValueError("比較候補には最終学習モデルのみ指定できます")
        snapshot = self.training.create_candidate_snapshot(experiment_id)
        return self.mock.add_candidate_from_snapshot(snapshot, inference_config_id, comment)

    def experiment_deletion_info(self, experiment_id: str, *, measure_size: bool = True):
        """学習記録の状態に加え、比較候補・リリース済みモデルからの参照を確認する。"""
        info = self.training.experiment_deletion_info(experiment_id, measure_size=measure_size)
        if info.allowed:
            reason = self.mock._experiment_reference_reason(experiment_id)
            if reason:
                info = replace(info, allowed=False, reason=reason)
        return info

    def delete_experiment(self, experiment_id: str) -> None:
        """参照がないことを確かめてから実験フォルダを削除する。"""
        reason = self.mock._experiment_reference_reason(experiment_id)
        if reason:
            raise ValueError(reason)
        self.training.delete_experiment(experiment_id)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self.mock)))


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
    "recover",
    "create_candidate_snapshot",
)


def _delegate_to_training_service(method_name):
    def delegated(self, *args, **kwargs):
        return getattr(self.training, method_name)(*args, **kwargs)

    delegated.__name__ = method_name
    return delegated


for _method_name in _TRAINING_METHODS:
    setattr(HybridBackend, _method_name, _delegate_to_training_service(_method_name))
