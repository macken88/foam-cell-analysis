"""HybridBackend の委譲（比較・評価設計 2.2・13.5・19.4）。"""

import pytest

from foam_cell_analysis.services.backend import MaskExportParams
from foam_cell_analysis.services.hybrid_backend import COMPARISON_METHODS, HybridBackend
from foam_cell_analysis.services.models import ArtifactGroup, ExperimentDeletionInfo, PruneResult
from tests.services.test_comparison_service import FakeTraining, _dataset, _write_evaluation


@pytest.fixture
def hybrid(tmp_path):
    """学習版・検証版と、試行 1・2 を持つ実験 exp_0001 を置いた hybrid。

    TrainingService の候補スナップショットだけを FakeTraining に差し替える。
    """
    _dataset(tmp_path, "train_v000", "train", ["分類A", "分類B"], "val_v000")
    _dataset(tmp_path, "val_v000", "val", ["分類A", "分類C"])
    fake = FakeTraining(tmp_path)
    fake.add_attempt("exp_0001", 1)
    fake.add_attempt("exp_0001", 2, weights=b"weights-2")
    backend = HybridBackend(tmp_path, process_alive=lambda _record: False)
    calls = []

    def snapshot(experiment_id, *, attempt=None):
        calls.append((experiment_id, attempt))
        return fake.create_candidate_snapshot(experiment_id, attempt=attempt or 2)

    backend.training.create_candidate_snapshot = snapshot
    backend.snapshot_calls = calls
    return backend


def _config(backend):
    return backend.create_inference_config(
        "mask_rcnn", backend.default_inference_params("mask_rcnn")
    )


def test_hybrid_starts_without_mock_comparison_data(tmp_path):
    backend = HybridBackend(tmp_path)

    assert backend.mock.working["all"].items == []
    assert backend.list_experiments() == []
    assert backend.list_candidates() == []
    assert backend.list_inference_configs() == []
    assert backend.list_released_models() == []
    assert backend.get_routing() == {}
    assert backend.list_dataset_versions("train") == []
    assert backend.list_validation_versions() == []
    assert backend.default_experiment_config("cellpose")["model"]["pretrained_model"] == "cpsam"


def test_comparison_methods_are_defined_on_hybrid_itself():
    missing = [name for name in COMPARISON_METHODS if name not in HybridBackend.__dict__]
    assert missing == []


def test_mock_only_comparison_api_does_not_fall_through(tmp_path):
    backend = HybridBackend(tmp_path)
    for name in ("add_candidate_from_snapshot", "_seed_validation_data", "start_evaluation"):
        assert not hasattr(backend, name)


def test_validation_versions_come_from_dataset_store(hybrid):
    assert [item.version for item in hybrid.list_validation_versions()] == ["val_v000"]
    assert [item.version for item in hybrid.list_dataset_versions("val")] == ["val_v000"]
    train = hybrid.list_dataset_versions("train")
    assert [item.version for item in train] == ["train_v000"]
    assert train[0].base_validation_version == "val_v000"
    assert {item.version for item in hybrid.list_dataset_versions()} == {
        "train_v000",
        "val_v000",
    }
    assert [item.item_id for item in hybrid.list_validation_items("val_v000")] == [
        "val_v000_0",
        "val_v000_1",
    ]
    assert [item.item_id for item in hybrid.list_validation_items("val_v000", "分類C")] == [
        "val_v000_1"
    ]
    assert [item.item_id for item in hybrid.get_dataset_version_items("val_v000")] == [
        "val_v000_0",
        "val_v000_1",
    ]


def test_add_candidate_uses_explicit_attempt(hybrid):
    config = _config(hybrid)

    candidate = hybrid.add_candidate("exp_0001", 1, config.config_id, "試行1")

    assert hybrid.snapshot_calls == [("exp_0001", 1)]
    assert candidate.source_attempt_number == 1
    assert (hybrid.workspace_root / "comparison" / "candidates" / "RC-001").is_dir()
    assert hybrid.mock.candidates == {}
    assert hybrid.default_validation_version() == "val_v000"
    assert hybrid.base_validation_version_for("RC-001") == "val_v000"


def test_deletion_protection_uses_comparison_references(hybrid, monkeypatch):
    config = _config(hybrid)
    candidate = hybrid.add_candidate("exp_0001", 1, config.config_id)
    deleted = []
    monkeypatch.setattr(
        hybrid.training,
        "experiment_deletion_info",
        lambda experiment_id, measure_size=True: ExperimentDeletionInfo(experiment_id, True),
    )
    monkeypatch.setattr(hybrid.training, "delete_experiment", deleted.append)

    info = hybrid.experiment_deletion_info("exp_0001")
    assert not info.allowed
    assert candidate.candidate_id in info.reason
    with pytest.raises(ValueError, match=candidate.candidate_id):
        hybrid.delete_experiment("exp_0001")
    assert deleted == []

    hybrid.reject_candidate(candidate.candidate_id)
    assert hybrid.experiment_deletion_info("exp_0001").allowed
    hybrid.delete_experiment("exp_0001")
    assert deleted == ["exp_0001"]


def test_cleanup_passes_comparison_protection_for_final(hybrid, monkeypatch):
    config = _config(hybrid)
    hybrid.add_candidate("exp_0001", 1, config.config_id)
    received = []

    def plan(experiment_ids, *, protected_final=None):
        received.append(protected_final)
        reason = protected_final("exp_0001", 1)
        return [ArtifactGroup("exp_0001", 1, "final", 1, 10, not reason, reason)]

    def estimate(experiment_ids, categories, *, protected_final=None):
        received.append(protected_final)
        return 0

    def prune(experiment_ids, categories, *, protected_final=None):
        received.append(protected_final)
        return PruneResult(0, 0, plan(experiment_ids, protected_final=protected_final))

    monkeypatch.setattr(hybrid.training, "artifact_cleanup_plan", plan)
    monkeypatch.setattr(hybrid.training, "estimate_freed_bytes", estimate)
    monkeypatch.setattr(hybrid.training, "prune_artifacts", prune)

    groups = hybrid.artifact_cleanup_plan(["exp_0001"])
    assert not groups[0].deletable
    assert "RC-001" in groups[0].reason
    hybrid.estimate_freed_bytes(["exp_0001"], ["final"])
    result = hybrid.prune_artifacts(["exp_0001"], ["final"])
    assert result.skipped and result.skipped[0].category == "final"
    assert all(item == hybrid.comparison.protected_final_reason for item in received)


def test_evaluation_activity_is_passed_to_comparison(hybrid):
    config = _config(hybrid)
    candidate = hybrid.add_candidate("exp_0001", 1, config.config_id)
    hybrid.set_evaluation_activity(lambda candidate_id: candidate_id == candidate.candidate_id)

    with pytest.raises(ValueError, match="評価中"):
        hybrid.reject_candidate(candidate.candidate_id)


def test_recover_runs_training_comparison_and_evaluation_recovery(hybrid, monkeypatch):
    order = []
    monkeypatch.setattr(hybrid.training, "recover", lambda: order.append("training") or [])
    monkeypatch.setattr(hybrid.comparison, "recover", lambda: order.append("comparison") or [])
    monkeypatch.setattr(
        hybrid.comparison, "recover_evaluations", lambda: order.append("evaluations") or []
    )
    hybrid.comparison.recovery_blockers = ["評価 eval_001 を終了できませんでした"]

    assert hybrid.recover() == []
    assert order == ["training", "comparison", "evaluations"]
    assert hybrid.recovery_blockers == ["評価 eval_001 を終了できませんでした"]


def test_export_masks_from_adopted_evaluation_and_old_call_forms(hybrid, tmp_path):
    config = _config(hybrid)
    candidate = hybrid.add_candidate("exp_0001", 1, config.config_id)
    _write_evaluation(hybrid.comparison, candidate.candidate_id, "eval_001")
    adopted = hybrid.get_candidate_evaluation(candidate.candidate_id, "val_v000")
    assert adopted.evaluation_id == "eval_001"
    output = tmp_path / "out"
    output.mkdir()
    progress = []

    result = hybrid.export_particle_masks(
        MaskExportParams(
            str(output),
            [(candidate.candidate_id, "eval_001")],
            {"label", "binary"},
            "png",
            ["val_v000_1"],
        ),
        lambda done, total: progress.append((done, total)),
        lambda: False,
    )

    assert result.n_images == 1
    assert result.folder is not None and (result.folder / "export_info.json").is_file()
    assert progress[-1] == (1, 1)
    labels = hybrid.get_candidate_prediction(candidate.candidate_id, "eval_001", "val_v000_0")
    assert labels.ndim == 2
    released = hybrid.release_candidate(candidate.candidate_id, "eval_001", "採用")
    assert released.validation_dataset == "val_v000"
    state = hybrid.get_routing_state()
    applied = hybrid.apply_routing({"分類A": released.model_id}, expected_revision=state.revision)
    assert applied.assignments["分類A"] == released.model_id
    with pytest.raises(ValueError, match="別の操作"):
        hybrid.apply_routing({"分類A": None}, expected_revision=state.revision)
