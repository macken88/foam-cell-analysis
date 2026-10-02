"""MockBackend の主要なユースケース。"""

from inspect import getmembers, isfunction

import pytest

from foam_cell_analysis.services.backend import Backend
from foam_cell_analysis.services.mock.backend import MockBackend
from foam_cell_analysis.services.models import AugmentationProfile


def test_seed_data_covers_all_modes():
    backend = MockBackend()
    assert len(backend.get_working_dataset("train").items) >= 50
    assert len(backend.get_working_dataset("val").items) >= 25
    assert {e.experiment_id for e in backend.list_experiments()} >= {
        "exp_0042",
        "exp_0043",
        "exp_0044",
        "exp_0045",
    }
    assert {c.candidate_id for c in backend.list_candidates()} >= {"RC-001", "RC-002", "RC-003"}
    assert backend.resolve_model("分類A").model_id == "model_007"


def test_data_preparation_has_one_usage_based_working_dataset():
    backend = MockBackend()
    items = backend.get_working_items()
    assert len(items) == 113
    assert backend.get_working_dataset("train").base_version == "train_v003"
    assert backend.get_working_dataset("val").base_version == "val_v003"
    assert {item.usage for item in items} == {"train", "val", "excluded", "unassigned"}
    selected = [items[5].item_id, items[6].item_id]
    backend.set_usage(selected, "unassigned")
    assert all(
        next(item for item in items if item.item_id == item_id).usage == "unassigned"
        for item_id in selected
    )
    assert backend.validate_items().errors


def test_auto_split_preview_does_not_mutate_and_finalize_summary_has_both_purposes():
    backend = MockBackend()
    candidates = backend.scan_import_folders({"A": "C:/preview"}, "C:/masks")[:4]
    imported = backend.import_folders(
        candidates,
        {
            candidates[0].source_relpath.rsplit("/", 1)[0]: {
                "classification": "分類A",
                "quality": "良",
            }
        },
    )
    before = [item.usage for item in imported]
    item_ids = [item.item_id for item in imported]
    preview = backend.preview_auto_split(
        {"ratio": 50, "unit": "item", "stratify": True, "seed": 7}, item_ids
    )
    assert preview["train"] + preview["val"] == len(imported)
    assert [item.usage for item in imported] == before
    assignments = backend.preview_auto_split_assignments(
        {"ratio": 50, "unit": "source_folder", "stratify": True, "seed": 7}, item_ids
    )
    assert set(assignments) == set(item_ids)
    assert set(assignments.values()) <= {"train", "val", "excluded"}
    assert [item.usage for item in imported] == before
    backend.apply_auto_split(
        {"ratio": 50, "unit": "source_folder", "stratify": True, "seed": 7}, item_ids
    )
    assert all(item.usage in {"train", "val"} for item in imported)
    assert set(backend.summarize_finalize()) == {"train", "val"}


def test_validation_then_finalize_creates_version():
    backend = MockBackend()
    ds = backend.get_working_dataset("train")
    for item in ds.items:
        if item.included:
            item.classification = item.classification or "分類A"
            item.quality = item.quality or "良"
    report = backend.validate_working_dataset("train")
    assert not report.errors
    version = backend.finalize_dataset("train", "テスト")
    assert version.version == "train_v004"
    assert ds.state == "WORKING"


def test_finalize_working_rejects_duplicate_train_validation_images_before_versioning():
    backend = MockBackend()
    before = [item.version for item in backend.list_dataset_versions()]
    working = backend.get_working_items()
    for item in working:
        if item.usage in {"train", "val"}:
            item.classification = item.classification or "分類A"
            item.quality = item.quality or "良"
            item.mask_revisions = item.mask_revisions or ["rev_001"]
    train = next(item for item in working if item.usage == "train")
    validation = next(item for item in working if item.usage == "val")
    train.sha256 = validation.sha256
    train.change = "changed"
    with pytest.raises(ValueError, match="識別子または画像の重複"):
        backend.finalize_working_dataset()
    assert [item.version for item in backend.list_dataset_versions()] == before


def test_validation_only_finalize_keeps_existing_training_pair():
    backend = MockBackend()
    training = next(
        item for item in backend.list_dataset_versions("train") if item.version == "train_v003"
    )
    original_pair = training.base_validation_version
    working = backend.get_working_items()
    baseline_train = backend._items_for_version("train_v003")
    for item in baseline_train:
        item.usage = "train"
    working[:] = [item for item in working if item.usage != "train"] + baseline_train
    for item in baseline_train:
        item.change = None
    for item in working:
        if item.usage in {"train", "val"}:
            item.classification = item.classification or "分類A"
            item.quality = item.quality or "良"
            item.mask_revisions = item.mask_revisions or ["rev_001"]
    validation_item = next(item for item in working if item.usage == "val")
    validation_item.comment = "changed validation only"
    validation_item.change = "changed"
    created = backend.finalize_working_dataset()
    assert [item.purpose for item in created] == ["val"]
    assert training.base_validation_version == original_pair


def _evaluate(backend: MockBackend, candidate_id: str) -> str:
    """候補に固定した検証用の版での評価を、模擬の評価イベントで完了させて評価 ID を返す。"""
    prepared = backend.prepare_evaluation_run(candidate_id)
    evaluation_id = prepared.run_id.rsplit("/", 1)[1]
    event = {"v": 1, "run_id": prepared.run_id, "seq": 1, "time": 0, "type": "completed"}
    backend.apply_evaluation_event(candidate_id, evaluation_id, event)
    outcome = backend.conclude_evaluation_run(candidate_id, evaluation_id)
    assert outcome.status == "completed"
    return evaluation_id


def test_candidate_duplicate_and_release():
    backend = MockBackend()
    with pytest.raises(ValueError, match="登録済み"):
        backend.add_candidate("exp_0042", 1, "infer_v005")
    evaluation_id = _evaluate(backend, "RC-003")
    model = backend.release_candidate("RC-003", evaluation_id)
    assert model.model_id.startswith("model_")
    assert (model.model_type, model.inference_config_id) == ("mask_rcnn", "infer_v007")
    assert backend.resolve_model("分類A").model_id == "model_007"


def test_seeded_candidates_match_releases_and_oof_applicability():
    backend = MockBackend()
    released = {model.candidate_id: model for model in backend.list_released_models()}
    for candidate_id, model in released.items():
        candidate = backend.get_candidate(candidate_id)
        assert candidate.status == "released"
        assert candidate.released_model_id == model.model_id
        assert model.inference_config_id == candidate.inference_config_id
    assert backend.get_candidate("RC-003").status == "candidate"
    matching = backend.get_candidate("RC-002")
    assert matching.oof_applicability == "matching"
    different = backend.get_candidate("RC-001")
    assert different.oof_applicability == "different"
    assert different.oof_reason == "推論設定が学習時と異なります"


def test_routing_updates_and_history():
    backend = MockBackend()
    backend.apply_routing({"分類A": "model_012"}, expected_revision=backend.routing_revision)
    assert backend.resolve_model("分類A").model_id == "model_012"
    assert backend.list_routing_history()[-1].before_model_id == "model_007"


def test_protocol_and_mock_backend_methods_match():
    protocol_methods = {
        name
        for name, method in Backend.__dict__.items()
        if callable(method) and not name.startswith("_")
    }
    backend_methods = {
        name for name, method in getmembers(MockBackend, isfunction) if not name.startswith("_")
    }
    assert protocol_methods == backend_methods


def test_seed_experiments_checkpoints_and_references_are_consistent():
    backend = MockBackend()
    for experiment_id in ("exp_0042", "exp_0043"):
        experiment = backend.get_experiment(experiment_id)
        assert experiment.status == "completed"
        assert experiment.current_epoch == experiment.total_epochs == 100
        assert len(experiment.history) == 100
        assert [point.epoch for point in experiment.history if point.map is not None] == list(
            range(5, 101, 5)
        )
        for fold in range(1, 6):
            assert {f"epoch_{epoch:03d}.pt" for epoch in range(10, 101, 10)} <= {
                checkpoint.name for checkpoint in experiment.checkpoints if checkpoint.fold == fold
            }
        assert any(checkpoint.name == "final.pt" for checkpoint in experiment.checkpoints)
        assert experiment.selected_epoch is not None
        assert experiment.oof_evaluation is not None

    checkpoints = {checkpoint.name for checkpoint in backend.get_experiment("exp_0042").checkpoints}
    assert "final.pt" in checkpoints
    for candidate in backend.list_candidates():
        experiment = backend.get_experiment(candidate.experiment_id)
        assert any(checkpoint.name == candidate.checkpoint for checkpoint in experiment.checkpoints)
        assert candidate.inference_config_id in backend.inference_configs
    for model in backend.list_released_models():
        assert model.candidate_id in backend.candidates
        assert model.experiment_id == backend.get_candidate(model.candidate_id).experiment_id


def test_augmentation_profiles_are_independent_and_saved_by_copy():
    backend = MockBackend()
    profile1 = backend.get_augmentation_profile("aug_v001")
    profile2 = backend.get_augmentation_profile("aug_v002")
    assert {item.key for item in profile1.transforms} == {
        "horizontal_flip",
        "vertical_flip",
        "rotation",
        "scale",
        "translate",
        "crop",
        "elastic",
        "brightness",
        "contrast",
        "gamma",
        "blur",
        "noise",
        "channel_dropout",
        "channel_intensity",
    }
    assert profile1.transforms[0] is not profile2.transforms[0]
    profile = AugmentationProfile(
        "draft", "編集版", "aug_v003", profile1.transforms, profile1.order
    )
    saved = backend.save_augmentation_profile(profile)
    assert saved.profile_id == "aug_v004"
    assert profile.profile_id == "draft"
    saved.transforms[0].enabled = False
    assert profile.transforms[0].enabled
    assert backend.get_augmentation_profile("aug_v003").transforms[0].enabled
    assert backend.get_augmentation_profile("aug_v003").used_by_experiments == ["exp_0042"]


def test_inclusion_change_state_restores_added_and_original_changes():
    backend = MockBackend()
    added = backend.get_working_dataset().items[2]
    backend.update_item("train", added.item_id, included=False)
    assert added.change == "excluded"
    backend.update_item("train", added.item_id, included=True)
    assert added.change == "added"
    changed = backend.get_working_dataset().items[3]
    backend.update_item("train", changed.item_id, included=False)
    backend.update_item("train", changed.item_id, included=True)
    assert changed.change == "changed"
    untouched = backend.get_working_dataset().items[5]
    backend.update_item("train", untouched.item_id, included=False)
    backend.update_item("train", untouched.item_id, included=True)
    assert untouched.change is None


def test_evaluation_can_be_started_and_completed():
    backend = MockBackend()
    _evaluate(backend, "RC-003")
    assert backend.get_candidate("RC-003").status == "candidate"
    assert backend.get_candidate("RC-003").validation_version == "val_v003"
    evaluation = backend.get_candidate_evaluation("RC-003").evaluation
    assert evaluation.overall_map > 0
    assert len(backend.list_validation_items("val_v003", "分類A")) == 10


# ---- 比較・評価設計 16.1 の新しい API の形（段階 D1） ----


def test_mock_evaluation_lifecycle_through_fake_evaluation_job(qtbot):
    from foam_cell_analysis.gui.evaluation_runner import EvaluationRunner

    backend = MockBackend()
    runner = EvaluationRunner(backend)
    backend.set_evaluation_activity(runner.is_evaluation_active)
    outcomes = []
    progressed = []
    runner.ended.connect(outcomes.append)
    runner.progressed.connect(progressed.append)

    runner.start(["RC-003"])
    with pytest.raises(ValueError, match="評価中"):
        backend.reject_candidate("RC-003")
    qtbot.waitUntil(lambda: not runner.is_busy, timeout=5000)

    assert [(item.status, item.evaluation_id) for item in outcomes] == [("completed", "eval_001")]
    assert progressed
    record = backend.get_candidate_evaluation("RC-003")
    assert record.evaluation_id == "eval_001"
    assert record.validation_version == "val_v003"
    assert record.evaluation.overall_map > 0
    assert record.evaluation.n_images == 30
    assert backend.get_candidate("RC-003").evaluations["val_v003"] is record.evaluation
    assert backend.get_evaluation_progress("RC-003") is None


def test_mock_evaluation_stop_and_numbering(qtbot):
    from foam_cell_analysis.gui.evaluation_runner import EvaluationRunner

    backend = MockBackend()
    runner = EvaluationRunner(backend)
    outcomes = []
    runner.ended.connect(outcomes.append)
    runner.start(["RC-003"])
    qtbot.waitUntil(lambda: runner.job is not None, timeout=5000)
    runner.request_stop()
    qtbot.waitUntil(lambda: not runner.is_busy, timeout=5000)
    assert (outcomes[0].status, outcomes[0].reason) == ("stopped", "user_stop")
    assert backend.get_candidate_evaluation("RC-003") is None

    prepared = backend.prepare_evaluation_run("RC-003")
    assert prepared.fake
    assert prepared.run_id == "RC-003/val_v003/eval_002"
    statuses = [item.status for item in backend.list_candidate_evaluations("RC-003", "val_v003")]
    assert statuses == ["stopped", "running"]


def test_mock_release_prediction_and_external_results_by_evaluation_id():
    backend = MockBackend()
    record = backend.get_candidate_evaluation("RC-001")
    assert record is not None and record.status == "completed"
    item_id = backend.list_validation_items("val_v003")[0].item_id
    labels = backend.get_candidate_prediction("RC-001", record.evaluation_id, item_id)
    assert labels.ndim == 2

    evaluation_id = _evaluate(backend, "RC-003")
    backend.save_external_analysis("RC-003", evaluation_id, {item_id: 1.5}, software="X")
    assert backend.list_external_results("RC-003")[-1]["evaluation_id"] == evaluation_id
    assert backend.get_candidate("RC-003").external_summary["mean"] == 1.5
    with pytest.raises(ValueError, match="評価がありません"):
        backend.release_candidate("RC-003", "eval_099")
    model = backend.release_candidate("RC-003", evaluation_id, "新形式")
    assert model.validation_dataset == "val_v003"
    assert model.external_summary["n_images"] == 1
    assert backend.get_candidate("RC-003").released_model_id == model.model_id


def test_mock_routing_revision_and_validation():
    backend = MockBackend()
    state = backend.get_routing_state()
    assert "分類A" in backend.list_routing_classifications()

    applied = backend.apply_routing({"分類A": "model_012"}, expected_revision=state.revision)
    assert applied.revision == state.revision + 1
    assert applied.assignments["分類A"] == "model_012"
    with pytest.raises(ValueError, match="別の操作"):
        backend.apply_routing({"分類A": None}, expected_revision=state.revision)
    with pytest.raises(ValueError, match="未登録"):
        backend.apply_routing(
            {"分類B": None, "分類C": "model_999"}, expected_revision=applied.revision
        )
    assert backend.get_routing()["分類B"] == "model_012"


def test_mock_inference_config_range_and_reuse():
    backend = MockBackend()
    with pytest.raises(ValueError, match="範囲"):
        backend.create_inference_config("mask_rcnn", {"box_score_thresh": 1.5})
    first = backend.create_inference_config("cellpose", {"flow_threshold": 0.8})
    again = backend.create_inference_config("cellpose", {"flow_threshold": 0.8})
    assert first.config_id == again.config_id


def test_mock_add_candidate_with_attempt_number():
    backend = MockBackend()
    config = backend.create_inference_config("mask_rcnn", {"box_score_thresh": 0.3})
    candidate = backend.add_candidate("exp_0042", 1, config.config_id)
    assert candidate.source_attempt_number == 1
    with pytest.raises(ValueError, match="登録済み"):
        backend.add_candidate("exp_0042", 1, config.config_id)
    with pytest.raises(ValueError, match="試行 9"):
        backend.add_candidate("exp_0042", 9, config.config_id)


def test_mock_inference_result_does_not_depend_on_releases():
    empty = MockBackend(seed_samples=False)
    seeded = MockBackend()
    image, labels = empty.get_inference_result("foam_0001.tif", "model_999")
    other_image, other_labels = seeded.get_inference_result("foam_0001.tif", "model_999")
    assert (image == other_image).all() and (labels == other_labels).all()


def test_mock_without_samples_has_no_comparison_seed():
    backend = MockBackend(seed_samples=False)
    assert backend.list_inference_configs() == []
    assert backend.list_candidates() == []
    assert backend.list_released_models() == []
    assert backend.get_routing() == {}
    assert backend.list_validation_versions() == []


def test_mock_cleanup_plan_protects_candidate_final_and_records_pruned():
    backend = MockBackend()
    plan = backend.artifact_cleanup_plan(["exp_0042"])
    final = next(group for group in plan if group.category == "final")
    assert not final.deletable and "RC-003" in final.reason
    assert all(group.category != "temporary" for group in plan)
    freed = backend.estimate_freed_bytes(["exp_0042"], ["fold_periodic", "final"])
    assert freed > 0

    result = backend.prune_artifacts(["exp_0042"], ["fold_periodic", "final"])
    assert result.freed_bytes == freed
    assert [group.category for group in result.skipped] == ["final"]
    assert "checkpoints/fold_1/epoch_010.pt" in backend.pruned_paths("exp_0042", 1)
    remaining = backend.artifact_cleanup_plan(["exp_0042"])
    assert all(group.category != "fold_periodic" for group in remaining)
    stopped = backend.artifact_cleanup_plan(["exp_0044"])
    assert any(group.category == "temporary" and group.deletable for group in stopped)


def test_mock_export_counts_images_and_can_be_cancelled():
    backend = MockBackend()
    params = {
        "output_parent": "C:/unused",
        "selections": [("RC-001", "eval_001"), ("RC-002", "eval_001")],
        "contents": ["label", "binary"],
        "file_format": "png",
    }
    progress = []
    result = backend.export_particle_masks(
        params, lambda done, total: progress.append((done, total)), lambda: False
    )
    assert result.n_images == 60 and result.folder is None and not result.cancelled
    assert progress[-1] == (60, 60)

    checks = []

    def cancel_after_two():
        checks.append(1)
        return len(checks) > 2

    cancelled = backend.export_particle_masks(params, lambda *_: None, cancel_after_two)
    assert cancelled.cancelled and cancelled.n_images == 2
    with pytest.raises(ValueError, match="評価がありません"):
        backend.export_particle_masks(
            {**params, "selections": [("RC-003", "eval_001")]}, lambda *_: None, lambda: False
        )


def test_mock_add_candidate_keeps_training_oof_and_judges_applicability():
    backend = MockBackend()
    experiment = backend.get_experiment("exp_0042")
    default = backend.create_inference_config("mask_rcnn", {})
    changed = backend.create_inference_config("mask_rcnn", {"box_score_thresh": 0.3})

    same = backend.add_candidate("exp_0042", 1, default.config_id)
    other = backend.add_candidate("exp_0042", 1, changed.config_id)

    assert same.oof_applicability == "matching"
    assert other.oof_applicability == "different"
    assert other.oof_reason
    assert same.oof_evaluation.overall_map == other.oof_evaluation.overall_map
    assert experiment.oof_evaluation is None or (
        same.oof_evaluation.overall_map == experiment.oof_evaluation.overall_map
    )
