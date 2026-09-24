"""MockBackend の主要なユースケース。"""

from inspect import getmembers, isfunction

import numpy as np
import pytest

from foam_cell_analysis.services.backend import Backend
from foam_cell_analysis.services.mock.backend import MockBackend
from foam_cell_analysis.services.mock.synthetic import make_sample
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


def test_candidate_duplicate_and_release():
    backend = MockBackend()
    with pytest.raises(ValueError):
        backend.add_candidate("exp_0042", "best.pt", "infer_v005")
    backend.evaluate_candidate("RC-003")
    model = backend.release_candidate("RC-003")
    assert model.model_id.startswith("model_")
    assert backend.resolve_model("分類A").model_id == "model_007"


def test_routing_updates_and_history():
    backend = MockBackend()
    backend.apply_routing({"分類A": "model_012"})
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
        expected_extension = ".pt" if experiment.model_type == "mask_rcnn" else ""
        assert {f"epoch_{epoch:03d}{expected_extension}" for epoch in range(10, 101, 10)} <= {
            checkpoint.name for checkpoint in experiment.checkpoints
        }
        best = next(
            checkpoint
            for checkpoint in experiment.checkpoints
            if checkpoint.name in {"best", "best.pt"}
        )
        assert best.map == max(point.map for point in experiment.history if point.map is not None)

    checkpoints = {checkpoint.name for checkpoint in backend.get_experiment("exp_0042").checkpoints}
    assert "epoch_080.pt" in checkpoints
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


def test_image_apis_return_deterministic_revision_and_candidate_results():
    backend = MockBackend()
    item = backend.get_working_dataset("train").items[0]
    image = backend.get_item_image("train", item.item_id, "A")
    assert image.shape == (512, 512)
    mask1 = backend.get_item_mask("train", item.item_id, "rev_001")
    mask2 = backend.get_item_mask("train", item.item_id, "rev_002")
    assert mask1.dtype == np.int32
    assert not np.array_equal(mask1, mask2)
    prediction1 = backend.get_candidate_prediction("RC-001", "item_001001")
    prediction2 = backend.get_candidate_prediction("RC-002", "item_001001")
    assert not np.array_equal(prediction1, prediction2)
    first = backend.get_inference_result("sample.tif", "model_007")
    second = backend.get_inference_result("sample.tif", "model_007")
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_synthetic_mask_contains_contact_and_summary_reports_next_version():
    backend = MockBackend()
    _, labels = make_sample(7, size=128)
    vertical_contacts = (
        (labels[1:, :] != labels[:-1, :]) & (labels[1:, :] > 0) & (labels[:-1, :] > 0)
    )
    horizontal_contacts = (
        (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] > 0) & (labels[:, :-1] > 0)
    )
    assert vertical_contacts.any() or horizontal_contacts.any()
    summary = backend.summarize_working_changes("train")
    assert summary["next_version"] == "train_v004"
    assert summary["n_images"] == 59
    assert summary["missing_metadata"] == 2


def test_record_epoch_obeys_checkpoint_interval_and_best_mode():
    backend = MockBackend()
    config = backend.default_experiment_config("mask_rcnn")
    config["checkpoint"]["save_every"] = 3
    config["checkpoint"]["best_mode"] = "min"
    experiment = backend.start_training(config)
    backend.record_epoch(experiment.experiment_id, 1, 0.8, 0.8)
    backend.record_epoch(experiment.experiment_id, 2, 0.7, 0.9)
    best_name = "best.pt"
    assert next(item for item in experiment.checkpoints if item.name == best_name).epoch == 1
    backend.record_epoch(experiment.experiment_id, 3, 0.6, 0.7)
    assert any(item.name == "epoch_003.pt" for item in experiment.checkpoints)
    assert next(item for item in experiment.checkpoints if item.name == best_name).epoch == 3


def test_estimate_training_items_applies_real_filters_and_duplicate_warning():
    backend = MockBackend()
    train, validation = backend.estimate_training_items(classification="分類A")
    assert (train, validation) == (15, 4)
    good_train, good_validation = backend.estimate_training_items(quality_filter="good_only")
    assert good_train + good_validation == 20
    config = backend.get_experiment("exp_0042").config.values
    config["experiment"]["id"] = "exp_0099"
    config["experiment"]["description"] = "別の説明"
    warnings = backend.validate_experiment_config(config)
    assert any("同一設定の実験 exp_0042" in result["message"] for result in warnings)


def test_evaluation_can_be_started_and_completed():
    backend = MockBackend()
    backend.start_evaluation(["RC-003"], "val_v002")
    assert backend.get_candidate("RC-003").status == "evaluating"
    evaluation = backend.evaluate_candidate("RC-003", "val_v002")
    assert backend.get_candidate("RC-003").status == "candidate"
    assert evaluation.overall_map > 0
    assert len(backend.list_validation_items("val_v002", "分類A")) == 10
