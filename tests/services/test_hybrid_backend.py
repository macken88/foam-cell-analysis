from foam_cell_analysis.services.hybrid_backend import HybridBackend
from foam_cell_analysis.services.models import CandidateSnapshot


def test_hybrid_backend_has_empty_mock_catalog_and_persistent_training_service(tmp_path):
    backend = HybridBackend(tmp_path)

    assert backend.mock.working["all"].items == []
    assert backend.list_experiments() == []
    assert backend.list_candidates() == []
    assert backend.list_released_models() == []
    assert backend.list_dataset_versions("train") == []
    assert backend.default_experiment_config("cellpose")["model"]["pretrained_model"] == "cpsam"
    assert [item.version for item in backend.list_validation_versions()] == ["val_v003"]


def test_snapshot_candidate_keeps_real_oof_and_releases_without_mock_experiment(
    tmp_path, monkeypatch
):
    backend = HybridBackend(tmp_path)
    snapshot = CandidateSnapshot(
        experiment_id="exp_real_001",
        attempt=2,
        selected_epoch=7,
        run_id="exp_real_001/attempt_002",
        checkpoint_path="checkpoints/final.pt",
        oof_evaluation={"ap": 0.731, "per_class": {"A": [0.731, 12]}, "n_images": 12},
        experiment_config={"model": {"type": "cellpose", "scale_range": 0.2}},
    )
    candidate = backend.mock.add_candidate_from_snapshot(snapshot, "infer_v006", "measured")

    assert candidate.source_attempt_number == 2
    assert candidate.snapshot.checkpoint_path == "checkpoints/final.pt"
    assert candidate.oof_evaluation.overall_map == 0.731
    assert candidate.oof_evaluation.per_class == {"A": (0.731, 12)}
    assert candidate.snapshot.oof_evaluation["n_images"] == 12
    monkeypatch.setattr(
        backend.mock,
        "get_experiment",
        lambda _experiment_id: (_ for _ in ()).throw(AssertionError("mock experiment lookup")),
    )

    backend.mock.start_evaluation([candidate.candidate_id], "val_v003")
    backend.mock.evaluate_candidate(candidate.candidate_id, "val_v003")
    released = backend.mock.release_candidate(candidate.candidate_id, "ready", "val_v003")

    assert released.source_attempt_number == 2
    assert released.preprocessing_config == {"type": "cellpose", "scale_range": 0.2}
