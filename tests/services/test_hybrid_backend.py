from foam_cell_analysis.services.hybrid_backend import HybridBackend


def test_hybrid_backend_has_empty_mock_catalog_and_persistent_training_service(tmp_path):
    backend = HybridBackend(tmp_path)

    assert backend.mock.working["all"].items == []
    assert backend.list_experiments() == []
    assert backend.list_candidates() == []
    assert backend.list_released_models() == []
    assert backend.list_dataset_versions("train") == []
    assert backend.default_experiment_config("cellpose")["model"]["pretrained_model"] == "cpsam"
