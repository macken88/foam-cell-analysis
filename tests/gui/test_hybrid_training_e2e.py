import os
import shutil
from pathlib import Path

import pytest

from foam_cell_analysis.gui.training_runner import TrainingRunner
from foam_cell_analysis.services.hybrid_backend import HybridBackend
from foam_cell_analysis.training.protocol import read_run_spec, write_run_spec


@pytest.mark.slow
def test_hybrid_runner_completes_real_fake_model_process(qtbot, tmp_path):
    from tests.training.test_training_process import _workspace

    _workspace(tmp_path, epochs=1, n_items=4)
    shutil.rmtree(tmp_path / "experiments")
    backend = HybridBackend(tmp_path)
    config = backend.default_experiment_config("cellpose")
    config["data"]["dataset_version"] = "train_v000"
    config["data"]["cv"]["n_folds"] = 2
    config["data"]["input_channels"] = ["A"]
    config["training"]["epochs"] = 1
    config["model"]["min_train_masks"] = 0
    config["checkpoint"]["validation_interval"] = 1
    config["checkpoint"]["save_every"] = 1
    experiment = backend.start_training(config)
    prepare = backend.prepare_training_run

    def prepare_fake(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        spec_path = Path(prepared.run_dir) / "run_spec.json"
        spec = read_run_spec(Path(prepared.run_dir))
        spec["config"]["model"]["type"] = "fake"
        write_run_spec(Path(prepared.run_dir), spec)
        source = str(Path(__file__).resolve().parents[2] / "src")
        prepared.env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (source, prepared.env.get("PYTHONPATH")))
        )
        assert spec_path.is_file()
        return prepared

    backend.prepare_training_run = prepare_fake
    runner = TrainingRunner(backend)
    outcomes = []
    runner.ended.connect(outcomes.append)
    runner.start(experiment.experiment_id)
    qtbot.waitUntil(lambda: not runner.is_busy, timeout=120_000)

    assert outcomes[0].status == "completed", outcomes[0]
    assert backend.get_experiment(experiment.experiment_id).status == "completed"
    assert (
        Path(backend.training.root)
        / experiment.experiment_id
        / "runs"
        / "attempt_001"
        / "result.json"
    ).is_file()
