import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from foam_cell_analysis.evaluation.ap import METRIC
from foam_cell_analysis.services.models import (
    Experiment,
    ExperimentConfig,
    JobExit,
    RunAttempt,
)
from foam_cell_analysis.services.training_service import TrainingService
from foam_cell_analysis.training.adapters.fake import FakeAdapter
from foam_cell_analysis.training.checkpoints import create_selected, write_label
from foam_cell_analysis.training.loop import common_candidate_epochs
from foam_cell_analysis.training.preflight import estimate_required_bytes, run_preflight
from foam_cell_analysis.training.protocol import read_events, write_run_spec
from foam_cell_analysis.training.run import run_job
from foam_cell_analysis.training.seeds import derive


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace(
    root: Path, *, epochs: int = 2, save_fold_models: bool = True, n_items: int = 4
) -> tuple[Path, dict]:
    dataset = root / "datasets" / "train_v000"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    (dataset / "dataset_info.json").write_text(
        json.dumps({"purpose": "train", "status": "RELEASED", "dataset_version": "train_v000"}),
        encoding="utf-8",
    )
    metadata_rows = []
    manifest_rows = []
    item_ids = [f"item_{index}" for index in range(n_items)]
    for index, item_id in enumerate(item_ids):
        image = np.zeros((64, 64), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint16)
        top = 8 + index * 2
        image[top : top + 16, 10:26] = min(180 + index * 10, 250)
        mask[top : top + 16, 10:26] = 1
        image[36:48, 36:50] = 230
        mask[36:48, 36:50] = 2
        image_path = dataset / "images" / f"{item_id}.png"
        mask_path = dataset / "masks" / f"{item_id}.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)
        metadata_rows.append(
            {
                "item_id": item_id,
                "source_relpath": f"source_{index}/image.png",
                "channel": "A",
                "usage": "train",
                "classification": "A" if index % 2 else "B",
                "quality": "良",
                "mask_revision": "r1",
            }
        )
        manifest_rows.append(
            {
                "item_id": item_id,
                "image_path": f"images/{item_id}.png",
                "image_sha256": _digest(image_path),
                "mask_path": f"masks/{item_id}.png",
                "mask_sha256": _digest(mask_path),
                "mask_revision": "r1",
            }
        )
    for filename, rows in (("metadata.csv", metadata_rows), ("manifest.csv", manifest_rows)):
        with (dataset / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    experiment_id = "exp_0001"
    run_dir = root / "experiments" / experiment_id / "runs" / "attempt_001"
    run_dir.mkdir(parents=True)
    fold_assignments = {
        item_id: 1 if index < n_items // 2 else 2 for index, item_id in enumerate(item_ids)
    }
    config = {
        "model": {"type": "fake"},
        "data": {"dataset_version": "train_v000", "cv": {"n_folds": 2}},
        "training": {
            "epochs": epochs,
            "learning_rate": 0.02,
            "weight_decay": 0.0,
            "early_stopping": {"enabled": False, "patience": 2},
        },
        "checkpoint": {
            "validation_interval": 1,
            "save_every": 1,
            "save_fold_models": save_fold_models,
        },
        "augmentation": {"profile": "aug_v001"},
    }
    spec = {
        "schema": 1,
        "protocol": 1,
        "run_id": f"{experiment_id}/attempt_001",
        "experiment_id": experiment_id,
        "attempt": 1,
        "queue_id": None,
        "created_at": datetime.now().astimezone().isoformat(),
        "config": config,
        "dataset": {
            "version": "train_v000",
            "path": "datasets/train_v000",
            "sha256": {
                "manifest.csv": _digest(dataset / "manifest.csv"),
                "metadata.csv": _digest(dataset / "metadata.csv"),
            },
        },
        "used_item_ids": item_ids,
        "fold_assignments": fold_assignments,
        "fold_algorithm": "group_greedy_v1",
        "augmentation_profile": {"value": {"order": [], "transforms": []}},
        "preprocessing": {"low_percentile": 1.0, "high_percentile": 99.0},
        "eval_params": {},
        "metric": METRIC,
        "seeds": {
            "folds": {"1": derive(17, "fold", 1), "2": derive(17, "fold", 2)},
            "final": derive(17, "final"),
        },
    }
    write_run_spec(run_dir, spec)
    return run_dir, spec


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _launch(
    run_dir: Path, *, timeout: int = 60, process_env: dict[str, str] | None = None
) -> tuple[int, list[dict], str]:
    process = subprocess.Popen(
        [sys.executable, "-m", "foam_cell_analysis.training.run", "--run-dir", str(run_dir)],
        cwd=run_dir,
        env=process_env or _environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    hello = json.loads(process.stdout.readline())
    assert hello["type"] == "hello" and "seq" not in hello
    stdout_lines = []

    def collect_stdout():
        for line in process.stdout:
            stdout_lines.append(line)

    reader = threading.Thread(target=collect_stdout, daemon=True)
    reader.start()
    process.stdin.write("go\n")
    process.stdin.flush()
    returncode = process.wait(timeout=timeout)
    process.stdin.close()
    reader.join(timeout=5)
    stderr = process.stderr.read()
    events = [json.loads(line) for line in stdout_lines if line]
    return returncode, [hello, *events], stderr


@pytest.mark.ml
@pytest.mark.slow
def test_mask_rcnn_process_smoke_on_train_v000(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    shutil.copytree(
        repository / "workspace" / "datasets" / "train_v000",
        tmp_path / "datasets" / "train_v000",
    )
    service = TrainingService(tmp_path, process_alive=lambda _record: False)
    config = service.default_experiment_config("mask_rcnn")
    config["data"]["cv"]["n_folds"] = 2
    config["training"].update(epochs=2, batch_size=2)
    config["training"]["early_stopping"]["enabled"] = False
    config["checkpoint"].update(validation_interval=1, save_every=1)
    config["model"]["pretrained_weights"] = None
    config["model"]["input"].update(min_size=256, max_size=256)
    experiment = service.start_training(config)
    prepared = service.prepare_training_run(experiment.experiment_id)

    environment = _environment()
    environment.update(prepared.env)
    environment["TORCH_HOME"] = str(tmp_path / "pretrained" / "torch")
    returncode, output, stderr = _launch(
        Path(prepared.run_dir), timeout=600, process_env=environment
    )

    assert returncode == 0, stderr
    events = [row for row in output if row["type"] != "hello"]
    assert events[0]["type"] == "started"
    assert events[-1]["type"] == "completed"
    assert (Path(prepared.run_dir) / "checkpoints" / "final.pt").is_file()
    outcome = service.conclude_training_run(
        experiment.experiment_id,
        1,
        JobExit(returncode=returncode),
    )
    assert outcome.status == "completed"


@pytest.mark.ml
@pytest.mark.slow
def test_process_e2e_completes_and_training_service_accepts_manifest(tmp_path):
    run_dir, spec = _workspace(tmp_path)
    returncode, output, stderr = _launch(run_dir)
    assert returncode == 0, stderr
    events = [row for row in output if row["type"] != "hello"]
    assert events[0]["type"] == "started" and events[0]["seq"] == 1
    assert [row["seq"] for row in events] == list(range(1, len(events) + 1))
    assert events[-1]["type"] == "completed"
    assert read_events(run_dir / "events.jsonl") == events
    assert [row["type"] for row in events].index("oof") < [row["type"] for row in events].index(
        "selected"
    )
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["selected_epoch"] in {1, 2}
    resolved = json.loads((run_dir / "resolved_data.json").read_text(encoding="utf-8"))
    assert resolved["image_shapes"]["item_0"] == [64, 64]
    assert resolved["ground_truth_instance_counts"]["item_0"] == 2
    assert resolved["free_bytes"] > 0
    service = TrainingService(tmp_path, process_alive=lambda _record: False)
    experiment = Experiment(
        spec["experiment_id"],
        "foam_study",
        "",
        "fake",
        ExperimentConfig(spec["config"]),
        "running",
        runs=[RunAttempt(1, datetime.now().astimezone())],
    )
    service.experiments[spec["experiment_id"]] = experiment
    assert service._valid_result(run_dir)
    checkpoint_meta = json.loads(
        (run_dir / "checkpoints" / "final.pt.json").read_text(encoding="utf-8")
    )
    assert checkpoint_meta["kind"] == "final"
    assert checkpoint_meta["run_id"] == spec["run_id"]
    assert len(checkpoint_meta["sha256"]) == 64
    for checkpoint in (run_dir / "checkpoints").rglob("*.pt"):
        metadata_path = checkpoint.with_suffix(".pt.json")
        assert metadata_path.is_file()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["sha256"] == _digest(checkpoint)
    assert not any(
        (run_dir / name).exists() for name in ("status.json", "process.json", "stop_request.json")
    )
    outcome = service.conclude_training_run(spec["experiment_id"], 1)
    assert outcome.status == "completed"


@pytest.mark.ml
def test_eof_before_go_exits_without_loading_training_stack(tmp_path):
    run_dir, _ = _workspace(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "foam_cell_analysis.training.run", "--run-dir", str(run_dir)],
        cwd=run_dir,
        env=_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    hello = json.loads(process.stdout.readline())
    assert hello["type"] == "hello"
    process.stdin.close()
    assert process.wait(timeout=10) == 3
    assert not (run_dir / "events.jsonl").exists()


@pytest.mark.ml
def test_invalid_run_spec_exits_with_code_two_without_hello(tmp_path):
    run_dir, spec = _workspace(tmp_path)
    spec.pop("fold_assignments")
    write_run_spec(run_dir, spec)
    process = subprocess.run(
        [sys.executable, "-m", "foam_cell_analysis.training.run", "--run-dir", str(run_dir)],
        cwd=run_dir,
        env=_environment(),
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    assert process.returncode == 2
    assert process.stdout == ""


@pytest.mark.parametrize(
    ("path", "key"),
    [
        (("config", "training"), "epochs"),
        (("config", "training"), "learning_rate"),
        (("config", "checkpoint"), "save_every"),
        (("config", "model"), "type"),
        (("seeds",), "folds"),
        (("dataset",), "path"),
        (("dataset",), "sha256"),
        (("augmentation_profile",), "value"),
        ((), "preprocessing"),
        ((), "eval_params"),
        ((), "metric"),
        (("fold_assignments",), "item_0"),
    ],
)
def test_run_spec_requires_nested_training_fields(tmp_path, path, key):
    from foam_cell_analysis.training.run import _validate_spec

    run_dir, spec = _workspace(tmp_path)
    parent = spec
    for part in path:
        parent = parent[part]
    parent.pop(key)
    with pytest.raises(ValueError):
        _validate_spec(run_dir, spec)


@pytest.mark.ml
def test_eof_after_go_terminates_process(tmp_path):
    run_dir, _ = _workspace(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "foam_cell_analysis.training.run", "--run-dir", str(run_dir)],
        cwd=run_dir,
        env=_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    hello = json.loads(process.stdout.readline())
    assert hello["type"] == "hello"
    process.stdin.write("go\n")
    process.stdin.flush()
    process.stdin.close()
    assert process.wait(timeout=10) == 3


@pytest.mark.ml
def test_preflight_rejects_hash_mismatch_and_injected_capacity_shortage(tmp_path):
    run_dir, spec = _workspace(tmp_path)
    import torch

    class BuildObservedAdapter(FakeAdapter):
        build_called = False

        def build(self, model_config, device):
            self.build_called = True
            super().build(model_config, device)

    adapter = BuildObservedAdapter()
    emitted = []

    def emit(kind, **fields):
        emitted.append({"type": kind, **fields})
        return emitted[-1]

    broken_image = tmp_path / "datasets" / "train_v000" / "images" / "item_0.png"
    broken_image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        run_preflight(
            run_dir,
            spec,
            adapter,
            torch.device("cpu"),
            emit,
            free_bytes_fn=lambda _path: 10**12,
        )
    assert not adapter.build_called
    assert (run_dir / "environment.json").is_file()

    run_dir, spec = _workspace(tmp_path / "capacity")

    class ConfigAwareAdapter(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.weight_configurations = []

        def initial_weights_file(self, model_config):
            self.weight_configurations.append(model_config)
            return None

    adapter = ConfigAwareAdapter()
    with pytest.raises(ValueError, match="空き容量が不足"):
        run_preflight(
            run_dir,
            spec,
            adapter,
            torch.device("cpu"),
            emit,
            free_bytes_fn=lambda _path: 0,
            weight_size_fn=lambda _adapter: 100,
        )
    expected_model_config = dict(spec["config"]["model"])
    expected_model_config["eval_params"] = spec["eval_params"]
    assert adapter.weight_configurations == [expected_model_config] * 2


@pytest.mark.ml
def test_training_exception_writes_error_manifest(tmp_path):
    run_dir, _ = _workspace(tmp_path)

    class ExplodingAdapter(FakeAdapter):
        def train_one_epoch(self, samples, lr):
            raise RuntimeError("injected training failure")

    output = io.StringIO()
    code = run_job(
        run_dir,
        stdin=io.StringIO("go\n"),
        stdout=output,
        start_watchdog=False,
        adapter_factory=ExplodingAdapter,
    )
    assert code == 1
    error = json.loads((run_dir / "error.json").read_text(encoding="utf-8"))
    assert error["message"] == "injected training failure"
    assert error["exception_type"] == "RuntimeError"
    assert "error_type" not in error
    assert json.loads(output.getvalue().splitlines()[-1])["type"] == "error"


@pytest.mark.ml
def test_save_fold_models_false_writes_no_fold_checkpoint(tmp_path):
    run_dir, _ = _workspace(tmp_path, epochs=1, save_fold_models=False)
    code = run_job(
        run_dir,
        stdin=io.StringIO("go\n"),
        stdout=io.StringIO(),
        start_watchdog=False,
    )
    assert code == 0
    checkpoints = list((run_dir / "checkpoints").rglob("*.pt"))
    assert [path.name for path in checkpoints] == ["final.pt"]


@pytest.mark.ml
@pytest.mark.slow
def test_early_stopping_reduces_validated_candidate_epochs(tmp_path):
    run_dir, spec = _workspace(tmp_path, epochs=5)
    spec["config"]["training"]["early_stopping"] = {"enabled": True, "patience": 1}
    write_run_spec(run_dir, spec)

    class PlateauAdapter(FakeAdapter):
        def train_one_epoch(self, samples, lr):
            del samples, lr
            return 0.0

    code = run_job(
        run_dir,
        stdin=io.StringIO("go\n"),
        stdout=io.StringIO(),
        start_watchdog=False,
        adapter_factory=PlateauAdapter,
    )
    assert code == 0
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert all(row["last_epoch"] < 5 for row in result["folds"].values())
    assert result["selected_epoch"] < 5


def test_common_candidate_epochs_rejects_empty_intersection():
    with pytest.raises(ValueError, match="共通部分が空"):
        common_candidate_epochs([{1, 2}, {3, 4}])


@pytest.mark.ml
@pytest.mark.ml
def test_training_samples_are_generated_lazily(tmp_path):
    run_dir, _ = _workspace(tmp_path, epochs=1)

    class StreamingAdapter(FakeAdapter):
        observed = False

        def train_one_epoch(self, samples, lr):
            from itertools import chain

            assert not isinstance(samples, list)
            iterator = iter(samples)
            first = next(iterator)
            self.observed = bool(first.item_id)
            return super().train_one_epoch(chain([first], iterator), lr)

    code = run_job(
        run_dir,
        stdin=io.StringIO("go\n"),
        stdout=io.StringIO(),
        start_watchdog=False,
        adapter_factory=StreamingAdapter,
    )
    assert code == 0


@pytest.mark.ml
def test_validation_predictions_are_processed_in_batches_of_four(tmp_path):
    run_dir, _ = _workspace(tmp_path, epochs=1, n_items=10)

    class BatchAdapter(FakeAdapter):
        def predict(self, images):
            assert len(images) <= 4
            return super().predict(images)

    code = run_job(
        run_dir,
        stdin=io.StringIO("go\n"),
        stdout=io.StringIO(),
        start_watchdog=False,
        adapter_factory=BatchAdapter,
    )
    assert code == 0


def test_checkpoint_selected_move_hardlink_and_copy_fallback(tmp_path, monkeypatch):
    temporary = tmp_path / "tmp_ckpt" / "fold_1" / "epoch_001.pt"
    temporary.parent.mkdir(parents=True)
    temporary.write_bytes(b"state")
    moved = tmp_path / "checkpoints" / "fold_1" / "selected.pt"
    assert create_selected(temporary, moved) == "moved"
    assert moved.read_bytes() == b"state" and not temporary.exists()

    periodic = tmp_path / "checkpoints" / "fold_2" / "epoch_002.pt"
    periodic.parent.mkdir(parents=True)
    periodic.write_bytes(b"periodic")
    linked = tmp_path / "checkpoints" / "fold_2" / "selected.pt"
    assert create_selected(periodic, linked) == "hardlink"
    assert os.stat(periodic).st_ino == os.stat(linked).st_ino

    copied = tmp_path / "checkpoints" / "fold_3" / "epoch_003.pt"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(b"copy")
    destination = tmp_path / "checkpoints" / "fold_3" / "selected.pt"
    monkeypatch.setattr(
        "foam_cell_analysis.training.checkpoints.os.link",
        lambda *_: (_ for _ in ()).throw(OSError()),
    )
    assert create_selected(copied, destination) == "copied"
    assert destination.read_bytes() == copied.read_bytes()


def test_label_images_switch_to_uint32_tiff_above_uint16(tmp_path):
    small = write_label(tmp_path / "small", np.array([[0, 65535]], dtype=np.uint32))
    assert small.suffix == ".png"
    large_values = np.arange(65792, dtype=np.uint32).reshape(257, 256)
    large = write_label(tmp_path / "large", large_values)
    assert large.suffix == ".tif"
    import tifffile

    assert tifffile.imread(large).dtype == np.uint32


def test_capacity_estimate_matches_design_formula():
    value = estimate_required_bytes(
        weight_bytes=100,
        n_folds=2,
        n_epochs=5,
        validation_interval=2,
        save_every=3,
        save_fold_models=True,
        image_shapes=[(10, 20), (20, 20)],
    )
    validation = {2, 4, 5}
    saves = {3}
    expected = (
        2 * len(validation | saves) * 100
        + 2 * 100
        + len(validation) * (10 * 20 + 20 * 20) * 4
        + 100
        + 100
        + 1024**3
    )
    assert value["required"] == expected
