import csv
import hashlib
import json
from pathlib import Path

import pytest

from foam_cell_analysis.services.mock.backend import MockBackend
from foam_cell_analysis.services.models import JobExit
from foam_cell_analysis.services.training_service import TrainingService
from foam_cell_analysis.training.protocol import append_event


def _workspace(root):
    folder = root / "datasets" / "train_v000"
    folder.mkdir(parents=True)
    (folder / "dataset_info.json").write_text(
        json.dumps({"purpose": "train", "status": "RELEASED", "dataset_version": "train_v000"}),
        encoding="utf-8",
    )
    metadata_fields = [
        "item_id",
        "source_relpath",
        "channel",
        "usage",
        "classification",
        "quality",
        "mask_revision",
    ]
    with (folder / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata_fields)
        writer.writeheader()
        for group in range(4):
            writer.writerow(
                {
                    "item_id": f"i{group}",
                    "source_relpath": f"g{group}/image.png",
                    "channel": "A",
                    "usage": "train",
                    "classification": "A" if group % 2 else "B",
                    "quality": "良",
                    "mask_revision": "r1",
                }
            )
    with (folder / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "item_id",
                "image_path",
                "image_sha256",
                "mask_path",
                "mask_sha256",
                "mask_revision",
            ],
        )
        writer.writeheader()
        for group in range(4):
            writer.writerow(
                {
                    "item_id": f"i{group}",
                    "image_path": "images/x.png",
                    "image_sha256": "h",
                    "mask_path": "masks/x.png",
                    "mask_sha256": "m",
                    "mask_revision": "r1",
                }
            )


def _queued_service(tmp_path):
    _workspace(tmp_path)
    service = TrainingService(tmp_path, process_alive=lambda _record: False)
    config = service.default_experiment_config("mask_rcnn")
    config["data"]["cv"]["n_folds"] = 2
    experiment = service.add_training_queue_item(config)
    return service, experiment


def test_mask_rcnn_configuration_validation_covers_backbone_and_anchor_count(tmp_path):
    _workspace(tmp_path)
    service = TrainingService(tmp_path)
    config = service.default_experiment_config("mask_rcnn")
    config["data"]["cv"]["n_folds"] = 2

    config["model"]["backbone"] = "resnet101_fpn"
    config["model"]["pretrained_weights"] = "coco"
    issues = service.validate_experiment_config(config)
    assert any("ResNet101 と COCO" in issue["message"] for issue in issues)

    config["model"]["backbone"] = "resnet50_fpn_v2"
    config["model"]["pretrained_weights"] = "imagenet"
    config["model"]["anchors"]["sizes"] = [32, 64, 128, 256]
    issues = service.validate_experiment_config(config)
    assert any("アンカーサイズは 5 個" in issue["message"] for issue in issues)


def test_cellpose_configuration_validation_covers_batch_and_sampling(tmp_path):
    _workspace(tmp_path)
    service = TrainingService(tmp_path)
    config = service.default_experiment_config("cellpose")
    config["data"]["cv"]["n_folds"] = 2
    assert service.validate_experiment_config(config) == []

    config["model"]["nimg_per_epoch"] = 0
    config["model"]["min_train_masks"] = -1
    config["model"]["scale_range"] = float("nan")
    config["training"]["batch_size"] = 0
    issues = service.validate_experiment_config(config)
    messages = [issue["message"] for issue in issues]
    assert any("nimg_per_epoch" in message for message in messages)
    assert any("min_train_masks" in message for message in messages)
    assert any("scale_range" in message for message in messages)
    assert any("batch_size" in message for message in messages)
    config["training"]["batch_size"] = 2
    assert not any(
        "batch_size" in issue["message"] for issue in service.validate_experiment_config(config)
    )


def _write_valid_result(run_dir):
    checkpoint = run_dir / "checkpoints" / "final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"weights")
    spec = json.loads((run_dir / "run_spec.json").read_text(encoding="utf-8"))
    (checkpoint.with_suffix(".pt.json")).write_text(
        json.dumps(
            {
                "model_type": spec["config"]["model"]["type"],
                "fold": None,
                "epoch": 1,
                "kind": "final",
                "run_id": spec["run_id"],
                "sha256": hashlib.sha256(b"weights").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    result = {
        "selected_epoch": 1,
        "oof": {"ap": 0.75, "per_class": {"A": [0.75, 2]}, "n_images": 2},
        "metric": {
            "id": "cellpose_ap_iou50_95_image_mean_v1",
            "thresholds": [0.5],
            "aggregation": "image_mean",
            "empty_rule": "both_empty_is_1",
        },
        "artifacts": [
            {
                "path": "checkpoints/final.pt",
                "size": 7,
                "sha256": hashlib.sha256(b"weights").hexdigest(),
            }
        ],
    }
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")


def test_prepare_is_idempotent_after_queue_write_failure(tmp_path, monkeypatch):
    service, experiment = _queued_service(tmp_path)
    queue_id = experiment.experiment_id
    save_queue = service._save_queue

    def fail_once():
        monkeypatch.setattr(service, "_save_queue", save_queue)
        raise OSError("injected queue failure")

    monkeypatch.setattr(service, "_save_queue", fail_once)
    with pytest.raises(OSError, match="injected"):
        service.prepare_training_run(experiment.experiment_id, queue_id)
    prepared = service.prepare_training_run(experiment.experiment_id, queue_id)
    assert prepared.run_id == f"{experiment.experiment_id}/attempt_001"
    assert len(experiment.runs) == 1
    assert prepared.env["TORCH_HOME"].endswith("pretrained\\torch")
    assert prepared.env["CELLPOSE_LOCAL_MODELS_PATH"].endswith("pretrained\\cellpose")
    assert "C:\\" not in json.dumps(
        json.loads(
            (
                tmp_path
                / "experiments"
                / experiment.experiment_id
                / "runs"
                / "attempt_001"
                / "run_spec.json"
            ).read_text(encoding="utf-8")
        )
    )


def test_result_requires_checkpoint_sidecar_and_matching_metadata(tmp_path):
    service, experiment = _queued_service(tmp_path)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    run_dir = Path(prepared.run_dir)
    _write_valid_result(run_dir)
    sidecar = run_dir / "checkpoints" / "final.pt.json"
    assert service._valid_result(run_dir)
    sidecar.unlink()
    assert not service._valid_result(run_dir)
    _write_valid_result(run_dir)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["kind"] = "selected"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    assert not service._valid_result(run_dir)

    _write_valid_result(run_dir)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["sha256"] = "z" * 64
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    assert not service._valid_result(run_dir)


def test_recovery_restores_result_per_class_object_shape(tmp_path):
    service, experiment = _queued_service(tmp_path)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    run_dir = Path(prepared.run_dir)
    _write_valid_result(run_dir)
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["oof"]["per_class"] = {"A": {"ap": 0.75, "n_images": 2}}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    completed = service.conclude_training_run(experiment.experiment_id, 1, JobExit(returncode=0))
    assert completed.status == "completed"

    restored = TrainingService(tmp_path, process_alive=lambda _record: False)
    restored.recover()
    evaluation = restored.get_experiment(experiment.experiment_id).oof_evaluation
    assert evaluation is not None
    assert evaluation.overall_map == 0.75
    assert evaluation.per_class == {"A": (0.75, 2)}


@pytest.mark.parametrize(
    ("artifacts", "error", "stop", "prior", "alive", "start_failed", "expected", "reason"),
    [
        (
            True,
            True,
            True,
            {"status": "completed", "reason": None},
            False,
            False,
            "completed",
            None,
        ),
        (True, True, True, None, False, False, "stopped", "user_stop"),
        (True, True, False, None, False, False, "completed", None),
        (False, True, False, None, False, False, "failed", "error"),
        (False, False, False, None, False, True, "failed", "start_failed"),
        (False, False, False, None, True, False, "running", None),
        (False, False, False, None, False, False, "stopped", "interrupted"),
    ],
)
def test_conclusion_priority(
    tmp_path, artifacts, error, stop, prior, alive, start_failed, expected, reason
):
    service, experiment = _queued_service(tmp_path)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    run_dir = Path(prepared.run_dir)
    if artifacts:
        _write_valid_result(run_dir)
    if error:
        (run_dir / "error.json").write_text(json.dumps({"message": "broken"}), encoding="utf-8")
    if stop:
        service.request_training_stop(experiment.experiment_id, 1, "user_stop")
    if prior:
        (run_dir / "status.json").write_text(json.dumps(prior), encoding="utf-8")
    service.process_alive = lambda _record: alive
    result = service.conclude_training_run(
        experiment.experiment_id, 1, JobExit(start_failed=start_failed)
    )
    assert (result.status, result.reason) == (expected, reason)


def test_recovery_removes_preparing_and_requeues_or_links_rows(tmp_path):
    service, experiment = _queued_service(tmp_path)
    runs = tmp_path / "experiments" / experiment.experiment_id / "runs"
    preparing = runs / ".preparing_abcd"
    preparing.mkdir(parents=True)
    (preparing / "junk.tmp").write_text("x", encoding="utf-8")
    augmentation = tmp_path / "augmentation" / "profiles"
    augmentation.mkdir(parents=True)
    temporary = augmentation / "profile.json.tmp"
    temporary.write_text("x", encoding="utf-8")
    service.queue["rows"].append(
        {
            "queue_id": "orphan",
            "experiment_id": experiment.experiment_id,
            "kind": "retry",
            "state": "running",
            "attempt": None,
        }
    )
    service._save_queue()
    service.recover()
    assert not preparing.exists()
    assert not temporary.exists()
    row = next(item for item in service.queue["rows"] if item["queue_id"] == "orphan")
    assert row["state"] == "queued" and row["attempt"] is None


def test_recovery_links_run_spec_created_before_queue_update(tmp_path):
    service, experiment = _queued_service(tmp_path)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    row = service.queue["rows"][0]
    row.update(state="queued", attempt=None)
    service._save_queue()
    recovered = TrainingService(tmp_path, process_alive=lambda _record: False)
    outcomes = recovered.recover()
    linked = recovered.queue["rows"][0]
    assert linked["attempt"] == 1
    assert linked["state"] == "stopped"
    assert outcomes[0].reason == "interrupted"
    assert prepared.run_id.endswith("attempt_001")


def test_defaults_and_profile_units(tmp_path):
    _workspace(tmp_path)
    service = TrainingService(tmp_path)
    config = service.default_experiment_config("cellpose")
    assert config["training"]["epochs"] == 40
    assert config["model"]["pretrained_model"] == "cpsam"
    assert config["model"]["nimg_per_epoch"] is None
    profile = service.profile_store.get("aug_v001")
    settings = {item.key: item for item in profile.transforms}
    assert settings["translation"].range_min == -0.1
    assert settings["brightness"].range_min == -10.0
    assert settings["noise"].range_max == 12.0


def test_experiments_queue_and_profiles_restore_and_queue_operations(tmp_path):
    service, first = _queued_service(tmp_path)
    second_config = service.default_experiment_config("mask_rcnn")
    second_config["data"]["cv"]["n_folds"] = 2
    second = service.add_training_queue_item(second_config)
    service.reorder_training_queue([second.experiment_id, first.experiment_id])
    assert [item.experiment_id for item in service.list_training_queue()] == [
        second.experiment_id,
        first.experiment_id,
    ]
    duplicated = service.duplicate_training_queue_items([second.experiment_id])
    assert len(duplicated) == 1
    assert duplicated[0].experiment_id != second.experiment_id
    restored = TrainingService(tmp_path)
    assert len(restored.list_experiments()) == 3
    assert restored.get_augmentation_profile("aug_v001").profile_id == "aug_v001"
    started = restored.take_next_training_queue_item()
    assert started is not None and started.experiment_id == second.experiment_id
    restored.finish_training_queue_item(second.experiment_id, "completed")
    restored.clear_finished_training_queue_items()
    assert all(
        item.experiment_id != second.experiment_id for item in restored.list_training_queue()
    )
    restored.delete_training_queue_items([first.experiment_id])
    assert all(item.experiment_id != first.experiment_id for item in restored.list_training_queue())


def test_retry_reuses_fold_assignment_and_rejects_changed_dataset(tmp_path):
    service, experiment = _queued_service(tmp_path)
    first = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    first_spec_path = Path(first.run_dir) / "run_spec.json"
    first_spec = json.loads(first_spec_path.read_text(encoding="utf-8"))
    reservation = service.add_training_retry_reservation(experiment.experiment_id)
    retry_row = service.queue["rows"][-1]
    second = service.prepare_training_run(
        experiment.experiment_id, retry_row["queue_id"], retry=True
    )
    second_spec = json.loads((Path(second.run_dir) / "run_spec.json").read_text(encoding="utf-8"))
    assert second_spec["fold_assignments"] == first_spec["fold_assignments"]
    assert second_spec["attempt"] == 2
    assert reservation.experiment_id == experiment.experiment_id

    service.add_training_retry_reservation(experiment.experiment_id)
    manifest = tmp_path / "datasets" / "train_v000" / "manifest.csv"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    latest_row = service.queue["rows"][-1]
    with pytest.raises(ValueError, match="データセットが初回試行と異なります"):
        service.prepare_training_run(experiment.experiment_id, latest_row["queue_id"], retry=True)


def test_draft_and_queue_settings_cannot_overwrite_started_experiment(tmp_path):
    service, experiment = _queued_service(tmp_path)
    service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    with pytest.raises(ValueError, match="試行がある実験"):
        service.save_experiment_draft(experiment.config.values, experiment.experiment_id)
    service.queue["rows"][0].update(state="queued", attempt=None)
    with pytest.raises(ValueError, match="試行がある実験"):
        service.update_training_queue_item(experiment.experiment_id, experiment.config.values)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.pop("selected_epoch"),
        lambda result: result.pop("oof"),
        lambda result: result["oof"].pop("per_class"),
        lambda result: result.pop("metric"),
        lambda result: result.pop("artifacts"),
        lambda result: result["artifacts"].__setitem__(0, {"path": "../outside", "size": 7}),
        lambda result: result["artifacts"].__setitem__(
            0, {"path": "C:/outside/final.pt", "size": 7}
        ),
        lambda result: result["artifacts"].__setitem__(
            0, {"path": "checkpoints/final.pt", "size": 6}
        ),
        lambda result: result["artifacts"][0].pop("sha256"),
        lambda result: result["artifacts"][0].update(sha256="not-a-hash"),
        lambda result: result["artifacts"].__setitem__(0, {"path": "missing.bin", "size": 0}),
    ],
)
def test_invalid_result_manifest_is_not_complete(tmp_path, mutation):
    service, experiment = _queued_service(tmp_path)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    run_dir = Path(prepared.run_dir)
    _write_valid_result(run_dir)
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mutation(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert not service._valid_result(run_dir)


def test_old_attempt_conclusion_updates_its_own_run_record(tmp_path):
    service, experiment = _queued_service(tmp_path)
    service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    experiment = service.get_experiment(experiment.experiment_id)
    retry = service.add_training_retry_reservation(experiment.experiment_id)
    service.prepare_training_run(experiment.experiment_id, retry.queue_id, retry=True)
    service.request_training_stop(experiment.experiment_id, 1, "user_stop")
    outcome = service.conclude_training_run(experiment.experiment_id, 1)
    assert outcome.status == "stopped"
    assert experiment.runs[0].result == "stopped"
    assert experiment.runs[1].result == "実行中"
    assert experiment.runs[1].finished_at is None
    assert experiment.status == "running"


def test_retry_preparation_state_matches_restored_state(tmp_path):
    service, experiment = _queued_service(tmp_path)
    service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    retry = service.add_training_retry_reservation(experiment.experiment_id)
    service.prepare_training_run(experiment.experiment_id, retry.queue_id, retry=True)
    experiment = service.get_experiment(experiment.experiment_id)
    prepared_state = (experiment.used_item_ids.copy(), experiment.fold_assignments.copy())

    restored = TrainingService(tmp_path, process_alive=lambda _record: False)
    restored.recover()
    restored_experiment = restored.get_experiment(experiment.experiment_id)
    restored_state = (
        restored_experiment.used_item_ids,
        restored_experiment.fold_assignments,
    )
    assert restored_state == prepared_state


def test_non_latest_attempt_events_are_ignored(caplog, tmp_path):
    service, experiment = _queued_service(tmp_path)
    first = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    experiment = service.get_experiment(experiment.experiment_id)
    first_events = [
        _event(first.run_id, 1, "phase", phase="cv", fold=1, total_epochs=3),
        _event(first.run_id, 2, "epoch", phase="cv", fold=1, epoch=1, loss=0.5, lr=0.01),
    ]
    for event in first_events:
        service.apply_training_event(experiment.experiment_id, event)
    retry = service.add_training_retry_reservation(experiment.experiment_id)
    second = service.prepare_training_run(experiment.experiment_id, retry.queue_id, retry=True)
    latest_event = _event(
        second.run_id, 1, "epoch", phase="final", fold=None, epoch=2, loss=0.2, lr=0.01
    )
    service.apply_training_event(experiment.experiment_id, latest_event)
    latest_epoch = experiment.current_epoch
    latest_history = [item.epoch for item in experiment.final_history]
    first_history = [item.epoch for item in experiment.runs[0].fold_histories[1]]

    delayed_event = _event(first.run_id, 3, "epoch", phase="cv", fold=1, epoch=3, loss=0.1, lr=0.01)
    with caplog.at_level("WARNING"):
        service.apply_training_event(experiment.experiment_id, delayed_event)

    assert experiment.current_epoch == latest_epoch
    assert [item.epoch for item in experiment.final_history] == latest_history
    assert [item.epoch for item in experiment.runs[0].fold_histories[1]] == first_history
    assert "最新でない試行のイベントを無視" in caplog.text


def test_apply_training_events_updates_display_model(tmp_path):
    service, experiment = _queued_service(tmp_path)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    experiment = service.get_experiment(experiment.experiment_id)
    run_id = prepared.run_id
    common = {"v": 1, "run_id": run_id, "time": "2026-09-28T00:00:00Z"}
    service.apply_training_event(
        experiment.experiment_id,
        {
            **common,
            "seq": 1,
            "type": "preflight",
            "n_used": 4,
            "folds": {},
            "excluded": [],
            "required_bytes": 0,
            "free_bytes": 1,
        },
    )
    service.apply_training_event(
        experiment.experiment_id,
        {
            **common,
            "seq": 2,
            "type": "phase",
            "phase": "cv",
            "fold": 1,
            "total_epochs": 3,
        },
    )
    service.apply_training_event(
        experiment.experiment_id,
        {
            **common,
            "seq": 3,
            "type": "epoch",
            "phase": "cv",
            "fold": 1,
            "epoch": 1,
            "loss": 0.4,
            "lr": 0.01,
        },
    )
    service.apply_training_event(
        experiment.experiment_id,
        {
            **common,
            "seq": 4,
            "type": "val",
            "fold": 1,
            "epoch": 1,
            "ap": 0.8,
            "n_images": 2,
        },
    )
    service.apply_training_event(
        experiment.experiment_id,
        {
            **common,
            "seq": 5,
            "type": "selected",
            "epoch": 1,
            "ap": 0.8,
            "per_class": {"A": [0.8, 2]},
        },
    )
    assert experiment.used_item_ids == [f"i{index}" for index in range(4)]
    assert experiment.fold_histories[1][0].map == 0.8
    assert experiment.selected_epoch == 1
    assert experiment.oof_evaluation.per_class["A"] == (0.8, 2)


def _event(run_id, seq, event_type, **values):
    return {
        "v": 1,
        "run_id": run_id,
        "seq": seq,
        "time": "2026-09-28T00:00:00Z",
        "type": event_type,
        **values,
    }


def test_retry_clears_experiment_display_but_keeps_each_attempt_history(tmp_path):
    service, experiment = _queued_service(tmp_path)
    first = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    experiment = service.get_experiment(experiment.experiment_id)
    events = [
        _event(first.run_id, 1, "phase", phase="cv", fold=1, total_epochs=3),
        _event(first.run_id, 2, "epoch", phase="cv", fold=1, epoch=2, loss=0.2, lr=0.01),
        _event(first.run_id, 3, "val", fold=1, epoch=2, ap=0.7, n_images=2),
        _event(
            first.run_id,
            4,
            "checkpoint",
            fold=1,
            epoch=2,
            kind="periodic",
            path="fold_1.pt",
        ),
    ]
    for event in events:
        append_event(Path(first.run_dir) / "events.jsonl", event)
        service.apply_training_event(experiment.experiment_id, event)
    assert experiment.current_epoch == 2
    old_history = experiment.runs[0].fold_histories[1]
    assert len(old_history) == 1

    retry = service.add_training_retry_reservation(experiment.experiment_id)
    second = service.prepare_training_run(experiment.experiment_id, retry.queue_id, retry=True)
    assert experiment.current_epoch == 0
    assert experiment.fold_histories == {}
    assert experiment.runs[0].fold_histories[1][0].epoch == 2
    second_events = [
        _event(second.run_id, 1, "phase", phase="final", fold=None, total_epochs=2),
        _event(second.run_id, 2, "epoch", phase="final", fold=None, epoch=1, loss=0.1, lr=0.01),
        _event(
            second.run_id,
            3,
            "checkpoint",
            fold=None,
            epoch=1,
            kind="final",
            path="checkpoints/final.pt",
        ),
    ]
    for event in second_events:
        append_event(Path(second.run_dir) / "events.jsonl", event)
        service.apply_training_event(experiment.experiment_id, event)
    assert experiment.runs[0].fold_histories[1][0].epoch == 2
    assert experiment.runs[1].fold_histories == {}
    assert experiment.runs[1].phase == "final_training"
    assert [item.epoch for item in experiment.runs[1].final_history] == [1]
    assert [item.name for item in experiment.checkpoints] == ["final.pt"]
    assert [item.name for item in experiment.runs[0].checkpoints] == ["fold_1.pt"]
    assert old_history[0].epoch == 2


def test_event_seq_duplicate_is_ignored_and_gap_replays_events_jsonl(tmp_path):
    service, experiment = _queued_service(tmp_path)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    experiment = service.get_experiment(experiment.experiment_id)
    events_path = Path(prepared.run_dir) / "events.jsonl"
    warning = _event(prepared.run_id, 1, "warning", message="once")
    phase = _event(prepared.run_id, 2, "phase", phase="cv", fold=1, total_epochs=3)
    epoch = _event(prepared.run_id, 3, "epoch", phase="cv", fold=1, epoch=2, loss=0.2, lr=0.01)
    for event in (warning, phase, epoch):
        append_event(events_path, event)
    service.apply_training_event(experiment.experiment_id, warning)
    service.apply_training_event(experiment.experiment_id, warning)
    service.apply_training_event(experiment.experiment_id, epoch)
    assert experiment.current_epoch == 2
    assert len(experiment.fold_histories[1]) == 1
    assert service._last_event_seq[(experiment.experiment_id, 1)] == 3


def test_checkpoint_ap_and_final_epoch_are_reflected(tmp_path):
    service, experiment = _queued_service(tmp_path)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    experiment = service.get_experiment(experiment.experiment_id)
    event_list = [
        _event(prepared.run_id, 1, "phase", phase="cv", fold=1, total_epochs=3),
        _event(prepared.run_id, 2, "epoch", phase="cv", fold=1, epoch=1, loss=0.3, lr=0.01),
        _event(prepared.run_id, 3, "checkpoint", fold=1, epoch=1, kind="periodic", path="fold.pt"),
        _event(prepared.run_id, 4, "val", fold=1, epoch=1, ap=0.8, n_images=2),
        _event(prepared.run_id, 5, "epoch", phase="cv", fold=1, epoch=2, loss=0.2, lr=0.01),
        _event(prepared.run_id, 6, "checkpoint", fold=1, epoch=2, kind="periodic", path="fold2.pt"),
        _event(prepared.run_id, 7, "phase", phase="final", fold=None, total_epochs=3),
        _event(prepared.run_id, 8, "epoch", phase="final", fold=None, epoch=3, loss=0.1, lr=0.01),
        _event(
            prepared.run_id,
            9,
            "checkpoint",
            fold=None,
            epoch=3,
            kind="final",
            path="checkpoints/final.pt",
        ),
    ]
    for event in event_list:
        append_event(Path(prepared.run_dir) / "events.jsonl", event)
        service.apply_training_event(experiment.experiment_id, event)
    assert [item.map for item in experiment.checkpoints] == [0.8, None, None]
    assert experiment.current_epoch == 3
    assert experiment.phase == "final_training"

    service._reset_experiment_history(experiment)
    service._replay_attempt(experiment.experiment_id, int(prepared.run_id[-3:]))
    assert [item.map for item in experiment.checkpoints] == [0.8, None, None]


def test_hello_validates_common_fields_without_seq():
    from foam_cell_analysis.training.protocol import validate_event

    validate_event(
        {
            "v": 1,
            "run_id": "exp_0001/attempt_001",
            "time": "2026-09-28T00:00:00Z",
            "type": "hello",
            "pid": 123,
        }
    )
    with pytest.raises(ValueError, match="共通項目"):
        validate_event({"type": "hello", "pid": 123})


def test_mock_backend_new_run_contract_is_idempotent_and_concludes_failures():
    backend = MockBackend()
    config = backend.default_experiment_config("mask_rcnn")
    experiment = backend.add_training_queue_item(config)
    prepared = backend.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    repeated = backend.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    assert prepared.fake
    assert repeated.run_id == prepared.run_id
    assert len(backend.get_experiment(experiment.experiment_id).runs) == 1
    backend.fail_training_ids.add(experiment.experiment_id)
    result = backend.conclude_training_run(experiment.experiment_id, 1, JobExit(returncode=1))
    assert result.status == "failed" and result.reason == "error"
