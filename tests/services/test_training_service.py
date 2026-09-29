import csv
import hashlib
import json
import os
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


def test_candidate_snapshot_captures_completed_attempt_config_and_oof(tmp_path):
    service, experiment = _queued_service(tmp_path)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    _write_valid_result(Path(prepared.run_dir))
    outcome = service.conclude_training_run(experiment.experiment_id, 1, JobExit(returncode=0))
    assert outcome.status == "completed"

    snapshot = service.create_candidate_snapshot(experiment.experiment_id)

    assert snapshot.attempt == 1
    assert snapshot.selected_epoch == 1
    assert snapshot.run_id == prepared.run_id
    assert snapshot.checkpoint_path == "checkpoints/final.pt"
    assert snapshot.oof_evaluation["ap"] == 0.75
    assert snapshot.oof_evaluation["per_class"] == {"A": [0.75, 2]}
    assert snapshot.experiment_config["model"]["type"] == "mask_rcnn"


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


def _completed_service(tmp_path, service_factory=None):
    _workspace(tmp_path)
    service = (service_factory or TrainingService)(tmp_path, process_alive=lambda _record: False)
    config = service.default_experiment_config("mask_rcnn")
    config["data"]["cv"]["n_folds"] = 2
    experiment = service.add_training_queue_item(config)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    _write_valid_result(Path(prepared.run_dir))
    outcome = service.conclude_training_run(experiment.experiment_id, 1, JobExit(returncode=0))
    assert outcome.status == "completed"
    return service, experiment


def test_protocol_error_after_hello_concludes_as_failed_error(tmp_path):
    _workspace(tmp_path)
    service = TrainingService(tmp_path, process_alive=lambda _record: False)
    config = service.default_experiment_config("mask_rcnn")
    config["data"]["cv"]["n_folds"] = 2
    experiment = service.add_training_queue_item(config)
    prepared = service.prepare_training_run(experiment.experiment_id, experiment.experiment_id)
    message = "学習プロセスのプロトコルエラー: イベントの共通項目がありません: seq"

    outcome = service.conclude_training_run(
        experiment.experiment_id, 1, JobExit(returncode=1, message=message, protocol_error=True)
    )

    assert (outcome.status, outcome.reason, outcome.message) == ("failed", "error", message)
    status = json.loads((Path(prepared.run_dir) / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed" and status["message"] == message


def test_delete_experiment_removes_only_its_folder_and_refuses_running(tmp_path):
    service, experiment = _completed_service(tmp_path)
    expid = experiment.experiment_id
    kept = [tmp_path / "pretrained" / "keep.bin", tmp_path / "augmentation" / "keep.json"]
    for path in kept:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep")
    dataset_files = sorted((tmp_path / "datasets").rglob("*"))
    service.add_training_retry_reservation(expid)
    config = service.default_experiment_config("mask_rcnn")
    config["data"]["cv"]["n_folds"] = 2
    running = service.add_training_queue_item(config)
    service.prepare_training_run(running.experiment_id, running.experiment_id)

    info = service.experiment_deletion_info(expid)
    assert info.allowed
    assert info.attempts == 1
    assert info.size_bytes > 0
    blocked = service.experiment_deletion_info(running.experiment_id)
    assert not blocked.allowed
    assert "学習中" in blocked.reason
    with pytest.raises(ValueError, match="学習中"):
        service.delete_experiment(running.experiment_id)

    service.delete_experiment(expid)

    assert not (tmp_path / "experiments" / expid).exists()
    assert (tmp_path / "experiments" / running.experiment_id).is_dir()
    assert expid not in {item.experiment_id for item in service.list_experiments()}
    assert all(row["experiment_id"] != expid for row in service.queue["rows"])
    assert all(path.read_bytes() == b"keep" for path in kept)
    assert sorted((tmp_path / "datasets").rglob("*")) == dataset_files
    restored = TrainingService(tmp_path, process_alive=lambda _record: False)
    assert expid not in restored.experiments
    assert all(row["experiment_id"] != expid for row in restored.queue["rows"])


def test_hybrid_refuses_to_delete_experiment_referenced_by_candidate(tmp_path):
    from foam_cell_analysis.services.hybrid_backend import HybridBackend

    backend, experiment = _completed_service(tmp_path, HybridBackend)
    expid = experiment.experiment_id
    config = backend.create_inference_config(
        "mask_rcnn", backend.default_inference_params("mask_rcnn")
    )
    candidate = backend.add_candidate(expid, 1, config.config_id)

    info = backend.experiment_deletion_info(expid)
    assert not info.allowed
    assert f"比較候補 {candidate.candidate_id}" in info.reason
    with pytest.raises(ValueError, match=candidate.candidate_id):
        backend.delete_experiment(expid)
    assert (tmp_path / "experiments" / expid).is_dir()

    backend.reject_candidate(candidate.candidate_id)
    backend.delete_experiment(expid)
    assert not (tmp_path / "experiments" / expid).exists()


def test_deleted_experiment_number_is_not_reused(tmp_path):
    service, experiment = _queued_service(tmp_path)
    service.delete_training_queue_items([experiment.experiment_id])
    newest = service.save_experiment_draft(service.default_experiment_config("mask_rcnn"))
    deleted_number = int(newest.experiment_id[-4:])
    service.delete_experiment(newest.experiment_id)

    reloaded = TrainingService(tmp_path, process_alive=lambda _record: False)
    assert int(service.next_experiment_id()[-4:]) > deleted_number
    assert int(reloaded.next_experiment_id()[-4:]) > deleted_number


def _write_fold_checkpoints(run_dir):
    """fold の定期保存・選択モデルと一時ファイルを置く。"""
    files = {
        "checkpoints/fold_1/epoch_010.pt": b"periodic1",
        "checkpoints/fold_2/epoch_010.pt": b"periodic2",
        "checkpoints/fold_1/selected.pt": b"selected",
        "tmp_pred/fold_1/epoch_005/i0.png": b"pred",
        "tmp_ckpt/fold_1/epoch_005.pt": b"tmpckpt",
    }
    for relative, content in files.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    sidecar = run_dir / "checkpoints/fold_1/epoch_010.pt.json"
    sidecar.write_text(json.dumps({"sha256": "a" * 64}), encoding="utf-8")
    return files


def _groups(plan):
    return {(group.attempt, group.category): group for group in plan}


def _deleting_entry(path, category):
    return {
        "path": path,
        "category": category,
        "size": 1,
        "sha256": None,
        "state": "deleting",
        "requested_at": "2026-09-29T00:00:00+09:00",
        "deleted_at": None,
    }


def test_cleanup_plan_detects_categories_and_prune_records_deleted(tmp_path):
    service, experiment = _completed_service(tmp_path)
    expid = experiment.experiment_id
    run_dir = tmp_path / "experiments" / expid / "runs" / "attempt_001"
    _write_fold_checkpoints(run_dir)

    groups = _groups(service.artifact_cleanup_plan([expid]))
    assert groups[(1, "fold_periodic")].n_files == 2
    assert groups[(1, "fold_periodic")].size_bytes == len(b"periodic1") + len(b"periodic2")
    assert groups[(1, "fold_selected")].deletable
    assert groups[(1, "final")].deletable
    temporary = groups[(1, "temporary")]
    assert temporary.n_files == 2 and not temporary.deletable and temporary.reason

    result = service.prune_artifacts([expid], ["fold_periodic", "final", "temporary"])

    assert result.n_files == 3
    assert result.freed_bytes == len(b"periodic1") + len(b"periodic2") + len(b"weights")
    assert [group.category for group in result.skipped] == ["temporary"]
    assert not (run_dir / "checkpoints/fold_1/epoch_010.pt").exists()
    assert not (run_dir / "checkpoints/final.pt").exists()
    assert (run_dir / "checkpoints/final.pt.json").exists()
    assert (run_dir / "checkpoints/fold_1/epoch_010.pt.json").exists()
    assert (run_dir / "checkpoints/fold_1/selected.pt").exists()
    assert (run_dir / "tmp_ckpt/fold_1/epoch_005.pt").exists()
    record = json.loads((run_dir / "pruned.json").read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in record["entries"]}
    assert set(entries) == {
        "checkpoints/fold_1/epoch_010.pt",
        "checkpoints/fold_2/epoch_010.pt",
        "checkpoints/final.pt",
    }
    assert all(entry["state"] == "deleted" and entry["deleted_at"] for entry in entries.values())
    assert entries["checkpoints/final.pt"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert entries["checkpoints/fold_1/epoch_010.pt"]["sha256"] == "a" * 64
    assert entries["checkpoints/fold_2/epoch_010.pt"]["sha256"] is None
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert "checkpoints/final.pt" in service.pruned_paths(expid, 1)
    with pytest.raises(ValueError, match="成果物の整理で削除されています"):
        service.create_candidate_snapshot(expid, attempt=1)

    # 欠けたチェックポイントがあっても、履歴の作り直しと状態の復元は失敗しない
    restored = TrainingService(tmp_path, process_alive=lambda _record: False)
    restored.recover()
    assert restored.get_experiment(expid).status == "completed"
    assert "checkpoints/final.pt" in restored.pruned_paths(expid, 1)


def test_cleanup_temporary_only_for_stopped_or_failed_attempt(tmp_path):
    service, experiment = _queued_service(tmp_path)
    expid = experiment.experiment_id
    prepared = service.prepare_training_run(expid, expid)
    run_dir = Path(prepared.run_dir)
    _write_fold_checkpoints(run_dir)
    service.request_training_stop(expid, 1, "user_stop")
    assert service.conclude_training_run(expid, 1).status == "stopped"
    service.finish_training_queue_item(expid, "stopped")

    group = _groups(service.artifact_cleanup_plan([expid]))[(1, "temporary")]
    assert group.deletable
    result = service.prune_artifacts([expid], ["temporary"])
    assert result.n_files == 2 and not result.skipped
    assert not [path for path in run_dir.glob("tmp_*/**/*") if path.is_file()]
    assert (run_dir / "checkpoints/fold_1/epoch_010.pt").exists()
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "stopped"


def test_cleanup_refuses_running_queued_and_unconcluded_attempts(tmp_path):
    service, experiment = _queued_service(tmp_path)
    expid = experiment.experiment_id
    prepared = service.prepare_training_run(expid, expid)
    run_dir = Path(prepared.run_dir)
    _write_valid_result(run_dir)
    _write_fold_checkpoints(run_dir)

    running = service.prune_artifacts([expid], list(service.ARTIFACT_CATEGORIES))
    assert running.n_files == 0
    assert running.skipped and all("学習中" in group.reason for group in running.skipped)
    assert (run_dir / "checkpoints/final.pt").exists()

    # 実験が学習中でなくても、status.json のない試行は確定していないので消さない
    service.experiments[expid].status = "stopped"
    service.queue["rows"] = []
    unconcluded = service.prune_artifacts([expid], ["final"])
    assert unconcluded.n_files == 0
    assert "確定していない" in unconcluded.skipped[0].reason
    assert (run_dir / "checkpoints/final.pt").exists()

    assert service.conclude_training_run(expid, 1, JobExit(returncode=0)).status == "completed"
    service.add_training_retry_reservation(expid)
    queued = service.prune_artifacts([expid], ["final"])
    assert queued.n_files == 0
    assert "待機中" in queued.skipped[0].reason
    assert (run_dir / "checkpoints/final.pt").exists()
    assert not (run_dir / "pruned.json").exists()


def test_cleanup_counts_hard_linked_selected_checkpoint_once(tmp_path):
    service, experiment = _completed_service(tmp_path)
    expid = experiment.experiment_id
    run_dir = tmp_path / "experiments" / expid / "runs" / "attempt_001"
    periodic = run_dir / "checkpoints" / "fold_1" / "epoch_010.pt"
    periodic.parent.mkdir(parents=True)
    periodic.write_bytes(b"x" * 100)
    selected = periodic.with_name("selected.pt")
    try:
        os.link(periodic, selected)
    except OSError:
        pytest.skip("この環境ではハードリンクを作れません")

    groups = _groups(service.artifact_cleanup_plan([expid]))
    assert groups[(1, "fold_selected")].size_bytes == 100
    assert groups[(1, "fold_selected")].shared_bytes == 100
    assert groups[(1, "fold_periodic")].shared_bytes == 100
    assert groups[(1, "final")].shared_bytes == 0
    assert service.estimate_freed_bytes([expid], ["fold_selected"]) == 0
    assert service.estimate_freed_bytes([expid], ["fold_selected", "fold_periodic"]) == 100

    only_selected = service.prune_artifacts([expid], ["fold_selected"])
    assert (only_selected.n_files, only_selected.freed_bytes) == (1, 0)
    assert periodic.read_bytes() == b"x" * 100

    selected.unlink(missing_ok=True)
    os.link(periodic, selected)
    both = service.prune_artifacts([expid], ["fold_selected", "fold_periodic"])
    assert (both.n_files, both.freed_bytes) == (2, 100)
    assert not periodic.exists() and not selected.exists()


def test_cleanup_keeps_final_protected_by_comparison(tmp_path):
    service, experiment = _completed_service(tmp_path)
    expid = experiment.experiment_id
    run_dir = tmp_path / "experiments" / expid / "runs" / "attempt_001"
    calls = []

    def protected(experiment_id, attempt):
        calls.append((experiment_id, attempt))
        return "比較候補 RC-001 が参照しています"

    result = service.prune_artifacts([expid], ["final"], protected_final=protected)
    assert calls == [(expid, 1)]
    assert result.n_files == 0
    assert result.skipped[0].reason == "比較候補 RC-001 が参照しています"
    assert (run_dir / "checkpoints/final.pt").exists()
    assert service.create_candidate_snapshot(expid, attempt=1).weights_size == len(b"weights")

    result = service.prune_artifacts([expid], ["final"], protected_final=lambda *_args: "")
    assert result.n_files == 1
    assert not (run_dir / "checkpoints/final.pt").exists()


def test_recovery_finishes_deleting_entries(tmp_path):
    service, experiment = _completed_service(tmp_path)
    expid = experiment.experiment_id
    run_dir = tmp_path / "experiments" / expid / "runs" / "attempt_001"
    _write_fold_checkpoints(run_dir)
    record = {
        "schema": 1,
        "entries": [
            _deleting_entry("checkpoints/final.pt", "final"),
            _deleting_entry("checkpoints/fold_1/selected.pt", "fold_selected"),
            _deleting_entry("../../../outside.pt", "final"),
        ],
    }
    (run_dir / "pruned.json").write_text(json.dumps(record), encoding="utf-8")
    outside = tmp_path / "experiments" / "outside.pt"
    outside.write_bytes(b"x")
    # 手順 2 の途中で止まり、final.pt だけ先に消えていた場合
    (run_dir / "checkpoints/final.pt").unlink()

    restored = TrainingService(tmp_path, process_alive=lambda _record: False)
    restored.recover()

    record = json.loads((run_dir / "pruned.json").read_text(encoding="utf-8"))
    states = {entry["path"]: entry["state"] for entry in record["entries"]}
    assert states["checkpoints/final.pt"] == "deleted"
    assert states["checkpoints/fold_1/selected.pt"] == "deleted"
    assert states["../../../outside.pt"] == "deleting"
    assert not (run_dir / "checkpoints/fold_1/selected.pt").exists()
    assert outside.read_bytes() == b"x"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert restored.get_experiment(expid).status == "completed"


def test_cleanup_never_deletes_through_links_outside_run_dir(tmp_path):
    service, experiment = _completed_service(tmp_path)
    expid = experiment.experiment_id
    run_dir = tmp_path / "experiments" / expid / "runs" / "attempt_001"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "epoch_010.pt").write_bytes(b"outside")
    (outside_dir / "selected.pt").write_bytes(b"outside")
    link = run_dir / "checkpoints" / "fold_3"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        _winapi = pytest.importorskip("_winapi")
        _winapi.CreateJunction(str(outside_dir), str(link))

    plan = _groups(service.artifact_cleanup_plan([expid]))
    assert (1, "fold_periodic") not in plan
    assert (1, "fold_selected") not in plan
    service.prune_artifacts([expid], list(service.ARTIFACT_CATEGORIES))
    assert (outside_dir / "epoch_010.pt").read_bytes() == b"outside"
    assert (outside_dir / "selected.pt").read_bytes() == b"outside"


def test_candidate_snapshot_uses_explicit_attempt_and_records_weights(tmp_path):
    service, experiment = _completed_service(tmp_path)
    expid = experiment.experiment_id
    service.add_training_retry_reservation(expid)
    row = service.take_next_training_queue_item()
    prepared = service.prepare_training_run(expid, row.queue_id, retry=True)
    run_dir = Path(prepared.run_dir)
    (run_dir / "error.json").write_text(json.dumps({"message": "x"}), encoding="utf-8")
    assert service.conclude_training_run(expid, 2, JobExit(returncode=1)).status == "failed"

    with pytest.raises(ValueError, match="完了した実験のみ"):
        service.create_candidate_snapshot(expid)
    with pytest.raises(ValueError, match="完了していません"):
        service.create_candidate_snapshot(expid, attempt=2)

    snapshot = service.create_candidate_snapshot(expid, attempt=1)
    assert snapshot.attempt == 1
    assert snapshot.run_id == f"{expid}/attempt_001"
    assert snapshot.weights_size == len(b"weights")
    assert snapshot.weights_sha256 == hashlib.sha256(b"weights").hexdigest()
    assert snapshot.training_eval_params["box_detections_per_img"] == 300
    assert snapshot.preprocessing["method"] == "percentile"
    assert snapshot.training_dataset["version"] == "train_v000"

    first = tmp_path / "experiments" / expid / "runs" / "attempt_001" / "checkpoints" / "final.pt"
    first.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="大きさ"):
        service.create_candidate_snapshot(expid, attempt=1)


def test_recovery_does_not_resume_pruning_through_linked_attempt_dir(tmp_path):
    service, experiment = _completed_service(tmp_path)
    expid = experiment.experiment_id
    runs = tmp_path / "experiments" / expid / "runs"
    outside_dir = tmp_path / "outside_attempt"
    outside_dir.mkdir()
    victim = outside_dir / "victim.pt"
    victim.write_bytes(b"outside")
    record = {"schema": 1, "entries": [_deleting_entry("victim.pt", "final")]}
    (outside_dir / "pruned.json").write_text(json.dumps(record), encoding="utf-8")
    link = runs / "attempt_009"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        _winapi = pytest.importorskip("_winapi")
        _winapi.CreateJunction(str(outside_dir), str(link))

    TrainingService(tmp_path, process_alive=lambda _record: False).recover()

    assert victim.read_bytes() == b"outside"
    states = json.loads((outside_dir / "pruned.json").read_text("utf-8"))["entries"]
    assert states[0]["state"] == "deleting"
