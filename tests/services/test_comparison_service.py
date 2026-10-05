"""ComparisonService の保存・復元・評価の読み取り・リリース・振り分けのテスト。"""

import copy
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from foam_cell_analysis.data.dataset_store import DatasetStore
from foam_cell_analysis.services import comparison_service
from foam_cell_analysis.services.comparison_service import (
    ComparisonService,
    is_recovered_release,
    resolve_recorded_path,
    validate_evaluation_result,
)
from foam_cell_analysis.services.models import CandidateSnapshot

MASK_RCNN_EVAL = {
    "box_score_thresh": 0.5,
    "box_nms_thresh": 0.5,
    "box_detections_per_img": 300,
    "mask_thresh": 0.5,
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dataset(
    root, version, purpose, classifications, base_validation_version=None, *, hash_offset=0
):
    folder = root / "datasets" / version
    folder.mkdir(parents=True)
    info = {"purpose": purpose, "status": "RELEASED", "dataset_version": version}
    if base_validation_version:
        info["base_validation_version"] = base_validation_version
    (folder / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    with (folder / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "source_relpath", "channel", "classification"])
        for index, name in enumerate(classifications):
            writer.writerow([f"{version}_{index}", f"g/{index}.png", "A", name])
    with (folder / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "image_path", "image_sha256", "mask_path"])
        for index in range(len(classifications)):
            writer.writerow([f"{version}_{index}", "i.png", f"{index + hash_offset:064x}", "m.png"])


class FakeTraining:
    """create_candidate_snapshot と dataset_store だけを持つ TrainingService の代役。"""

    app_version = "0.1.0"

    def __init__(self, root):
        self.workspace_root = root
        self.dataset_store = DatasetStore(root)
        self.snapshots = {}

    def create_candidate_snapshot(self, experiment_id, *, attempt):
        return copy.deepcopy(self.snapshots[(experiment_id, attempt)])

    def add_attempt(self, experiment_id, attempt, *, weights=b"weights-1", eval_params=None):
        run_dir = self.workspace_root / "experiments" / experiment_id / "runs"
        run_dir = run_dir / f"attempt_{attempt:03d}"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "checkpoints" / "final.pt").write_bytes(weights)
        self.snapshots[(experiment_id, attempt)] = CandidateSnapshot(
            experiment_id=experiment_id,
            attempt=attempt,
            selected_epoch=10,
            run_id=f"{experiment_id}/attempt_{attempt:03d}",
            checkpoint_path="checkpoints/final.pt",
            oof_evaluation={
                "ap": 0.8,
                "n_images": 4,
                "per_class": {"分類A": {"ap": 0.7, "n_images": 2}},
            },
            experiment_config={"model": {"type": "mask_rcnn", "num_classes": 2}},
            weights_size=len(weights),
            weights_sha256=_sha(weights),
            training_eval_params=copy.deepcopy(
                MASK_RCNN_EVAL if eval_params is None else eval_params
            ),
            preprocessing={"method": "percentile"},
            training_dataset={"version": "train_v000"},
        )
        return run_dir


@pytest.fixture
def env(tmp_path):
    _dataset(tmp_path, "train_v000", "train", ["分類A", "分類B"], "val_v000")
    _dataset(tmp_path, "val_v000", "val", ["分類A", "分類C"])
    training = FakeTraining(tmp_path)
    training.add_attempt("exp_0001", 1)
    service = ComparisonService(tmp_path, training)
    return service, training


def _default_config(service):
    return service.create_inference_config(
        "mask_rcnn", service.default_inference_params("mask_rcnn")
    )


def _write_evaluation(
    service,
    candidate_id,
    evaluation_id,
    *,
    version="val_v000",
    status="completed",
    contamination=None,
    input_fingerprint=None,
):
    """評価フォルダを直接書く。contamination を渡すと旧形式（schema 1）の評価になる。"""
    schema = 2 if contamination is None else 1
    run_dir = service.candidates_root / candidate_id / "evaluations" / version / evaluation_id
    (run_dir / "predictions").mkdir(parents=True)
    item_ids = [item.item_id for item in service.dataset_store.select_evaluation_items(version)]
    fingerprint = input_fingerprint or service.evaluation_input_fingerprint(candidate_id, version)
    run_id = f"{candidate_id}/{version}/{evaluation_id}"
    spec = {
        "schema": schema,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "evaluation_id": evaluation_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "input_fingerprint": fingerprint,
        "validation": {"version": version, "item_ids": item_ids},
        "metric": {"id": "cellpose_ap_iou50_95_image_mean_v1"},
    }
    (run_dir / "run_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    if status is None:
        return run_dir
    predictions = {}
    for index, item_id in enumerate(item_ids):
        path = run_dir / "predictions" / f"{item_id}.png"
        Image.fromarray(np.full((4, 5), index + 1, dtype=np.uint16)).save(path)
        data = path.read_bytes()
        predictions[item_id] = {
            "path": f"predictions/{item_id}.png",
            "shape": [4, 5],
            "dtype": "uint16",
            "bytes": len(data),
            "sha256": _sha(data),
        }
    (run_dir / "per_image.csv").write_text(
        "item_id,ap\n" + "".join(f"{item},0.5\n" for item in item_ids), encoding="utf-8"
    )
    np.savez(run_dir / "eval_counts.npz", tp=np.zeros((len(item_ids), 10)))
    (run_dir / "instances.csv").write_text("item_id,side\n", encoding="utf-8")
    artifacts = {
        key: {"path": name, "sha256": _sha((run_dir / name).read_bytes())}
        for key, name in (
            ("per_image_csv", "per_image.csv"),
            ("eval_counts", "eval_counts.npz"),
            ("instances_csv", "instances.csv"),
        )
    }
    result = {
        "schema": schema,
        "run_id": run_id,
        "evaluation_id": evaluation_id,
        "validation_version": version,
        "input_fingerprint": fingerprint,
        "overall": {"ap": 0.6, "n_images": len(item_ids)},
        "per_class": {
            "分類A": {"ap": 0.7, "n_images": len(item_ids)},
            "分類C": {"ap": None, "n_images": 0},
        },
        "metric": {"id": "cellpose_ap_iou50_95_image_mean_v1"},
        "predictions": predictions,
        **artifacts,
        "completed_at": "2026-09-29T10:00:00+09:00",
    }
    if contamination is not None:
        result["contamination"] = {"status": contamination, "pairs": []}
    if status == "completed":
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    return run_dir


def test_inference_config_range_and_reuse(env):
    service, _training = env
    with pytest.raises(ValueError, match="検出スコア閾値は 0〜1"):
        service.create_inference_config("mask_rcnn", {"box_score_thresh": 1.5})
    with pytest.raises(ValueError, match="最大検出数は整数"):
        service.create_inference_config("mask_rcnn", {"box_detections_per_img": 10.5})
    with pytest.raises(ValueError, match="セル確率閾値は -6〜6"):
        service.create_inference_config("cellpose", {"cellprob_threshold": -7})
    with pytest.raises(ValueError, match="変更できない"):
        service.create_inference_config("cellpose", {"min_size": 3})
    first = _default_config(service)
    assert first.config_id == "infer_v001"
    # 1/2 と 0.5、300 と 300.0 は同じ設定として再利用する
    again = service.create_inference_config(
        "mask_rcnn",
        {"box_score_thresh": 1 / 2, "box_nms_thresh": 0.5, "box_detections_per_img": 300.0},
    )
    assert again.config_id == first.config_id
    other = service.create_inference_config("mask_rcnn", {"box_score_thresh": 0.3})
    assert other.config_id == "infer_v002"
    assert [c.config_id for c in service.list_inference_configs("mask_rcnn")] == [
        "infer_v001",
        "infer_v002",
    ]
    assert service.list_inference_configs("cellpose") == []


def test_candidate_duplicate_by_fingerprint_across_config_ids(env):
    service, _training = env
    first = _default_config(service)
    # ID が違うだけの同一設定（手で置かれた古いファイルなど）
    duplicate = json.loads((service.configs_root / "infer_v001.json").read_text("utf-8"))
    duplicate["config_id"] = "infer_v002"
    (service.configs_root / "infer_v002.json").write_text(json.dumps(duplicate), "utf-8")
    candidate = service.add_candidate("exp_0001", 1, first.config_id)
    assert candidate.candidate_id == "RC-001"
    assert (service.candidates_root / "RC-001" / "candidate.json").is_file()
    with pytest.raises(ValueError, match="既に RC-001 として登録済みです"):
        service.add_candidate("exp_0001", 1, "infer_v002")
    service.reject_candidate("RC-001")
    assert service.add_candidate("exp_0001", 1, "infer_v002").candidate_id == "RC-002"
    assert not list(service.candidates_root.glob(".preparing_*"))


def test_oof_applicability_three_values(env):
    service, training = env
    training.add_attempt("exp_0002", 1, weights=b"legacy", eval_params={})
    default = _default_config(service)
    changed = service.create_inference_config("mask_rcnn", {"box_score_thresh": 0.3})
    matching = service.add_candidate("exp_0001", 1, default.config_id)
    different = service.add_candidate("exp_0001", 1, changed.config_id)
    unknown = service.add_candidate("exp_0002", 1, default.config_id)
    assert matching.oof_applicability == "matching" and matching.oof_reason == ""
    assert different.oof_applicability == "different"
    assert different.oof_reason == "推論設定が学習時と異なります"
    assert unknown.oof_applicability == "unknown"
    assert unknown.oof_reason == "学習時の評価条件を確認できません"
    # OOF の値は applicability によらず残す
    assert different.oof_evaluation.overall_map == pytest.approx(0.8)
    assert different.oof_evaluation.n_images == 4
    record = service.get_candidate_record(unknown.candidate_id)
    assert record["source"]["training_eval_params_filled"] is True
    assert record["effective_params"]["mask_thresh"] == 0.5
    assert service.get_candidate_record(different.candidate_id)["effective_params"] == {
        **MASK_RCNN_EVAL,
        "box_score_thresh": 0.3,
    }


def test_model_type_mismatch_rejected(env):
    service, _training = env
    config = service.create_inference_config("cellpose", {})
    with pytest.raises(ValueError, match="モデル種類"):
        service.add_candidate("exp_0001", 1, config.config_id)


def test_adoption_picks_latest_completed_matching(env):
    service, _training = env
    candidate = service.add_candidate("exp_0001", 1, _default_config(service).config_id)
    cid = candidate.candidate_id
    assert service.get_candidate_evaluation(cid, "val_v000") is None
    _write_evaluation(service, cid, "eval_001")
    _write_evaluation(service, cid, "eval_002", status="failed")
    _write_evaluation(service, cid, "eval_003", input_fingerprint="0" * 64)
    _write_evaluation(service, cid, "eval_004", status=None)  # 実行中（status.json なし）
    adopted = service.get_candidate_evaluation(cid, "val_v000")
    assert adopted.evaluation_id == "eval_001"
    assert adopted.status == "completed" and not adopted.broken
    assert adopted.evaluation.overall_map == pytest.approx(0.6)
    assert adopted.evaluation.n_images == 2
    assert adopted.evaluation.per_class["分類C"] == (None, 0)
    records = service.list_candidate_evaluations(cid, "val_v000")
    assert [item.status for item in records] == ["completed", "failed", "completed", "running"]
    _write_evaluation(service, cid, "eval_005")
    assert service.get_candidate_evaluation(cid, "val_v000").evaluation_id == "eval_005"
    assert service.get_candidate(cid).evaluations["val_v000"].overall_map == pytest.approx(0.6)


def test_broken_prediction_detected(env):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    run_dir = _write_evaluation(service, cid, "eval_001")
    item_id = "val_v000_0"
    assert service.get_candidate_prediction(cid, "eval_001", item_id).dtype == np.uint16
    assert service.verify_evaluation(cid, "eval_001")
    prediction = run_dir / "predictions" / f"{item_id}.png"
    data = bytearray(prediction.read_bytes())
    data[-5] ^= 0xFF  # 同じ大きさで中身だけ変える
    prediction.write_bytes(bytes(data))
    assert not service.get_candidate_evaluation(cid, "val_v000").broken  # サイズだけでは見えない
    assert not service.verify_evaluation(cid, "eval_001")
    with pytest.raises(ValueError, match="壊れています"):
        service.get_candidate_prediction(cid, "eval_001", item_id)
    prediction.write_bytes(b"short")
    adopted = service.get_candidate_evaluation(cid, "val_v000")
    assert adopted.broken
    assert "val_v000" not in service.get_candidate(cid).evaluations
    service.list_candidates()  # 壊れた結果があっても一覧は例外にならない


def test_path_traversal_rejected(env, tmp_path):
    service, _training = env
    for bad in ("../x.png", "..\\x.png", "/abs/x.png", "C:\\x.png", "C:x.png", "\\x.png", ""):
        with pytest.raises(ValueError):
            resolve_recorded_path(tmp_path / "base", bad)
    assert resolve_recorded_path(tmp_path, "a/b.png") == tmp_path / "a" / "b.png"
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    run_dir = _write_evaluation(service, cid, "eval_001")
    result = json.loads((run_dir / "result.json").read_text("utf-8"))
    outside = run_dir.parent / "outside.png"
    outside.write_bytes((run_dir / "predictions" / "val_v000_0.png").read_bytes())
    result["predictions"]["val_v000_0"]["path"] = "../outside.png"
    (run_dir / "result.json").write_text(json.dumps(result), "utf-8")
    assert not validate_evaluation_result(run_dir)
    with pytest.raises(ValueError):
        service.get_candidate_prediction(cid, "eval_001", "val_v000_0")


def test_release_copies_weights_independently(env):
    service, training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.save_external_analysis(cid, "eval_001", {"val_v000_0": 12.3}, software="X")
    model = service.release_candidate(cid, "eval_001", "初回")
    assert model.model_id == "model_001"
    assert model.validation_dataset == "val_v000"
    assert model.evaluation_result.overall_map == pytest.approx(0.6)
    assert model.evaluation_id == "eval_001"
    assert model.external_summary["mean"] == pytest.approx(12.3)
    released = service.releases_root / "model_001"
    release = json.loads((released / "release.json").read_text("utf-8"))
    assert release["weights"]["sha256"] == _sha(b"weights-1")
    assert release["dataset_pair"] == {
        "training_version": "train_v000",
        "validation_version": "val_v000",
    }
    assert "contamination" not in release["evaluation"]
    assert release["external_results"][0]["evaluation_id"] == "eval_001"
    # 元の final.pt を書き換えてもリリースは変わらない（ハードリンクではない）
    source = training.workspace_root / "experiments/exp_0001/runs/attempt_001/checkpoints/final.pt"
    with source.open("r+b") as stream:
        stream.write(b"XX")
    assert (released / "model.pt").read_bytes() == b"weights-1"
    candidate = service.get_candidate(cid)
    assert candidate.status == "released" and candidate.released_model_id == "model_001"
    assert [item.model_id for item in service.list_released_models()] == ["model_001"]
    with pytest.raises(ValueError, match="リリース済み"):
        service.save_external_analysis(cid, "eval_001", {})
    with pytest.raises(ValueError, match="候補状態"):
        service.release_candidate(cid, "eval_001")


def test_released_candidate_can_be_reevaluated_without_changing_public_release(env):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.release_candidate(cid, "eval_001", "公開時コメント")
    release_path = service.releases_root / "model_001" / "release.json"
    before_release = release_path.read_bytes()
    before_weight = (release_path.parent / "model.pt").read_bytes()
    original_weight = service._weights_path(service._read_candidate(cid)["source"])
    original_weight.unlink()

    # EvaluationRunner sets its busy callback before asking the backend to prepare the run.
    service.is_evaluation_active = lambda _candidate_id: True
    prepared = service.prepare_evaluation_run(cid)
    service.is_evaluation_active = lambda _candidate_id: False
    assert prepared.run_id.endswith("eval_002")
    spec = json.loads((Path(prepared.run_dir) / "run_spec.json").read_text("utf-8"))
    assert spec["weights"]["path"] == "releases/model_001/model.pt"
    _write_evaluation(service, cid, "eval_002")
    current = service.get_candidate_evaluation(cid)

    assert current.evaluation_id == "eval_002"
    assert service.get_candidate(cid).status == "released"
    assert release_path.read_bytes() == before_release
    assert (release_path.parent / "model.pt").read_bytes() == before_weight
    assert service.get_release_record("model_001")["evaluation"]["evaluation_id"] == "eval_001"
    values = {
        item.item_id: 2.5 for item in service.dataset_store.select_evaluation_items("val_v000")
    }
    service.save_external_analysis(cid, "eval_002", values, software="再評価")
    assert release_path.read_bytes() == before_release


def test_release_archive_delete_preserves_history_and_blocks_routing(env):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.release_candidate(cid, "eval_001")
    release_path = service.releases_root / "model_001" / "release.json"
    release_hash = _sha(release_path.read_bytes())
    size = (release_path.parent / "model.pt").stat().st_size

    service.set_release_archived("model_001", True)
    assert service.list_released_models() == []
    assert service.list_released_models(include_archived=True)[0].lifecycle_status == "archived"
    with pytest.raises(ValueError, match="保管中"):
        service.apply_routing({"分類A": "model_001"}, expected_revision=0)
    assert service.set_release_archived("model_001", False).lifecycle_status == "active"
    service.apply_routing({"分類A": "model_001"}, expected_revision=0)
    with pytest.raises(ValueError, match="使用中"):
        service.estimate_release_delete_bytes("model_001")
    service.apply_routing({"分類A": None}, expected_revision=1)
    service.is_evaluation_active = lambda _candidate_id: True
    with pytest.raises(ValueError, match="評価中または評価待ち"):
        service.estimate_release_delete_bytes("model_001")
    service.is_evaluation_active = lambda _candidate_id: False
    assert service.estimate_release_delete_bytes("model_001") == size
    assert service.delete_released_model("model_001") == size

    assert not (release_path.parent / "model.pt").exists()
    assert _sha(release_path.read_bytes()) == release_hash
    assert service.list_released_models() == []
    archived = service.list_released_models(include_deleted=True)[0]
    assert archived.lifecycle_status == "deleted"
    assert archived.evaluation_id == "eval_001"
    assert service.get_candidate(cid).status == "released"
    assert service.get_candidate_prediction(cid, "eval_001", "val_v000_0").size > 0
    with pytest.raises(ValueError, match="削除済み"):
        service.get_release_record("model_001")
    with pytest.raises(ValueError, match="保管中"):
        service.apply_routing({"分類A": "model_001"}, expected_revision=2)


def test_release_delete_failure_restores_archived_weight_and_state(env, monkeypatch):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.release_candidate(cid, "eval_001")
    service.set_release_archived("model_001", True)
    weight = service.releases_root / "model_001" / "model.pt"
    original_unlink = type(weight).unlink

    def fail_staged_unlink(path, *args, **kwargs):
        if path.name == "model.pt" and ".deleting" in path.parts:
            raise PermissionError("simulated delete failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(weight), "unlink", fail_staged_unlink)
    with pytest.raises(PermissionError, match="simulated"):
        service.delete_released_model("model_001")

    assert weight.is_file()
    assert service.list_released_models() == []
    assert service.list_released_models(include_archived=True)[0].lifecycle_status == "archived"


def test_release_move_failure_restores_archived_state(env, monkeypatch):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.release_candidate(cid, "eval_001")
    service.set_release_archived("model_001", True)
    weight = service.releases_root / "model_001" / "model.pt"
    real_replace = comparison_service.os.replace

    def fail_move(source, destination):
        if Path(source) == weight:
            raise PermissionError("simulated move failure")
        return real_replace(source, destination)

    monkeypatch.setattr(comparison_service.os, "replace", fail_move)
    with pytest.raises(PermissionError, match="simulated move"):
        service.delete_released_model("model_001")

    assert weight.is_file()
    assert service.list_released_models() == []
    assert service.list_released_models(include_archived=True)[0].lifecycle_status == "archived"
    assert not (service.releases_root / ".deleting" / "model_001").exists()


def test_release_blocked_conditions(env):
    service, training = env
    config = _default_config(service)
    cid = service.add_candidate("exp_0001", 1, config.config_id).candidate_id
    # 旧形式で学習混入が見つかっていた評価は、閲覧はできるが新規リリースには使えない
    _write_evaluation(service, cid, "eval_001", contamination="found")
    assert service.get_candidate_evaluation(cid).contamination_found
    with pytest.raises(ValueError, match="同じ画像が学習データに見つかっている"):
        service.release_candidate(cid, "eval_001")
    _write_evaluation(service, cid, "eval_002", status="failed")
    with pytest.raises(ValueError, match="完了した評価"):
        service.release_candidate(cid, "eval_002")
    run_dir = _write_evaluation(service, cid, "eval_003")
    with pytest.raises(ValueError, match="評価中"):
        service.release_candidate(cid, "eval_003", is_evaluation_active=lambda _id: True)
    (run_dir / "instances.csv").write_text("changed\n", "utf-8")
    with pytest.raises(ValueError, match="壊れています"):
        service.release_candidate(cid, "eval_003")
    assert not list(service.releases_root.glob("*"))
    assert service.get_candidate(cid).status == "candidate"


def test_release_recovery_after_crash_between_rename_and_candidate_update(env):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.release_candidate(cid, "eval_001")
    # 改名の直後（candidate.json の更新前）に落ちた状態を作る
    path = service.candidates_root / cid / "candidate.json"
    record = json.loads(path.read_text("utf-8"))
    record["status"] = "candidate"
    record.pop("released_model_id")
    path.write_text(json.dumps(record), "utf-8")
    leftover = service.releases_root / ".preparing_abc"
    leftover.mkdir()
    (leftover / "model.pt").write_bytes(b"partial")
    (service.configs_root / "infer_v009.json.tmp").write_text("{", "utf-8")
    fresh = ComparisonService(service.workspace_root, service.training_service)
    assert fresh.recover() == [cid]
    assert not leftover.exists()
    assert not list(service.configs_root.glob("*.tmp"))
    candidate = fresh.get_candidate(cid)
    assert candidate.status == "released" and candidate.released_model_id == "model_001"


def test_routing_all_or_nothing_and_revision_conflict(env):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.release_candidate(cid, "eval_001")
    assert service.list_routing_classifications() == ["分類A", "分類B", "分類C"]
    state = service.get_routing_state()
    assert state.revision == 0 and state.assignments == {}
    with pytest.raises(ValueError, match="未登録"):
        service.apply_routing({"分類A": "model_001", "分類B": "model_099"}, expected_revision=0)
    with pytest.raises(ValueError, match="対象にない分類"):
        service.apply_routing({"分類Z": "model_001"}, expected_revision=0)
    assert service.get_routing_state().revision == 0
    assert service.get_routing() == {}
    state = service.apply_routing({"分類A": "model_001", "分類B": None}, expected_revision=0)
    assert state.revision == 1
    assert state.assignments == {"分類A": "model_001", "分類B": None}
    with pytest.raises(ValueError, match="別の操作で変更されました"):
        service.apply_routing({"分類C": "model_001"}, expected_revision=0)
    history = service.list_routing_history()
    assert [(h.classification, h.before_model_id, h.after_model_id) for h in history] == [
        ("分類A", None, "model_001")
    ]
    assert service.resolve_model("分類A").model_id == "model_001"
    assert service.resolve_model("分類B") is None
    assert service.resolve_model(None) is None
    (service.releases_root / "model_001" / "model.pt").unlink()
    with pytest.raises(ValueError, match="model_001"):
        service.resolve_model("分類A")


def test_protection_reasons_and_reject(env):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    assert cid in service.protected_final_reason("exp_0001", 1)
    assert service.protected_final_reason("exp_0001", 2) == ""
    assert cid in service.experiment_reference_reason("exp_0001")
    with pytest.raises(ValueError, match="評価中"):
        service.reject_candidate(cid, is_evaluation_active=lambda _id: True)
    service.reject_candidate(cid)
    assert service.protected_final_reason("exp_0001", 1) == ""
    assert service.experiment_reference_reason("exp_0001") == ""
    with pytest.raises(ValueError, match="候補状態"):
        service.reject_candidate(cid)


def test_candidate_fixes_paired_validation_version(env):
    """候補は作成時に学習用の版の組の検証用の版を固定し、採用する評価もその版だけ。"""
    service, training = env
    config = _default_config(service)
    candidate = service.add_candidate("exp_0001", 1, config.config_id)
    record = service.get_candidate_record(candidate.candidate_id)
    assert candidate.validation_version == "val_v000"
    assert candidate.training_version == "train_v000"
    assert record["validation_version"] == "val_v000"
    assert record["dataset_pair"]["training_version"] == "train_v000"
    # 別の検証用の版で行った旧評価は履歴として読めるが、採用しない
    _dataset(service.workspace_root, "val_v001", "val", ["分類A"], hash_offset=100)
    _write_evaluation(service, candidate.candidate_id, "eval_001", version="val_v001")
    assert service.get_candidate_evaluation(candidate.candidate_id) is None
    assert [item.validation_version for item in service.list_candidate_evaluations("RC-001")] == [
        "val_v001"
    ]
    with pytest.raises(ValueError, match="val_v001 で行われたため"):
        service.release_candidate(candidate.candidate_id, "eval_001")
    # 組のない学習用の版の試行は候補にできない
    _dataset(service.workspace_root, "train_v001", "train", ["分類A"], hash_offset=200)
    training.add_attempt("exp_0002", 1, weights=b"weights-2")
    training.snapshots[("exp_0002", 1)].training_dataset = {"version": "train_v001"}
    with pytest.raises(ValueError, match="組になる検証用データセットが記録されていません"):
        service.add_candidate("exp_0002", 1, config.config_id)
    assert [item.candidate_id for item in service.list_candidates()] == ["RC-001"]


def _legacy_candidate(service, training, *, train_hash_offset):
    """組の記録（validation_version）がない旧形式の候補 RC-001 を作る。"""
    root = service.workspace_root
    shutil.rmtree(root / "datasets" / "train_v000")
    _dataset(
        root, "train_v000", "train", ["分類A", "分類B"], "val_v000", hash_offset=train_hash_offset
    )
    snapshot = training.snapshots[("exp_0001", 1)]
    snapshot.training_dataset = {
        "version": "train_v000",
        "sha256": {
            name: _sha((root / "datasets" / "train_v000" / name).read_bytes())
            for name in ("manifest.csv", "metadata.csv")
        },
    }
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    path = service.candidates_root / cid / "candidate.json"
    record = json.loads(path.read_text("utf-8"))
    record.pop("validation_version")
    record.pop("dataset_pair")
    path.write_text(json.dumps(record, ensure_ascii=False), "utf-8")
    return cid, path


def test_legacy_candidate_is_paired_once_after_duplicate_check(env):
    service, training = env
    cid, path = _legacy_candidate(service, training, train_hash_offset=100)
    _write_evaluation(service, cid, "eval_001")
    # 読み取りでは補完・推測しない（評価・リリースもしない）
    assert service.get_candidate(cid).validation_version is None
    assert service.get_candidate_evaluation(cid) is None
    with pytest.raises(ValueError, match="組を確認できていない"):
        service.prepare_evaluation_run(cid)
    assert "validation_version" not in json.loads(path.read_text("utf-8"))

    fresh = ComparisonService(service.workspace_root, training)
    fresh.recover()
    assert fresh.migrated_candidates == [cid]
    record = json.loads(path.read_text("utf-8"))
    assert record["validation_version"] == "val_v000"
    assert record["dataset_pair"]["method"] == "migrated"
    assert record["dataset_pair"]["duplicate_check"]["same_images"] == 0
    assert fresh.get_candidate(cid).validation_version == "val_v000"
    assert fresh.get_candidate_evaluation(cid).evaluation_id == "eval_001"
    # 一度補完した候補は、次の起動で書き換えない
    before = path.read_bytes()
    ComparisonService(service.workspace_root, training).recover()
    assert path.read_bytes() == before


@pytest.mark.parametrize("problem", ["same_images", "no_pair", "changed_training"])
def test_legacy_candidate_is_not_paired_when_unconfirmed(env, problem):
    service, training = env
    cid, path = _legacy_candidate(
        service, training, train_hash_offset=0 if problem == "same_images" else 100
    )
    info_path = service.workspace_root / "datasets" / "train_v000" / "dataset_info.json"
    if problem == "no_pair":
        info = json.loads(info_path.read_text("utf-8"))
        info.pop("base_validation_version")
        info_path.write_text(json.dumps(info), "utf-8")
    elif problem == "changed_training":
        metadata = service.workspace_root / "datasets" / "train_v000" / "metadata.csv"
        metadata.write_text(metadata.read_text("utf-8") + "\n", "utf-8")
    before = path.read_bytes()
    fresh = ComparisonService(service.workspace_root, training)
    fresh.recover()
    assert fresh.migrated_candidates == []
    assert path.read_bytes() == before
    reason = fresh.get_candidate(cid).pairing_issue
    expected = {
        "same_images": "同じ画像 2 件",
        "no_pair": "記録されていません",
        "changed_training": "学習時の内容と一致しません",
    }[problem]
    assert expected in reason
    with pytest.raises(ValueError, match=expected):
        fresh.prepare_evaluation_run(cid)


def test_external_analysis_is_validated_and_saved_as_snapshots(env):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    for values, unit, message in (
        ({"other": 1.0}, "µm", "対象にない画像"),
        ({"val_v000_0": -1.0}, "µm", "負"),
        ({"val_v000_0": float("nan")}, "µm", "有限"),
        ({"val_v000_0": float("inf")}, "µm", "有限"),
        ({"val_v000_0": "12"}, "µm", "数値"),
        ({"val_v000_0": 1.0}, "mm", "単位"),
    ):
        with pytest.raises(ValueError, match=message):
            service.save_external_analysis(cid, "eval_001", values, unit=unit)
    assert service.list_external_results(cid) == []
    candidate = service.save_external_analysis(
        cid, "eval_001", {"val_v000_0": 10.0, "val_v000_1": 14.0}, unit="px", software="ImageJ"
    )
    assert candidate.external_summary["mean"] == pytest.approx(12.0)
    assert candidate.external_summary["unit"] == "px"
    assert (candidate.external_summary["n_images"], candidate.external_summary["n_total"]) == (2, 2)
    assert candidate.external_summary["per_class"]["分類C"]["mean"] == pytest.approx(14.0)
    # 保存のたびに全体を 1 件追加する。空欄（None）は値の削除
    candidate = service.save_external_analysis(
        cid, "eval_001", {"val_v000_0": 10.0, "val_v000_1": None}
    )
    records = service.list_external_results(cid)
    assert [item["values"] for item in records] == [
        {"val_v000_0": 10.0, "val_v000_1": 14.0},
        {"val_v000_0": 10.0},
    ]
    assert records[-1]["evaluation_id"] == "eval_001"
    assert records[-1]["validation_version"] == "val_v000"
    assert candidate.external_summary["n_images"] == 1
    assert candidate.external_summary["per_class"]["分類C"]["mean"] is None
    # 旧形式（自由項目）の記録は残したまま読める
    path = service.candidates_root / cid / "external.json"
    value = json.loads(path.read_text("utf-8"))
    value["records"].insert(0, {"record_id": "ext_000", "evaluation_id": "eval_001", "results": []})
    path.write_text(json.dumps(value, ensure_ascii=False), "utf-8")
    assert service.get_candidate(cid).external_summary["n_images"] == 1
    assert len(service.list_external_results(cid)) == 3


def test_rejected_candidate_keeps_external_analysis_editable(env):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.reject_candidate(cid)

    assert service.external_analysis_editable(cid, "eval_001")
    saved = service.save_external_analysis(cid, "eval_001", {"val_v000_0": 12.0})
    assert saved.status == "rejected"
    assert service.list_external_results(cid)[-1]["values"] == {"val_v000_0": 12.0}


def test_broken_external_record_cannot_be_overwritten(env):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    external_path = service.candidates_root / cid / "external.json"
    external_path.write_bytes(b"{broken external")

    with pytest.raises(ValueError, match="外部解析"):
        service.save_external_analysis(cid, "eval_001", {"val_v000_0": 1.0})

    assert external_path.read_bytes() == b"{broken external"


def test_broken_evaluation_result_does_not_block_candidate_settings_copy(env):
    service, _training = env
    candidate = service.add_candidate("exp_0001", 1, _default_config(service).config_id)
    run_dir = _write_evaluation(service, candidate.candidate_id, "eval_001")
    result_path = run_dir / "result.json"
    result_path.write_bytes(b"{broken result")
    original_candidate = (
        service.candidates_root / candidate.candidate_id / "candidate.json"
    ).read_bytes()

    service.recover_evaluations()
    copied = service.copy_candidate_settings(candidate.candidate_id)

    assert copied.candidate_id != candidate.candidate_id
    assert copied.status == "candidate"
    assert copied.evaluations == {}
    assert result_path.read_bytes() == b"{broken result"
    assert (
        service.candidates_root / candidate.candidate_id / "candidate.json"
    ).read_bytes() == original_candidate


def test_candidate_with_malformed_source_protects_experiment_and_weights(env):
    service, _training = env
    candidate = service.add_candidate("exp_0001", 1, _default_config(service).config_id)
    path = service.candidates_root / candidate.candidate_id / "candidate.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["source"] = []
    path.write_text(json.dumps(record), encoding="utf-8")

    assert service.experiment_reference_reason("exp_0001")
    assert service.protected_final_reason("exp_0001", 1)


def test_released_weight_copy_can_seed_new_candidate_when_training_weight_is_pruned(env):
    service, _training = env
    candidate = service.add_candidate("exp_0001", 1, _default_config(service).config_id)
    _write_evaluation(service, candidate.candidate_id, "eval_001")
    service.release_candidate(candidate.candidate_id, "eval_001")
    original = service.get_candidate_record(candidate.candidate_id)
    original_bytes = (
        service.candidates_root / candidate.candidate_id / "candidate.json"
    ).read_bytes()
    source_path = service._weights_path(original["source"])
    source_path.unlink()
    service.recovery_issues[candidate.candidate_id] = ("unrecoverable", "評価履歴を読めません")

    copied = service.copy_candidate_settings(candidate.candidate_id)

    copied_record = service.get_candidate_record(copied.candidate_id)
    assert copied_record["source"]["weights"]["release_model_id"] == "model_001"
    assert service._weights_path(copied_record["source"]).is_file()
    assert (
        service.candidates_root / candidate.candidate_id / "candidate.json"
    ).read_bytes() == original_bytes


def test_candidate_list_tolerates_malformed_nested_snapshot_values(env):
    service, _training = env
    candidate = service.add_candidate("exp_0001", 1, _default_config(service).config_id)
    path = service.candidates_root / candidate.candidate_id / "candidate.json"
    value = json.loads(path.read_text("utf-8"))
    value["source"] = []
    value["oof"] = []
    value["effective_params"] = []
    path.write_text(json.dumps(value), encoding="utf-8")

    listed = service.list_candidates()

    assert [item.candidate_id for item in listed] == [candidate.candidate_id]


def _rewrite_result(run_dir, mutate):
    path = run_dir / "result.json"
    result = json.loads(path.read_text("utf-8"))
    mutate(result)
    path.write_text(json.dumps(result), "utf-8")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(contamination={"status": "none", "pairs": []}),
        lambda r: r.update(schema=1),
        lambda r: r["overall"].update(ap=1.5),
        lambda r: r["overall"].update(n_images=99),
        lambda r: r["per_class"]["分類A"].update(n_images=1),
        lambda r: r.pop("per_class"),
        lambda r: r.update(per_class=[1, 2]),
    ],
)
def test_invalid_result_summary_is_broken_and_not_releasable(env, mutate):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    run_dir = _write_evaluation(service, cid, "eval_001")
    assert validate_evaluation_result(run_dir)
    _rewrite_result(run_dir, mutate)
    assert not validate_evaluation_result(run_dir)
    with pytest.raises(ValueError, match="壊れています"):
        service.release_candidate(cid, "eval_001")
    assert service.get_candidate_evaluation(cid, "val_v000").broken
    assert "val_v000" not in service.get_candidate(cid).evaluations


def _fail_candidate_release_save_once(monkeypatch):
    """公開後の candidate.json（released への更新）の保存を 1 回だけ PermissionError にする。"""
    original = comparison_service._write_json
    state = {"failed": False}

    def flaky(path, value):
        if (
            path.name == "candidate.json"
            and value.get("status") == "released"
            and not state["failed"]
        ):
            state["failed"] = True
            raise PermissionError("locked")
        return original(path, value)

    monkeypatch.setattr(comparison_service, "_write_json", flaky)
    return state


def _published(service):
    return sorted(path.name for path in service.releases_root.glob("model_*"))


def test_release_retry_after_candidate_save_failure_reuses_published(env, monkeypatch):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    state = _fail_candidate_release_save_once(monkeypatch)
    with pytest.raises(ValueError, match="model_001 としてリリースしました"):
        service.release_candidate(cid, "eval_001", "初回")
    assert state["failed"]
    assert _published(service) == ["model_001"]
    assert service.get_candidate(cid).status == "candidate"
    model = service.release_candidate(cid, "eval_001", "再試行")
    assert model.model_id == "model_001" and is_recovered_release(model)
    assert _published(service) == ["model_001"]
    candidate = service.get_candidate(cid)
    assert candidate.status == "released" and candidate.released_model_id == "model_001"
    # 公開済みの記録（コメント）は再試行で書き換えない
    assert service.get_release_record("model_001")["comment"] == "初回"
    assert not list(service.releases_root.glob(".preparing_*"))


def test_release_retry_with_other_evaluation_keeps_published_evaluation(env, monkeypatch):
    """公開後に候補保存が失敗し、再評価した別の評価で再試行しても、公開済みの評価を保持する。"""
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    _fail_candidate_release_save_once(monkeypatch)
    with pytest.raises(ValueError):
        service.release_candidate(cid, "eval_001", "初回")
    _write_evaluation(service, cid, "eval_002")
    published = service.get_release_record("model_001")
    model = service.release_candidate(cid, "eval_002", "再評価後")
    assert is_recovered_release(model)
    assert model.model_id == "model_001" and model.evaluation_id == "eval_001"
    assert _published(service) == ["model_001"]
    # 公開済みの記録は変えない（今回の評価に差し替えない）
    assert service.get_release_record("model_001") == published
    candidate = service.get_candidate(cid)
    assert candidate.status == "released" and candidate.released_model_id == "model_001"


def test_published_release_with_missing_evaluation_is_not_repaired(env, monkeypatch):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    run_dir = _write_evaluation(service, cid, "eval_001")
    _fail_candidate_release_save_once(monkeypatch)
    with pytest.raises(ValueError):
        service.release_candidate(cid, "eval_001")
    # リリースが参照する評価が消えている（参照先の欠落）は自動で直さない
    shutil.rmtree(run_dir)
    _write_evaluation(service, cid, "eval_002")
    with pytest.raises(ValueError, match="食い違い"):
        service.release_candidate(cid, "eval_002")
    fresh = ComparisonService(service.workspace_root, service.training_service)
    assert fresh.recover() == []
    assert fresh.get_candidate(cid).status == "candidate"
    assert _published(service) == ["model_001"]


def test_release_refuses_inconsistent_references_without_changes(env):
    service, training = env
    config = _default_config(service)
    cid = service.add_candidate("exp_0001", 1, config.config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.release_candidate(cid, "eval_001")
    path = service.candidates_root / cid / "candidate.json"
    # 同じ候補のリリースが 2 件ある状態（旧版の二重登録）を作る
    shutil.copytree(service.releases_root / "model_001", service.releases_root / "model_002")
    duplicate = service.releases_root / "model_002" / "release.json"
    value = json.loads(duplicate.read_text("utf-8"))
    value["model_id"] = "model_002"
    duplicate.write_text(json.dumps(value), "utf-8")
    record = json.loads(path.read_text("utf-8"))
    record["status"] = "candidate"
    record.pop("released_model_id")
    path.write_text(json.dumps(record), "utf-8")
    with pytest.raises(ValueError, match="食い違い"):
        service.release_candidate(cid, "eval_001")
    assert _published(service) == ["model_001", "model_002"]
    fresh = ComparisonService(service.workspace_root, service.training_service)
    assert fresh.recover() == []
    assert _published(fresh) == ["model_001", "model_002"]
    assert fresh.get_candidate(cid).status == "candidate"
    # 候補が参照するリリースが見つからない
    shutil.rmtree(service.releases_root / "model_002")
    record["released_model_id"] = "model_009"
    path.write_text(json.dumps(record), "utf-8")
    with pytest.raises(ValueError, match="食い違い"):
        service.release_candidate(cid, "eval_001")
    assert _published(service) == ["model_001"]
    # 別の候補（同じ試行・別の推論設定）のリリースとは混同しない
    training.add_attempt("exp_0002", 1, weights=b"weights-2")
    other = service.add_candidate("exp_0002", 1, config.config_id).candidate_id
    _write_evaluation(service, other, "eval_001")
    assert service.release_candidate(other, "eval_001").model_id == "model_002"


def test_release_failure_before_publish_removes_own_preparing(env, monkeypatch):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")

    def broken_copy(_reader, writer, _length):
        writer.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(comparison_service.shutil, "copyfileobj", broken_copy)
    with pytest.raises(OSError, match="disk full"):
        service.release_candidate(cid, "eval_001")
    assert not list(service.releases_root.iterdir())
    assert service.get_candidate(cid).status == "candidate"
    monkeypatch.undo()
    model = service.release_candidate(cid, "eval_001")
    assert model.model_id == "model_001" and not is_recovered_release(model)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (None, ("matching", "")),
        ({"box_score_thresh": 0.9}, ("different", "推論設定が学習時と異なります")),
    ],
)
def test_released_model_carries_oof_applicability(env, params, expected):
    service, _training = env
    config = service.create_inference_config(
        "mask_rcnn", params or service.default_inference_params("mask_rcnn")
    )
    cid = service.add_candidate("exp_0001", 1, config.config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.release_candidate(cid, "eval_001")
    model = service.list_released_models()[0]
    assert (model.oof_applicability, model.oof_reason) == expected
    assert model.oof_evaluation.overall_map == pytest.approx(0.8)


def test_released_model_without_recorded_applicability_is_not_matching(env):
    service, _training = env
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    _write_evaluation(service, cid, "eval_001")
    service.release_candidate(cid, "eval_001")
    path = service.releases_root / "model_001" / "release.json"
    record = json.loads(path.read_text("utf-8"))
    record["oof"].pop("applicability")
    record["oof"].pop("reason")
    path.write_text(json.dumps(record), "utf-8")
    assert service.list_released_models()[0].oof_applicability == ""
