"""ComparisonService の保存・復元・評価の読み取り・リリース・振り分けのテスト。"""

import copy
import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from foam_cell_analysis.data.dataset_store import DatasetStore
from foam_cell_analysis.services.comparison_service import (
    ComparisonService,
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


def _dataset(root, version, purpose, classifications, base_validation_version=None):
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
            writer.writerow([f"{version}_{index}", "i.png", f"{index:064x}", "m.png"])


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
    contamination="none",
    input_fingerprint=None,
):
    run_dir = service.candidates_root / candidate_id / "evaluations" / version / evaluation_id
    (run_dir / "predictions").mkdir(parents=True)
    item_ids = [item.item_id for item in service.dataset_store.select_evaluation_items(version)]
    fingerprint = input_fingerprint or service.evaluation_input_fingerprint(candidate_id, version)
    run_id = f"{candidate_id}/{version}/{evaluation_id}"
    spec = {
        "schema": 1,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "evaluation_id": evaluation_id,
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
        "schema": 1,
        "run_id": run_id,
        "evaluation_id": evaluation_id,
        "validation_version": version,
        "input_fingerprint": fingerprint,
        "overall": {"ap": 0.6, "n_images": len(item_ids)},
        "per_class": {"分類A": {"ap": 0.7, "n_images": 1}, "分類C": {"ap": None, "n_images": 0}},
        "metric": {"id": "cellpose_ap_iou50_95_image_mean_v1"},
        "contamination": {"status": contamination, "pairs": []},
        "predictions": predictions,
        **artifacts,
        "completed_at": "2026-09-29T10:00:00+09:00",
    }
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
    service.save_external_results(
        cid, "eval_001", [{"name": "平均粒子径", "value": 12.3, "unit": "µm"}], software="X"
    )
    model = service.release_candidate(cid, "eval_001", "初回")
    assert model.model_id == "model_001"
    assert model.validation_dataset == "val_v000"
    assert model.evaluation_result.overall_map == pytest.approx(0.6)
    released = service.releases_root / "model_001"
    release = json.loads((released / "release.json").read_text("utf-8"))
    assert release["weights"]["sha256"] == _sha(b"weights-1")
    assert release["evaluation"]["base_validation_version"] == "val_v000"
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
        service.save_external_results(cid, "eval_001", [])
    with pytest.raises(ValueError, match="候補状態"):
        service.release_candidate(cid, "eval_001")


def test_release_blocked_conditions(env):
    service, training = env
    config = _default_config(service)
    cid = service.add_candidate("exp_0001", 1, config.config_id).candidate_id
    _write_evaluation(service, cid, "eval_001", contamination="found")
    with pytest.raises(ValueError, match="同じ画像"):
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


def test_validation_version_defaults(env):
    service, _training = env
    assert service.default_validation_version() == "val_v000"
    _dataset(service.workspace_root, "val_v001", "val", ["分類A"])
    assert service.default_validation_version() == "val_v001"
    cid = service.add_candidate("exp_0001", 1, _default_config(service).config_id).candidate_id
    assert service.base_validation_version_for(cid) == "val_v000"
    assert service.default_validation_version() == "val_v000"
