"""評価の準備・評価プロセス本体（同一プロセスで実行）・終端判定・復旧のテスト。

偽アダプタ fake_numpy（torch を使わない）で、比較・推論設計 7 章の流れを確かめる。
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from foam_cell_analysis.inference.protocol import read_events
from foam_cell_analysis.inference.run import run_job
from foam_cell_analysis.jobs.protocol import read_json
from foam_cell_analysis.services.comparison_service import validate_evaluation_result
from foam_cell_analysis.services.models import JobExit

pytest.importorskip("cellpose")


def _run(prepared, **kwargs) -> tuple[int, list[dict]]:
    stdout = io.StringIO()
    code = run_job(
        prepared.run_dir,
        stdin=io.StringIO("go\n"),
        stdout=stdout,
        start_watchdog=False,
        **kwargs,
    )
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    return code, lines


def _evaluate(env, candidate_id="RC-001"):
    prepared = env.service.prepare_evaluation_run(candidate_id)
    code, lines = _run(prepared)
    return prepared, code, lines


def test_evaluation_completes_and_writes_valid_result(evaluation_env):
    env = evaluation_env
    service = env.service
    prepared, code, lines = _evaluate(env)
    run_dir = Path(prepared.run_dir)

    assert code == 0, (run_dir / "error.json").read_text("utf-8") if code else ""
    assert prepared.run_id == "RC-001/val_v000/eval_001"
    assert prepared.args[:2] == ["-m", "foam_cell_analysis.inference.run"]
    assert Path(prepared.args[-1]).is_absolute()
    assert "TORCH_HOME" in prepared.env and "CELLPOSE_LOCAL_MODELS_PATH" in prepared.env
    types = [line["type"] for line in lines]
    assert types[0] == "hello"
    assert types[1:3] == ["started", "preflight"]
    assert types.count("image_done") == 3 and types[-1] == "completed"
    events = read_events(run_dir / "events.jsonl", schema=2)
    assert [event["type"] for event in events] == types[1:]
    assert validate_evaluation_result(run_dir)

    # 現行形式: 候補に固定した組を記録し、学習混入の項目は run_spec・結果・イベントに出さない
    spec = read_json(run_dir / "run_spec.json")
    assert spec["schema"] == 2
    assert spec["dataset_pair"] == {
        "training_version": "train_v000",
        "validation_version": "val_v000",
    }
    assert "contamination_source" not in spec
    preflight = next(event for event in events if event["type"] == "preflight")
    assert "contamination" not in preflight
    assert "contamination" not in read_json(run_dir / "resolved_data.json")
    result = read_json(run_dir / "result.json")
    assert result["schema"] == 2 and "contamination" not in result
    assert result["evaluation_id"] == "eval_001"
    assert result["validation_version"] == "val_v000"
    assert result["overall"]["n_images"] == 3
    # 完全一致 1.0、過剰検出 1 件で 0.5、空画像同士 1.0 の画像平均
    assert result["overall"]["ap"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert result["per_class"]["分類B"] == {"ap": 0.5, "n_images": 1}
    assert result["per_class"]["未分類"] == {"ap": 1.0, "n_images": 1}
    assert result["device"] == "cpu"
    assert set(result["predictions"]) == {"val_a", "val_b", "val_c"}
    entry = result["predictions"]["val_a"]
    assert entry["path"] == "predictions/val_a.png" and entry["dtype"] == "uint16"
    for key in ("per_image_csv", "eval_counts", "instances_csv"):
        assert len(result[key]["sha256"]) == 64

    counts = np.load(run_dir / "eval_counts.npz")
    assert list(counts["item_ids"]) == ["val_a", "val_b", "val_c"]
    assert counts["tp"].shape == (3, 10) and counts["tp"].dtype == np.int64
    per_image = (run_dir / "per_image.csv").read_text("utf-8").splitlines()
    assert per_image[0] == "item_id,classification,ap,tp,fp,fn,n_true,n_pred"
    assert per_image[2].startswith("val_b,分類B,0.5,1,1,0,1,2")
    instances = (run_dir / "instances.csv").read_text("utf-8").splitlines()
    assert len(instances) == 1 + (2 + 2) + (1 + 2)
    resolved = read_json(run_dir / "resolved_data.json")
    assert resolved["items"]["val_a"]["n_true"] == 2
    environment = read_json(run_dir / "environment.json")
    assert environment["device"] == "cpu" and "torch" in environment["versions"]

    outcome = service.conclude_evaluation_run("RC-001", "eval_001", JobExit(returncode=0))
    assert (outcome.status, outcome.reason) == ("completed", None)
    assert read_json(run_dir / "status.json")["status"] == "completed"
    adopted = service.get_candidate_evaluation("RC-001", "val_v000")
    assert adopted.evaluation_id == "eval_001" and not adopted.broken
    assert service.verify_evaluation("RC-001", "eval_001")
    prediction = service.get_candidate_prediction("RC-001", "eval_001", "val_a")
    truth = np.asarray(Image.open(env.root / "datasets/val_v000/masks/val_a.png"))
    assert np.array_equal(prediction, truth)


def test_progress_follows_events_and_replays_gaps(evaluation_env):
    service = evaluation_env.service
    prepared, code, lines = _evaluate(evaluation_env)
    assert code == 0
    events = [line for line in lines if line["type"] != "hello"]
    service.apply_evaluation_event("RC-001", "eval_001", lines[0])
    service.apply_evaluation_event("RC-001", "eval_001", events[0])
    progress = service.apply_evaluation_event("RC-001", "eval_001", events[1])
    assert (progress.completed, progress.total, progress.phase) == (0, 3, "inference")
    # 途中を飛ばしても events.jsonl から埋める（ファイルにある最後のイベントまで反映する）
    progress = service.apply_evaluation_event("RC-001", "eval_001", events[4])
    assert (progress.completed, progress.total, progress.phase) == (3, 3, "completed")
    # 重複は無視する
    again = service.apply_evaluation_event("RC-001", "eval_001", events[2])
    assert again.completed == 3
    broken = dict(events[5])
    broken.pop("path")
    broken["seq"] = 99
    with pytest.raises(ValueError, match="必須項目"):
        service.apply_evaluation_event("RC-001", "eval_001", broken)
    other = dict(events[5], run_id="RC-002/val_v000/eval_001")
    with pytest.raises(ValueError, match="run_id"):
        service.apply_evaluation_event("RC-001", "eval_001", other)
    service.conclude_evaluation_run("RC-001", "eval_001", JobExit(returncode=0))
    assert service.get_evaluation_progress("RC-001") is None


def test_prepare_errors(evaluation_env):
    env = evaluation_env
    service = env.service
    with pytest.raises(ValueError, match="比較候補がありません"):
        service.prepare_evaluation_run("RC-009")
    env.add_candidate("RC-002", experiment_id="exp_0002", input_channels=("B",))
    with pytest.raises(ValueError, match="モデルが使うチャンネル B がない画像があります"):
        service.prepare_evaluation_run("RC-002")
    # 組の検証用の版に画像がない
    empty = env.root / "datasets" / "val_v001"
    shutil.copytree(env.root / "datasets" / "val_v000", empty)
    for name in ("metadata.csv", "manifest.csv"):
        header = (empty / name).read_text("utf-8").splitlines()[0]
        (empty / name).write_text(header + "\n", encoding="utf-8")
    info = json.loads((empty / "dataset_info.json").read_text("utf-8"))
    (empty / "dataset_info.json").write_text(
        json.dumps({**info, "dataset_version": "val_v001"}), encoding="utf-8"
    )
    train = env.root / "datasets" / "train_v001"
    shutil.copytree(env.root / "datasets" / "train_v000", train)
    info = json.loads((train / "dataset_info.json").read_text("utf-8"))
    (train / "dataset_info.json").write_text(
        json.dumps(
            {**info, "dataset_version": "train_v001", "base_validation_version": "val_v001"}
        ),
        encoding="utf-8",
    )
    env.add_candidate(
        "RC-003",
        experiment_id="exp_0003",
        training_version="train_v001",
        validation_version="val_v001",
    )
    with pytest.raises(ValueError, match="画像がありません"):
        service.prepare_evaluation_run("RC-003")
    record = service.get_candidate_record("RC-001")
    service.reject_candidate("RC-001")
    with pytest.raises(ValueError, match="候補状態"):
        service.prepare_evaluation_run("RC-001")
    assert record["status"] == "candidate"
    # 失敗した準備は評価として数えない
    assert not list(service.candidates_root.glob("*/evaluations/*/eval_*"))


def test_evaluation_uses_only_the_fixed_validation_version(evaluation_env):
    """評価先は候補に固定した版だけ。組の記録と食い違う・組がない候補は評価しない。"""
    env = evaluation_env
    service = env.service
    # 学習用の版の組の記録が候補の固定値と食い違う（黙って別の版へ切り替えない）
    other = env.root / "datasets" / "val_v001"
    shutil.copytree(env.root / "datasets" / "val_v000", other)
    info = json.loads((other / "dataset_info.json").read_text("utf-8"))
    (other / "dataset_info.json").write_text(
        json.dumps({**info, "dataset_version": "val_v001"}), encoding="utf-8"
    )
    train_info = env.root / "datasets" / "train_v000" / "dataset_info.json"
    original = train_info.read_text("utf-8")
    train_info.write_text(
        json.dumps({**json.loads(original), "base_validation_version": "val_v001"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="候補の記録（val_v000）と一致しません"):
        service.prepare_evaluation_run("RC-001")
    # 組の記録がない学習用の版
    train_info.write_text(
        json.dumps(
            {k: v for k, v in json.loads(original).items() if k != "base_validation_version"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="組になる検証用データセットが記録されていません"):
        service.prepare_evaluation_run("RC-001")
    train_info.write_text(original, encoding="utf-8")
    # 組の記録がない旧候補（起動時の補完ができていない）は評価しない
    env.add_candidate("RC-002", experiment_id="exp_0002", validation_version=None)
    with pytest.raises(ValueError, match="組を確認できていない"):
        service.prepare_evaluation_run("RC-002")
    assert not list(service.candidates_root.glob("*/evaluations/*/eval_*"))
    assert service.prepare_evaluation_run("RC-001").run_id == "RC-001/val_v000/eval_001"


def _tamper_image(env, run_dir):
    Image.fromarray(np.full((24, 24), 7, np.uint8)).save(
        env.root / "datasets/val_v000/images/val_a.png"
    )


def _tamper_manifest(env, run_dir):
    path = env.root / "datasets/val_v000/metadata.csv"
    path.write_text(path.read_text("utf-8") + "\n", encoding="utf-8")


def _tamper_weights(env, run_dir):
    (env.root / "experiments/exp_0001/runs/attempt_001/checkpoints/final.pt").write_bytes(
        b"changed-body"
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (_tamper_image, "sha256 が manifest と一致しません"),
        (_tamper_manifest, "metadata.csv が評価の準備時と一致しません"),
        (_tamper_weights, "モデルファイル"),
    ],
)
def test_preflight_failures_make_evaluation_failed(evaluation_env, tamper, message):
    env = evaluation_env
    prepared = env.service.prepare_evaluation_run("RC-001")
    tamper(env, Path(prepared.run_dir))
    code, lines = _run(prepared)
    run_dir = Path(prepared.run_dir)

    assert code == 1
    error = read_json(run_dir / "error.json")
    assert message in error["message"] and error["phase"] == "preflight"
    assert lines[-1]["type"] == "error"
    assert not (run_dir / "result.json").exists()
    outcome = env.service.conclude_evaluation_run("RC-001", "eval_001", JobExit(returncode=1))
    assert (outcome.status, outcome.reason) == ("failed", "error")
    assert message in outcome.message


def test_preflight_rejects_mask_with_other_shape(evaluation_env):
    env = evaluation_env
    folder = env.root / "datasets/val_v000"
    Image.fromarray(np.zeros((12, 12), np.uint16)).save(folder / "masks/val_c.png")
    # manifest の sha256 は正しく書き直し、形の検査だけで落ちることを確かめる
    import hashlib

    rows = (folder / "manifest.csv").read_text("utf-8").splitlines()
    digest = hashlib.sha256((folder / "masks/val_c.png").read_bytes()).hexdigest()
    rows = [
        ",".join(row.split(",")[:4] + [digest]) if row.startswith("val_c,") else row for row in rows
    ]
    (folder / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    prepared = env.service.prepare_evaluation_run("RC-001")
    code, _lines = _run(prepared)
    assert code == 1
    assert "大きさが一致しません" in read_json(Path(prepared.run_dir) / "error.json")["message"]


def test_capacity_shortage_fails(evaluation_env):
    prepared = evaluation_env.service.prepare_evaluation_run("RC-001")
    code, _lines = _run(prepared, free_bytes_fn=lambda _path: 10)
    assert code == 1
    assert "空き容量" in read_json(Path(prepared.run_dir) / "error.json")["message"]


def _legacy_spec(spec: dict) -> dict:
    """現行の run_spec を旧形式（schema 1、学習混入の項目あり）へ戻す。"""
    legacy = {key: value for key, value in spec.items() if key != "dataset_pair"}
    legacy["schema"] = 1
    legacy["contamination_source"] = {
        "run_id": "exp_0001/attempt_001",
        "dataset_version": "train_v000",
        "dataset_sha256": None,
        "used_item_ids": ["train_1"],
    }
    return legacy


def test_legacy_and_current_record_formats_are_told_apart(evaluation_env):
    """旧形式（学習混入の項目あり）と現行形式（なし）を run_spec の schema で見分ける。

    どちらの形式も読めるが、形式に合わない項目の混ざった記録は壊れた記録として扱う。
    """
    from foam_cell_analysis.inference.protocol import validate_event, validate_run_spec

    env = evaluation_env
    prepared, code, lines = _evaluate(env)
    assert code == 0
    run_dir = Path(prepared.run_dir)
    spec = read_json(run_dir / "run_spec.json")
    result = read_json(run_dir / "result.json")
    preflight = next(line for line in lines if line["type"] == "preflight")
    contamination = {"status": "found", "pairs": [], "reason": None}

    # 現行形式に学習混入の項目が混ざったもの
    with pytest.raises(ValueError, match="形式に合わない"):
        validate_run_spec({**spec, "contamination_source": {}}, run_dir)
    with pytest.raises(ValueError, match="形式に合わない"):
        validate_event({**preflight, "contamination": contamination}, schema=2)
    (run_dir / "result.json").write_text(
        json.dumps({**result, "contamination": contamination}), encoding="utf-8"
    )
    assert not validate_evaluation_result(run_dir)

    # 旧形式: 学習混入の項目が必須で、揃っていれば読める
    legacy = _legacy_spec(spec)
    validate_run_spec(legacy, run_dir)
    with pytest.raises(ValueError, match="必須項目"):
        validate_event(preflight, schema=1)
    validate_event({**preflight, "contamination": contamination}, schema=1)
    (run_dir / "run_spec.json").write_text(json.dumps(legacy), encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps({**result, "schema": 1, "contamination": contamination}), encoding="utf-8"
    )
    assert validate_evaluation_result(run_dir)
    (run_dir / "result.json").write_text(json.dumps({**result, "schema": 1}), encoding="utf-8")
    assert not validate_evaluation_result(run_dir)


def test_legacy_events_replay_and_legacy_spec_is_not_executed(evaluation_env):
    """旧形式の評価の events.jsonl（preflight に contamination）を復旧・再生で読める。

    旧形式の run_spec は評価プロセスでは実行しない（新しく評価をやり直す）。
    """
    env = evaluation_env
    service = env.service
    prepared, code, lines = _evaluate(env)
    assert code == 0
    run_dir = Path(prepared.run_dir)
    spec = read_json(run_dir / "run_spec.json")
    (run_dir / "run_spec.json").write_text(json.dumps(_legacy_spec(spec)), encoding="utf-8")
    events = [line for line in lines if line["type"] != "hello"]
    legacy_events = [
        {**event, "contamination": {"status": "none", "pairs": [], "reason": None}}
        if event["type"] == "preflight"
        else event
        for event in events
    ]
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in legacy_events),
        encoding="utf-8",
    )
    assert [event["type"] for event in read_events(run_dir / "events.jsonl", schema=1)] == [
        event["type"] for event in legacy_events
    ]
    # 途中のイベントから始めても、旧形式の events.jsonl から埋める
    progress = service.apply_evaluation_event("RC-001", "eval_001", legacy_events[4])
    assert (progress.completed, progress.total) == (3, 3)
    with pytest.raises(ValueError, match="形式に合わない|必須項目"):
        service.apply_evaluation_event(
            "RC-001", "eval_001", {**events[1], "seq": legacy_events[-1]["seq"] + 1}
        )

    second = service.prepare_evaluation_run("RC-001")
    second_dir = Path(second.run_dir)
    legacy = _legacy_spec(read_json(second_dir / "run_spec.json"))
    (second_dir / "run_spec.json").write_text(json.dumps(legacy), encoding="utf-8")
    code, _lines = _run(second)
    assert code == 1
    assert "旧形式" in read_json(second_dir / "error.json")["message"]


def test_invalid_run_spec_exits_with_code_two_without_hello(evaluation_env):
    prepared = evaluation_env.service.prepare_evaluation_run("RC-001")
    spec_path = Path(prepared.run_dir) / "run_spec.json"
    spec = read_json(spec_path)
    spec["validation"]["item_ids"] = ["../escape"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    stdout = io.StringIO()
    code = run_job(prepared.run_dir, stdin=io.StringIO("go\n"), stdout=stdout, start_watchdog=False)
    assert code == 2 and stdout.getvalue() == ""


def test_eof_before_go_exits_with_code_three(evaluation_env):
    prepared = evaluation_env.service.prepare_evaluation_run("RC-001")
    stdout = io.StringIO()
    code = run_job(prepared.run_dir, stdin=io.StringIO(""), stdout=stdout, start_watchdog=False)
    assert code == 3
    assert json.loads(stdout.getvalue())["type"] == "hello"
    assert not (Path(prepared.run_dir) / "events.jsonl").exists()


def test_conclusion_priorities(evaluation_env):
    env = evaluation_env
    service = env.service
    # 停止要求と完了の競合: result.json が妥当でも停止要求が優先する
    prepared, code, _lines = _evaluate(env)
    assert code == 0
    service.request_evaluation_stop("RC-001", "eval_001", "user_stop")
    outcome = service.conclude_evaluation_run("RC-001", "eval_001", JobExit(returncode=0))
    assert (outcome.status, outcome.reason) == ("stopped", "user_stop")
    # 保存済みの状態は書き換えない
    again = service.conclude_evaluation_run("RC-001", "eval_001", JobExit(returncode=1))
    assert (again.status, again.reason) == ("stopped", "user_stop")
    assert read_json(Path(prepared.run_dir) / "status.json")["returncode"] == 0
    with pytest.raises(ValueError, match="中断理由"):
        service.request_evaluation_stop("RC-001", "eval_001", "other")

    # 起動失敗
    service.prepare_evaluation_run("RC-001")
    failed = service.conclude_evaluation_run(
        "RC-001", "eval_002", JobExit(start_failed=True, message="起動できません")
    )
    assert (failed.status, failed.reason, failed.message) == (
        "failed",
        "start_failed",
        "起動できません",
    )
    # プロトコルエラー（error.json なし）は failed
    service.prepare_evaluation_run("RC-001")
    protocol = service.conclude_evaluation_run(
        "RC-001", "eval_003", JobExit(returncode=-1, protocol_error=True, message="欠けたイベント")
    )
    assert (protocol.status, protocol.reason, protocol.message) == (
        "failed",
        "error",
        "欠けたイベント",
    )
    # プロセスが生きている間は確定しない
    run_dir = Path(service.prepare_evaluation_run("RC-001").run_dir)
    service.record_evaluation_process("RC-001", "eval_004", 4242, 1000.0)
    env.alive[4242] = True
    assert service.conclude_evaluation_run("RC-001", "eval_004").status == "running"
    assert not (run_dir / "status.json").exists()
    env.alive[4242] = False
    stopped = service.conclude_evaluation_run("RC-001", "eval_004")
    assert (stopped.status, stopped.reason) == ("stopped", "interrupted")


def test_reevaluation_keeps_previous_and_numbers_per_candidate(evaluation_env):
    env = evaluation_env
    service = env.service
    first, code, _lines = _evaluate(env)
    assert code == 0
    service.conclude_evaluation_run("RC-001", "eval_001", JobExit(returncode=0))
    before = {
        path.name: path.read_bytes() for path in Path(first.run_dir).rglob("*") if path.is_file()
    }
    # 別の検証用の版で行った旧評価（履歴）があっても、番号は候補ごとの通し番号
    history = service.candidates_root / "RC-001" / "evaluations" / "val_v001" / "eval_002"
    history.mkdir(parents=True)
    (history / "run_spec.json").write_text("{}", encoding="utf-8")
    third = service.prepare_evaluation_run("RC-001")

    assert third.run_id == "RC-001/val_v000/eval_003"
    after = {
        path.name: path.read_bytes() for path in Path(first.run_dir).rglob("*") if path.is_file()
    }
    assert after == before
    # 新しい評価が失敗しても、前回の成功した評価が採用される
    service.conclude_evaluation_run("RC-001", "eval_003", JobExit(start_failed=True, message="x"))
    assert service.get_candidate_evaluation("RC-001").evaluation_id == "eval_001"
    history_ids = [item.evaluation_id for item in service.list_candidate_evaluations("RC-001")]
    assert history_ids == ["eval_001", "eval_002", "eval_003"]
    # 別の候補は 1 から数える
    env.add_candidate("RC-002", experiment_id="exp_0002")
    assert service.prepare_evaluation_run("RC-002").run_id.endswith("eval_001")


def test_recovery_concludes_crashed_evaluations_and_cleans_temporaries(evaluation_env):
    env = evaluation_env
    service = env.service
    # 1. プロセスが落ち、status.json がない → stopped/interrupted
    crashed = Path(service.prepare_evaluation_run("RC-001").run_dir)
    service.record_evaluation_process("RC-001", "eval_001", 1111, 1000.0)
    (crashed / "predictions").mkdir()
    (crashed / "predictions" / "val_a.png.tmp").write_bytes(b"partial")
    # 2. result.json を書いた直後に落ちた → completed
    done, code, _lines = _evaluate(env)
    assert code == 0
    # 3. まだ生きているプロセス → 終了させてから確定する
    alive = Path(service.prepare_evaluation_run("RC-001").run_dir)
    service.record_evaluation_process("RC-001", "eval_003", 3333, 1000.0)
    env.alive[3333] = True
    preparing = crashed.parent / ".preparing_leftover"
    preparing.mkdir()

    outcomes = {item.evaluation_id: item for item in service.recover_evaluations()}

    assert (outcomes["eval_001"].status, outcomes["eval_001"].reason) == ("stopped", "interrupted")
    assert outcomes["eval_002"].status == "completed"
    assert (outcomes["eval_003"].status, outcomes["eval_003"].reason) == (
        "stopped",
        "interrupted",
    )
    assert env.terminated == [{"pid": 3333, "creation_time": 1000.0}]
    assert read_json(crashed / "status.json")["status"] == "stopped"
    assert read_json(alive / "status.json")["status"] == "stopped"
    assert not (crashed / "predictions" / "val_a.png.tmp").exists()
    assert not preparing.exists()
    assert Path(done.run_dir, "result.json").is_file()
    # 2 回目の復旧では何もしない
    assert service.recover_evaluations() == []


def test_recovery_blocks_when_live_process_cannot_be_terminated(evaluation_env, monkeypatch):
    from foam_cell_analysis.jobs import lifecycle

    env = evaluation_env
    service = env.service
    stuck = Path(service.prepare_evaluation_run("RC-001").run_dir)
    service.record_evaluation_process("RC-001", "eval_001", 4444, 1000.0)
    env.alive[4444] = True
    (stuck / "predictions").mkdir()
    partial = stuck / "predictions" / "val_a.png.tmp"
    partial.write_bytes(b"partial")
    preparing = stuck.parent / ".preparing_leftover"
    preparing.mkdir()
    (preparing / "x.tmp").write_bytes(b"x")
    service.process_terminator = lambda process: None  # 終了できない
    clock = {"now": 0.0}
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        lifecycle.time, "sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds)
    )

    def snapshot_files(root):
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }

    before = snapshot_files(service.candidates_root)

    assert service.recover_evaluations() == []

    assert snapshot_files(service.candidates_root) == before
    assert not (stuck / "status.json").exists()
    assert partial.read_bytes() == b"partial"
    assert (preparing / "x.tmp").read_bytes() == b"x"
    assert len(service.recovery_blockers) == 1
    assert "eval_001" in service.recovery_blockers[0]
    assert service.recovery_issues["RC-001"][0] == "unconfirmed"


def test_evaluation_fails_when_image_changes_after_preflight(evaluation_env):
    from foam_cell_analysis.inference.adapters import build_inference_adapter

    env = evaluation_env
    prepared = env.service.prepare_evaluation_run("RC-001")

    def factory(**kwargs):
        adapter = build_inference_adapter(**kwargs)
        _tamper_image(env, None)  # preflight 通過後に画像を差し替える
        return adapter

    code, _lines = _run(prepared, adapter_factory=factory)

    assert code == 1
    message = read_json(Path(prepared.run_dir) / "error.json")["message"]
    assert "変更されました" in message
    assert not (Path(prepared.run_dir) / "result.json").exists()


def test_prepare_failure_removes_preparing_folder(evaluation_env, monkeypatch):
    service = evaluation_env.service

    def fail_replace(_source, _target):
        raise OSError("path too long")

    monkeypatch.setattr("foam_cell_analysis.services.comparison_service.os.replace", fail_replace)
    with pytest.raises(OSError):
        service.prepare_evaluation_run("RC-001")
    assert not list((service.candidates_root / "RC-001").glob("evaluations/*/.preparing_*"))
