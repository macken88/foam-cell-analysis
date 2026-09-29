"""EvaluationRunner（比較・推論設計 15.2）と評価プロセスの起動、学習との排他のテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from foam_cell_analysis.gui.compute_coordinator import ComputeCoordinator
from foam_cell_analysis.gui.evaluation_runner import EvaluationRunner
from foam_cell_analysis.jobs.protocol import read_json
from foam_cell_analysis.services.models import EvaluationOutcome, PreparedRun


def test_real_process_evaluation_completes(qtbot, evaluation_env):
    """偽アダプタで 準備 → 起動 → preflight → image_done → result.json → completed。"""
    pytest.importorskip("cellpose")
    service = evaluation_env.service
    compute = ComputeCoordinator()
    runner = EvaluationRunner(service, compute=compute)
    progressed = []
    runner.progressed.connect(progressed.append)

    with qtbot.waitSignal(runner.ended, timeout=30_000) as signal:
        runner.start(["RC-001"], "val_v000")
        assert runner.is_evaluation_active("RC-001")

    outcome = signal.args[0]
    assert (outcome.status, outcome.evaluation_id) == ("completed", "eval_001"), outcome
    assert progressed and set(progressed) == {"RC-001"}
    run_dir = service.candidates_root / "RC-001" / "evaluations" / "val_v000" / "eval_001"
    assert read_json(run_dir / "status.json")["status"] == "completed"
    assert read_json(run_dir / "process.json")["pid"] > 0
    assert (run_dir / "stdout.log").is_file()
    assert not runner.is_busy and not compute.is_busy
    assert not runner.is_evaluation_active("RC-001")
    assert service.get_candidate_evaluation("RC-001", "val_v000").evaluation_id == "eval_001"


class StubBackend:
    """フェイクの PreparedRun を返し、呼び出しを記録する Backend の代役。"""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.calls: list[tuple] = []
        self.numbers: dict[str, int] = {}
        self.fail_conclude = False
        self.fail_prepare: set[str] = set()

    def prepare_evaluation_run(self, candidate_id, version):
        self.calls.append(("prepare", candidate_id))
        if candidate_id in self.fail_prepare:
            raise ValueError("準備できません")
        number = self.numbers.get(candidate_id, 0) + 1
        self.numbers[candidate_id] = number
        run_dir = self.tmp_path / candidate_id / f"eval_{number:03d}"
        run_dir.mkdir(parents=True)
        return PreparedRun(
            f"{candidate_id}/{version}/eval_{number:03d}", str(run_dir), "", [], {}, fake=True
        )

    def apply_evaluation_event(self, candidate_id, evaluation_id, event):
        self.calls.append(("event", candidate_id, event["type"]))

    def request_evaluation_stop(self, candidate_id, evaluation_id, reason):
        self.calls.append(("stop", candidate_id, reason))

    def conclude_evaluation_run(self, candidate_id, evaluation_id, job_exit):
        self.calls.append(("conclude", candidate_id))
        if self.fail_conclude:
            raise OSError("status.json を保存できません")
        status = "completed" if job_exit.returncode == 0 else "stopped"
        return EvaluationOutcome(candidate_id, evaluation_id, status)


@pytest.fixture
def stub(qapp, tmp_path):
    backend = StubBackend(tmp_path)
    compute = ComputeCoordinator()
    runner = EvaluationRunner(backend, compute=compute)
    outcomes: list[EvaluationOutcome] = []
    runner.ended.connect(outcomes.append)
    return backend, compute, runner, outcomes


def test_evaluation_waits_while_training_holds_compute(qtbot, stub):
    backend, compute, runner, outcomes = stub
    training = compute.request("training", "学習 exp_0001", lambda: None)

    runner.start(["RC-001"], "val_v000")

    assert runner.waiting_for_compute and runner.is_busy
    assert runner.is_evaluation_active("RC-001")
    assert backend.calls == []
    assert compute.wait_message("evaluation") == "学習の終了を待っています"

    compute.release(training)
    qtbot.waitUntil(lambda: bool(outcomes), timeout=5000)
    assert outcomes[0].status == "completed"
    assert backend.calls[0] == ("prepare", "RC-001")
    assert ("event", "RC-001", "image_done") in backend.calls
    assert not compute.is_busy and not runner.is_busy


def test_each_candidate_takes_its_own_ticket_and_training_cuts_in(qtbot, stub):
    backend, compute, runner, outcomes = stub
    started = []
    runner.start(["RC-001", "RC-002"], "val_v000")
    assert compute.active_owner == "evaluation"
    assert runner.is_evaluation_active("RC-002") and runner.queued_candidate_ids == ["RC-002"]
    # 1 件目の評価中に来た学習は、2 件目より先に入る
    training = compute.request("training", "学習 exp_0001", lambda: started.append("training"))
    qtbot.waitUntil(lambda: len(outcomes) == 1, timeout=5000)
    assert started == ["training"]
    assert runner.waiting_for_compute and runner.candidate_id == "RC-002"
    compute.release(training)
    qtbot.waitUntil(lambda: len(outcomes) == 2, timeout=5000)
    assert [item.candidate_id for item in outcomes] == ["RC-001", "RC-002"]
    assert [call for call in backend.calls if call[0] == "prepare"] == [
        ("prepare", "RC-001"),
        ("prepare", "RC-002"),
    ]


def test_prepare_failure_moves_to_next_candidate(qtbot, stub):
    backend, compute, runner, outcomes = stub
    backend.fail_prepare = {"RC-001"}
    runner.start(["RC-001", "RC-002"], "val_v000")
    qtbot.waitUntil(lambda: len(outcomes) == 2, timeout=5000)
    assert (outcomes[0].status, outcomes[0].reason) == ("failed", "prepare_failed")
    assert outcomes[0].evaluation_id is None
    assert outcomes[1].status == "completed"
    assert not compute.is_busy


def test_failed_terminal_save_blocks_next_until_evaluation_start(qtbot, stub):
    backend, compute, runner, outcomes = stub
    backend.fail_conclude = True
    runner.start(["RC-001", "RC-002"], "val_v000")
    qtbot.waitUntil(lambda: len(outcomes) == 2, timeout=5000)

    assert (outcomes[0].status, outcomes[0].reason) == ("failed", "conclusion_failed")
    assert (outcomes[1].candidate_id, outcomes[1].reason) == ("RC-002", "cancelled")
    assert compute.is_blocked
    started = []
    compute.request("training", "学習 exp_0001", lambda: started.append("training"))
    assert started == []

    # 評価側の解除は評価の開始操作で行う（先に待っていた学習から始まる）
    backend.fail_conclude = False
    runner.start(["RC-003"], "val_v000")
    assert not compute.is_blocked and started == ["training"]
    assert runner.waiting_for_compute


def test_request_stop_writes_stop_before_kill_and_cancels_waiting(qtbot, stub):
    backend, compute, runner, outcomes = stub
    runner.start(["RC-001", "RC-002"], "val_v000")
    assert runner.job is not None

    assert runner.request_stop("user_stop", timeout_ms=5000)

    assert ("stop", "RC-001", "user_stop") in backend.calls
    stop_index = backend.calls.index(("stop", "RC-001", "user_stop"))
    assert backend.calls.index(("conclude", "RC-001")) > stop_index
    assert {(item.candidate_id, item.status) for item in outcomes} == {
        ("RC-002", "stopped"),
        ("RC-001", "stopped"),
    }
    assert not runner.is_busy and not compute.is_busy
    assert ("prepare", "RC-002") not in backend.calls


def test_request_stop_while_waiting_cancels_ticket(qtbot, stub):
    backend, compute, runner, outcomes = stub
    training = compute.request("training", "学習 exp_0001", lambda: None)
    runner.start(["RC-001"], "val_v000")
    assert runner.request_stop()
    assert outcomes[0].reason == "cancelled"
    assert compute.waiting == []
    compute.release(training)
    assert backend.calls == [] and not compute.is_busy


def test_shutdown_requests_app_exit(qtbot, stub):
    backend, compute, runner, outcomes = stub
    runner.start(["RC-001"], "val_v000")
    assert runner.shutdown()
    assert ("stop", "RC-001", "app_exit") in backend.calls
    assert outcomes and not runner.is_busy


def test_app_context_shares_compute_with_evaluation_runner(shell):
    runner = shell.ctx.evaluation_runner
    assert isinstance(runner, EvaluationRunner)
    assert runner.compute is shell.ctx.compute is shell.ctx.training_runner.compute
