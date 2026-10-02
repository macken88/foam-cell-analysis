"""学習・評価で共通の終端判定（decide_terminal_state）。"""

import os
import time

import pytest

from foam_cell_analysis.jobs.lifecycle import TerminalDecision, decide_terminal_state

BASE = dict(
    existing_status=None,
    stop_request=None,
    result_valid=False,
    error_present=False,
    start_failed=False,
    process_alive=False,
)


def _decide(**overrides):
    return decide_terminal_state(**{**BASE, **overrides})


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        # 1. 保存済みの状態が最優先（他の条件がそろっていても書き換えない）
        (
            dict(
                existing_status={"status": "failed", "reason": "error", "message": "既存"},
                stop_request={"reason": "user_stop"},
                result_valid=True,
                error_present=True,
                start_failed=True,
                process_alive=True,
            ),
            TerminalDecision("failed", "error", "existing", "既存"),
        ),
        # 2. 停止要求は妥当な結果より優先
        (
            dict(stop_request={"reason": "user_stop"}, result_valid=True, error_present=True),
            TerminalDecision("stopped", "user_stop", "stop_request"),
        ),
        # 3. 妥当な結果はエラーより優先
        (
            dict(result_valid=True, error_present=True, start_failed=True),
            TerminalDecision("completed", None, "result"),
        ),
        # 4. 起動失敗（error.json があっても理由は start_failed）
        (
            dict(error_present=True, start_failed=True, process_alive=True),
            TerminalDecision("failed", "start_failed", "start_failed"),
        ),
        # 4'. エラー（プロセスが生きていても確定する）
        (
            dict(error_present=True, process_alive=True),
            TerminalDecision("failed", "error", "error"),
        ),
        # 5. プロセスが終わっていれば中断
        (dict(), TerminalDecision("stopped", "interrupted", "process_dead")),
        # 6. プロセスが生きていればまだ確定しない
        (dict(process_alive=True), None),
    ],
)
def test_decide_terminal_state_priority(overrides, expected):
    assert _decide(**overrides) == expected


def test_decide_terminal_state_evaluates_callables_only_when_needed():
    calls = []

    def result_valid():
        calls.append("result")
        return False

    def alive():
        calls.append("alive")
        return True

    decision = _decide(stop_request={"reason": "app_exit"}, result_valid=result_valid)
    assert decision.status == "stopped" and calls == []

    assert _decide(result_valid=result_valid, process_alive=alive) is None
    assert calls == ["result", "alive"]


@pytest.mark.skipif(os.name != "nt", reason="Windows のプロセス API を使う")
def test_terminate_process_kills_real_process():
    import subprocess
    import sys

    from foam_cell_analysis.jobs.lifecycle import (
        process_alive,
        process_creation_time,
        process_record,
        terminate_process,
    )

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        created = None
        for _ in range(50):
            created = process_creation_time(child.pid)
            if created is not None:
                break
            time.sleep(0.05)
        assert created is not None
        record = process_record(child.pid, created)
        assert process_alive(record)
        assert terminate_process(record) is True
        assert child.wait(timeout=5) is not None
        assert not process_alive(record)
    finally:
        child.kill()
        child.wait()
