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


def test_settle_process_rejects_broken_record_without_calling_terminator(tmp_path, monkeypatch):
    from foam_cell_analysis.jobs import lifecycle

    monkeypatch.setattr(lifecycle.os, "name", "posix")
    monkeypatch.setattr(lifecycle, "Path", lambda value: value)
    process_file = tmp_path / "process.json"
    process_file.write_text("{broken", encoding="utf-8")
    terminated = []
    assert (
        lifecycle.settle_process(
            process_file, lambda _record: True, lambda record: terminated.append(record)
        )
        == "unconfirmed"
    )
    assert terminated == []
    assert process_file.read_text(encoding="utf-8") == "{broken"


def test_settle_process_distinguishes_missing_and_confirmed_dead(tmp_path, monkeypatch):
    import json

    from foam_cell_analysis.jobs import lifecycle

    monkeypatch.setattr(lifecycle.os, "name", "posix")
    monkeypatch.setattr(lifecycle, "Path", lambda value: value)
    path = tmp_path / "process.json"
    assert lifecycle.settle_process(path, lambda _record: False, lambda _record: None) == "missing"
    path.write_text(json.dumps({"pid": 123, "creation_time": 1.0}), encoding="utf-8")
    assert lifecycle.settle_process(path, lambda _record: False, lambda _record: None) == "dead"


def test_process_identity_rejects_bool_nonpositive_and_nonfinite_creation_times():
    from foam_cell_analysis.jobs.lifecycle import process_created_at

    for value in (True, False, 0, -1, float("nan"), float("inf"), float("-inf")):
        assert process_created_at({"pid": 7, "creation_time": value}) is None


def test_unrecorded_worker_lookup_does_not_match_substring_arguments(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys
    from types import SimpleNamespace

    from foam_cell_analysis.jobs import lifecycle

    run_dir = tmp_path / "attempt_001"
    other_run_dir = tmp_path / "attempt_0010"
    row = {
        "ProcessId": 9001,
        "Name": "python.exe",
        "ExecutablePath": sys.executable,
        "CreationDate": "20261005120000.000000+000",
        "CommandLine": subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "foam_cell_analysis.training.run_extra",
                "--run-dir",
                str(other_run_dir),
            ]
        ),
    }
    monkeypatch.setattr(lifecycle.os, "name", "nt")
    monkeypatch.setattr(
        lifecycle.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(row), stderr=""),
    )
    monkeypatch.setattr(
        lifecycle,
        "windows_process_handle",
        lambda *_args, **_kwargs: pytest.fail("部分一致の候補を終了対象にしました"),
    )
    assert (
        lifecycle.settle_unrecorded_worker(
            run_dir,
            "foam_cell_analysis.training.run",
            lifecycle.process_alive,
            lifecycle.terminate_process,
        )
        == "dead"
    )


def test_unrecorded_worker_lookup_rejects_module_text_after_python_c(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from foam_cell_analysis.jobs import lifecycle

    run_dir = tmp_path / "attempt_001"
    row = {
        "ProcessId": 9002,
        "Name": "python.exe",
        "ExecutablePath": sys.executable,
        "CreationFileTime": 134356832261999410,
        "CommandLine": "python.exe -c 'sleep' -m foam_cell_analysis.training.run "
        f"--run-dir {run_dir}",
    }
    monkeypatch.setattr(lifecycle.os, "name", "nt")
    monkeypatch.setattr(
        lifecycle.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(row), stderr=""),
    )
    monkeypatch.setattr(
        lifecycle,
        "_windows_command_line_args",
        lambda _line: [
            sys.executable,
            "-c",
            "sleep",
            "-m",
            "foam_cell_analysis.training.run",
            "--run-dir",
            str(run_dir),
        ],
    )
    monkeypatch.setattr(
        lifecycle,
        "windows_process_handle",
        lambda *_args, **_kwargs: pytest.fail("-c の引数をworkerと誤認しました"),
    )

    assert (
        lifecycle.settle_unrecorded_worker(
            run_dir,
            "foam_cell_analysis.training.run",
            lifecycle.process_alive,
            lifecycle.terminate_process,
        )
        == "dead"
    )


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


@pytest.mark.skipif(os.name != "nt", reason="Windows のプロセス API を使う")
def test_process_identity_drift_never_terminates_live_unrelated_process():
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
        mismatched = process_record(child.pid, created + 0.5)
        assert not process_alive(mismatched)
        assert terminate_process(mismatched) is False
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()


@pytest.mark.skipif(os.name != "nt", reason="Windows のプロセス API を使う")
def test_terminate_process_waits_for_exact_child_process_tree(tmp_path):
    import subprocess
    import sys

    from foam_cell_analysis.jobs.lifecycle import (
        process_alive,
        process_creation_time,
        process_record,
        terminate_process,
    )

    child_pid_file = tmp_path / "child.pid"
    child_code = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
        f"open({str(child_pid_file)!r},'w').write(str(child.pid)); "
        "time.sleep(30)"
    )
    root = subprocess.Popen([sys.executable, "-c", child_code])
    try:
        deadline = time.monotonic() + 3
        while not child_pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid_file.exists()
        child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        root_created = process_creation_time(root.pid)
        child_created = process_creation_time(child_pid)
        assert root_created is not None and child_created is not None
        child_record = process_record(child_pid, child_created)
        assert terminate_process(process_record(root.pid, root_created))
        assert root.wait(timeout=5) is not None
        deadline = time.monotonic() + 5
        while process_alive(child_record) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not process_alive(child_record)
    finally:
        root.kill()
        root.wait()
