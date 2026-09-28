import sys
from pathlib import Path

from foam_cell_analysis.gui.training_jobs import ProcessTrainingJob
from foam_cell_analysis.services.models import PreparedRun


class ProcessRecorder:
    def __init__(self):
        self.recorded = None

    def record_training_process(self, experiment_id, attempt, pid, creation_time):
        self.recorded = (experiment_id, attempt, pid, creation_time)


def _prepared(tmp_path: Path, code: str) -> PreparedRun:
    run_dir = tmp_path / "experiments" / "exp_test" / "runs" / "attempt_001"
    run_dir.mkdir(parents=True)
    return PreparedRun(
        "exp_test/attempt_001",
        str(run_dir),
        sys.executable,
        ["-u", "-c", code],
        {},
    )


def test_process_job_waits_for_go_records_pid_then_forwards_events(qtbot, tmp_path):
    code = (
        "import json, os, sys, time\n"
        "print(json.dumps({'v':1,'run_id':'exp_test/attempt_001','time':time.time(),"
        "'type':'hello','pid':os.getpid()}), flush=True)\n"
        "assert sys.stdin.readline().strip() == 'go'\n"
        "print(json.dumps({'v':1,'run_id':'exp_test/attempt_001','time':time.time(),"
        "'seq':1,'type':'started'}), flush=True)\n"
    )
    backend = ProcessRecorder()
    prepared = _prepared(tmp_path, code)
    job = ProcessTrainingJob(prepared, backend)
    received = []
    job.event_received.connect(received.append)

    with qtbot.waitSignal(job.finished, timeout=5000) as signal:
        job.start()

    assert signal.args[0].returncode == 0
    assert backend.recorded[0:2] == ("exp_test", 1)
    assert backend.recorded[2] > 0
    assert backend.recorded[3] > 0
    assert received[0]["type"] == "started"
    assert (Path(prepared.run_dir) / "stdout.log").is_file()


def test_process_job_fails_if_hello_is_missing(qtbot, tmp_path):
    prepared = _prepared(tmp_path, "import time; time.sleep(3)")
    job = ProcessTrainingJob(prepared, ProcessRecorder(), hello_timeout_ms=100)

    with qtbot.waitSignal(job.finished, timeout=2000) as signal:
        job.start()

    assert signal.args[0].start_failed
    assert "応答" in signal.args[0].message
    assert job.process.state().name == "NotRunning"


def test_process_job_waits_for_exit_after_registration_failure(qtbot, tmp_path):
    code = (
        "import json, os, sys, time\n"
        "print(json.dumps({'v':1,'run_id':'exp_test/attempt_001','time':time.time(),"
        "'type':'hello','pid':os.getpid()}), flush=True)\n"
        "sys.stdin.readline()\n"
        "import time; time.sleep(5)\n"
    )

    class FailingRecorder(ProcessRecorder):
        def record_training_process(self, *_args):
            raise OSError("process.json を保存できません")

    prepared = _prepared(tmp_path, code)
    job = ProcessTrainingJob(prepared, FailingRecorder())
    states_at_end = []
    job.finished.connect(lambda _outcome: states_at_end.append(job.process.state().name))

    with qtbot.waitSignal(job.finished, timeout=3000) as signal:
        job.start()

    assert signal.args[0].start_failed
    assert "process.json" in signal.args[0].message
    assert states_at_end == ["NotRunning"]


def test_process_job_reports_nonzero_exit_after_go(qtbot, tmp_path):
    code = (
        "import json, os, sys, time\n"
        "print(json.dumps({'v':1,'run_id':'exp_test/attempt_001','time':time.time(),"
        "'type':'hello','pid':os.getpid()}), flush=True)\n"
        "sys.stdin.readline()\n"
        "sys.exit(7)\n"
    )
    backend = ProcessRecorder()
    job = ProcessTrainingJob(_prepared(tmp_path, code), backend)

    with qtbot.waitSignal(job.finished, timeout=5000) as signal:
        job.start()

    assert backend.recorded is not None
    assert signal.args[0].returncode == 7
