"""GUI が所有する学習プロセスとモック学習ジョブ。"""

from __future__ import annotations

import ctypes
import json
import os
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from ..services.models import JobExit, PreparedRun


def _valid_event_time(value) -> bool:
    """イベント時刻として扱える Unix 時刻または ISO 文字列か確認する。"""
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


class ProcessTrainingJob(QObject):
    """hello/go ハンドシェイク付きの学習子プロセス。"""

    event_received = Signal(object)
    finished = Signal(object)

    def __init__(self, prepared: PreparedRun, backend, hello_timeout_ms: int = 30_000, parent=None):
        super().__init__(parent)
        self.prepared = prepared
        self.backend = backend
        self.experiment_id = prepared.run_id.split("/attempt_", 1)[0]
        self.key = f"training:{self.experiment_id}"
        self.process = QProcess(self)
        self.process.setProgram(prepared.program)
        self.process.setArguments(prepared.args)
        self.process.setWorkingDirectory(str(Path(prepared.run_dir).resolve().parents[1]))
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in prepared.env.items():
            environment.insert(key, value)
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self._stdout_ready)
        self.process.readyReadStandardError.connect(self._stderr_ready)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._hello_timer = QTimer(self)
        self._hello_timer.setSingleShot(True)
        self._hello_timer.setInterval(hello_timeout_ms)
        self._hello_timer.timeout.connect(self._hello_timeout)
        self._hello = False
        self._hello_pid = None
        self._buffer = bytearray()
        self._job_handle = None
        self._done = False
        self._pending_failure = None
        run_dir = Path(prepared.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._stdout_log = (run_dir / "stdout.log").open("ab")
        self._stderr_log = (run_dir / "stderr.log").open("ab")

    def start(self):
        """プロセスを起動して hello を待つ。"""
        self.process.start()
        self._hello_timer.start()

    def _stdout_ready(self):
        data = bytes(self.process.readAllStandardOutput())
        self._stdout_log.write(data)
        self._stdout_log.flush()
        self._buffer.extend(data)
        while b"\n" in self._buffer:
            line, _, rest = self._buffer.partition(b"\n")
            self._buffer = bytearray(rest)
            self._handle_line(line)

    def _stderr_ready(self):
        data = bytes(self.process.readAllStandardError())
        self._stderr_log.write(data)
        self._stderr_log.flush()

    def _handle_line(self, line: bytes):
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._fail("学習プロセスから不正な JSON を受信しました")
        if not self._hello:
            required = {"v", "run_id", "time", "type", "pid"}
            if (
                not required.issubset(event)
                or event.get("v") != 1
                or event.get("type") != "hello"
                or event.get("run_id") != self.prepared.run_id
                or not _valid_event_time(event.get("time"))
                or not isinstance(event.get("pid"), int)
                or event.get("pid", 0) <= 0
            ):
                return self._fail(f"学習プロセスの hello が不正です (受信 PID: {event.get('pid')})")
            self._hello = True
            self._hello_pid = event["pid"]
            self._hello_timer.stop()
            try:
                self._attach_job_object()
                experiment_id, attempt = self.prepared.run_id.rsplit("/attempt_", 1)
                self.backend.record_training_process(
                    experiment_id, int(attempt), self._hello_pid, time.time()
                )
                self.process.write(b"go\n")
            except Exception as error:
                self._fail(f"学習プロセスを開始できません: {error}")
            return
        self.event_received.emit(event)

    def _attach_job_object(self):
        if os.name != "nt":
            return
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel.CreateJobObjectW.restype = ctypes.c_void_p
        kernel.SetInformationJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        kernel.SetInformationJobObject.restype = ctypes.c_int
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel.AssignProcessToJobObject.restype = ctypes.c_int
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_int
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise OSError("Windows Job Object を作成できません")

        class BasicLimit(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_ulong),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_ulong),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_ulong),
                ("SchedulingClass", ctypes.c_ulong),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimit),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = 0x2000
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            kernel.CloseHandle(handle)
            raise OSError("Windows Job Object の終了規則を設定できません")
        process_handle = kernel.OpenProcess(0x0001 | 0x0100, False, int(self.process.processId()))
        if not process_handle or not kernel.AssignProcessToJobObject(handle, process_handle):
            if process_handle:
                kernel.CloseHandle(process_handle)
            kernel.CloseHandle(handle)
            raise OSError("学習プロセスを Windows Job Object に登録できません")
        kernel.CloseHandle(process_handle)
        self._job_handle = handle

    def _hello_timeout(self):
        if not self._hello:
            self._fail("学習プロセスが制限時間内に応答しませんでした")

    def _process_error(self, _error):
        if self.process.state() == QProcess.ProcessState.NotRunning and not self._pending_failure:
            message = self._pending_failure or self.process.errorString()
            self._finish(JobExit(start_failed=True, message=message))

    def _process_finished(self, code, _status):
        if self._pending_failure:
            self._finish(JobExit(start_failed=True, message=self._pending_failure))
            return
        if not self._hello:
            self._finish(JobExit(start_failed=True, message="hello 前に学習プロセスが終了しました"))
            return
        self._finish(JobExit(returncode=int(code), message="学習プロセスが終了しました"))

    def _fail(self, message):
        if self._done or self._pending_failure:
            return
        self._pending_failure = message
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            return
        self._finish(JobExit(start_failed=True, message=message))

    def _finish(self, outcome):
        if self._done:
            return
        self._done = True
        self._hello_timer.stop()
        for stream in (self._stdout_log, self._stderr_log):
            if not stream.closed:
                stream.close()
        if self._job_handle:
            ctypes.windll.kernel32.CloseHandle(self._job_handle)
            self._job_handle = None
        self.finished.emit(outcome)

    def kill(self):
        """学習プロセスを終了する。"""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()


class FakeTrainingJob(QObject):
    """モック Backend にイベント列を送る時間差ジョブ。"""

    event_received = Signal(object)
    finished = Signal(object)

    def __init__(self, prepared: PreparedRun, experiment, interval_ms=30, parent=None):
        super().__init__(parent)
        self.prepared = prepared
        self.experiment = experiment
        self.experiment_id = experiment.experiment_id
        self.key = f"training:{self.experiment_id}"
        self.interval_ms = interval_ms
        try:
            speed = max(0.001, float(os.getenv("FOAM_MOCK_SPEED", "1.0")))
        except ValueError:
            speed = 1.0
        self.interval_ms = max(0, int(self.interval_ms / speed))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._step = 0
        self.step = 0
        self._seq = 0
        self._events = self._make_events()
        self.total_steps = max(1, self.experiment.total_epochs) * (
            max(1, len(set(self.experiment.fold_assignments.values()))) + 1
        )

    def _make_events(self):
        epochs = max(1, self.experiment.total_epochs)
        folds = max(1, len(set(self.experiment.fold_assignments.values())))
        events = [{"type": "phase", "phase": "cv"}]
        for fold in range(1, folds + 1):
            for epoch in range(1, epochs + 1):
                loss = max(0.02, 1.2 * (1 - epoch / epochs))
                events.append(
                    {"type": "epoch", "phase": "cv", "fold": fold, "epoch": epoch, "loss": loss}
                )
                if epoch % 5 == 0 or epoch == epochs:
                    events.append(
                        {
                            "type": "val",
                            "fold": fold,
                            "epoch": epoch,
                            "ap": min(0.99, 0.35 + 0.6 * epoch / epochs),
                        }
                    )
        events.append(
            {
                "type": "oof",
                "epoch": epochs,
                "ap": 0.8,
                "per_fold": {str(fold): 0.8 for fold in range(1, folds + 1)},
            }
        )
        events.append({"type": "selected", "epoch": epochs, "ap": 0.8, "per_class": {}})
        events.append({"type": "checkpoint", "epoch": epochs, "path": "selected.pt"})
        events.append({"type": "phase", "phase": "final"})
        for epoch in range(1, epochs + 1):
            events.append(
                {
                    "type": "epoch",
                    "phase": "final",
                    "epoch": epoch,
                    "loss": max(0.02, 1.2 * (1 - epoch / epochs)),
                }
            )
        events.append({"type": "checkpoint", "epoch": epochs, "path": "final.pt"})
        return events

    def start(self):
        self._timer.start(self.interval_ms)

    def _advance(self):
        if self._step >= len(self._events):
            self._timer.stop()
            self.finished.emit(JobExit(returncode=0))
            return
        self._seq += 1
        event = {
            "v": 1,
            "run_id": self.prepared.run_id,
            "time": time.time(),
            "seq": self._seq,
            **self._events[self._step],
        }
        self._step += 1
        if event["type"] == "epoch":
            self.step += 1
        self.event_received.emit(event)

    def kill(self):
        self._timer.stop()
        self.finished.emit(JobExit(returncode=-1, message="モック学習を中断しました"))

    def cancel(self):
        """旧画面テストとの互換用に停止操作を別名で提供する。"""
        parent = self.parent()
        if hasattr(parent, "request_stop"):
            parent.request_stop("user_stop")
        else:
            self.kill()
