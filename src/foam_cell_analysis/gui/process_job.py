"""GUI が所有する子プロセスジョブ（学習・評価で共通）。

起動 → hello → Windows Job Object への登録 → process.json（PID と作成時刻）→ go の順に
進め、以降は stdout のイベントを呼び出し側へ渡す。stdout の解釈は比較・推論設計 7.4
（``jobs.protocol.parse_stdout_line``）に従う。各イベントの必須項目の検証と
終端処理は呼び出し側（TrainingRunner / EvaluationRunner）が行う。
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from ..jobs.lifecycle import creation_time_or_now, parent_process_id, write_process_record
from ..jobs.protocol import parse_stdout_line
from ..services.models import JobExit

logger = logging.getLogger(__name__)

HELLO_TIMEOUT_MS = 30_000


@dataclass
class ProcessJobInfo:
    """子プロセスジョブの起動情報と通知先。

    run_id: hello とイベントの run_id と照合する文字列（中身は解釈しない）
    run_dir: stdout.log / stderr.log / process.json を置くフォルダ
    label: 利用者向けの文言に使う名前（例: 「学習プロセス」）
    on_event: 外形が妥当なイベントを受け取る関数（型ごとの検証は呼び出し側）
    on_exit: 終了時に 1 回だけ JobExit を受け取る関数
    record_process: hello 受信後に (pid, 作成時刻) を保存する関数。
        省略時は run_dir/process.json を書く。例外を送出すると起動失敗になる
    """

    run_id: str
    run_dir: str | Path
    program: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | Path | None = None
    label: str = "子プロセス"
    on_event: Callable[[dict[str, Any]], None] | None = None
    on_exit: Callable[[JobExit], None] | None = None
    record_process: Callable[[int, float], None] | None = None
    hello_timeout_ms: int = HELLO_TIMEOUT_MS


class ProcessJob(QObject):
    """hello/go ハンドシェイク付きの子プロセス。

    event_received(dict) と finished(JobExit) を出す。info の on_event / on_exit も
    同じ内容で呼ぶ。finished は必ず 1 回だけ出る。
    """

    event_received = Signal(object)
    finished = Signal(object)

    def __init__(self, info: ProcessJobInfo, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.info = info
        self.run_dir = Path(info.run_dir)
        self.process = QProcess(self)
        self.process.setProgram(info.program)
        self.process.setArguments(list(info.args))
        if info.cwd is not None:
            self.process.setWorkingDirectory(str(info.cwd))
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in info.env.items():
            environment.insert(key, value)
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self._stdout_ready)
        self.process.readyReadStandardError.connect(self._stderr_ready)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._hello_timer = QTimer(self)
        self._hello_timer.setSingleShot(True)
        self._hello_timer.setInterval(info.hello_timeout_ms)
        self._hello_timer.timeout.connect(self._hello_timeout)
        self._hello = False
        self.hello_pid: int | None = None
        self._buffer = bytearray()
        self._job_handle = None
        self._done = False
        self._pending_failure: str | None = None
        self._protocol_failure: str | None = None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._stdout_log = (self.run_dir / "stdout.log").open("ab")
        self._stderr_log = (self.run_dir / "stderr.log").open("ab")

    # ---- 起動と入出力 ----

    def start(self) -> None:
        """プロセスを起動して hello を待つ。"""
        self.process.start()
        self._hello_timer.start()

    def _stdout_ready(self) -> None:
        data = bytes(self.process.readAllStandardOutput())
        self._stdout_log.write(data)
        self._stdout_log.flush()
        self._buffer.extend(data)
        while b"\n" in self._buffer and not self._failing:
            line, _, rest = self._buffer.partition(b"\n")
            self._buffer = bytearray(rest)
            self._handle_line(line)

    def _stderr_ready(self) -> None:
        if self._stderr_log.closed:
            return
        data = bytes(self.process.readAllStandardError())
        self._stderr_log.write(data)
        self._stderr_log.flush()

    @property
    def _failing(self) -> bool:
        return self._done or bool(self._pending_failure or self._protocol_failure)

    def _handle_line(self, line: bytes) -> None:
        parsed = parse_stdout_line(
            line, run_id=self.info.run_id, hello_received=self._hello, label=self.info.label
        )
        if parsed.kind == "error":
            if self._hello:
                self._protocol_error(parsed.message)
            else:
                self._fail(parsed.message)
            return
        if parsed.kind == "ignored":
            # 内容は stdout.log に残っている
            logger.info("%s の JSON でない出力を無視しました", self.info.run_id)
            return
        if parsed.kind == "hello":
            self._accept_hello(parsed.event["pid"])
            return
        if self.info.on_event is not None:
            self.info.on_event(parsed.event)
        self.event_received.emit(parsed.event)

    def _hello_pid_matches(self, pid: int) -> bool:
        """hello の PID が起動したプロセス、またはその直接の子かを確かめる。

        venv の python.exe はランチャーで、実際の Python は子プロセスになるため、
        親 PID が QProcess の PID と一致するものも同じプロセスとして扱う。
        """
        own = int(self.process.processId())
        if own <= 0:
            return False
        return pid == own or parent_process_id(pid) == own

    def _accept_hello(self, pid: int) -> None:
        if not self._hello_pid_matches(pid):
            self._fail(
                f"{self.info.label}の hello の PID が起動したプロセスと一致しません"
                f" (受信 PID: {pid}, 起動 PID: {self.process.processId()})"
            )
            return
        self._hello = True
        self.hello_pid = pid
        self._hello_timer.stop()
        try:
            self._attach_job_object()
            created = creation_time_or_now(pid)
            if self.info.record_process is not None:
                self.info.record_process(pid, created)
            else:
                write_process_record(self.run_dir, pid, created)
            self.process.write(b"go\n")
        except Exception as error:
            self._fail(f"{self.info.label}を開始できません: {error}")

    # ---- Windows Job Object ----

    def _attach_job_object(self) -> None:
        """アプリ終了時に子プロセスも終わるよう Job Object へ登録する。"""
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
        # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: アプリが落ちても子プロセスを残さない
        info.BasicLimitInformation.LimitFlags = 0x2000
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            kernel.CloseHandle(handle)
            raise OSError("Windows Job Object の終了規則を設定できません")
        process_handle = kernel.OpenProcess(0x0001 | 0x0100, False, int(self.process.processId()))
        if not process_handle or not kernel.AssignProcessToJobObject(handle, process_handle):
            if process_handle:
                kernel.CloseHandle(process_handle)
            kernel.CloseHandle(handle)
            raise OSError(f"{self.info.label}を Windows Job Object に登録できません")
        kernel.CloseHandle(process_handle)
        if self.hello_pid is not None and self.hello_pid != int(self.process.processId()):
            # venv のランチャー経由では、実際の Python はランチャーが作った
            # KILL_ON_JOB_CLOSE の Job に入っていて、ここでの登録は拒否されることがある。
            # その場合もランチャーの終了で子は終わるため、登録は試みるだけにする
            child_handle = kernel.OpenProcess(0x0001 | 0x0100, False, self.hello_pid)
            if child_handle:
                kernel.AssignProcessToJobObject(handle, child_handle)
                kernel.CloseHandle(child_handle)
        self._job_handle = handle

    def _close_job_object(self) -> None:
        if self._job_handle:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self._job_handle))
            self._job_handle = None

    # ---- 終了 ----

    def _hello_timeout(self) -> None:
        if not self._hello:
            self._fail(f"{self.info.label}が制限時間内に応答しませんでした")

    def _process_error(self, _error) -> None:
        # hello 以降の異常終了（kill を含む）は finished 側で扱う
        if (
            not self._hello
            and self.process.state() == QProcess.ProcessState.NotRunning
            and not self._pending_failure
            and not self._protocol_failure
        ):
            self._finish(JobExit(start_failed=True, message=self.process.errorString()))

    def _process_finished(self, code, _status) -> None:
        # finished 時点で残っている stderr もログへ追記してから末尾を読む。
        self._stderr_ready()
        if self._pending_failure:
            self._finish(JobExit(start_failed=True, message=self._pending_failure))
            return
        if self._protocol_failure:
            self._finish(
                JobExit(returncode=int(code), message=self._protocol_failure, protocol_error=True)
            )
            return
        if not self._hello:
            detail = self._stderr_tail()
            message = f"hello 前に{self.info.label}が終了しました"
            if detail:
                message = f"{message}: {detail}"
            self._finish(JobExit(start_failed=True, message=message))
            return
        self._finish(JobExit(returncode=int(code), message=f"{self.info.label}が終了しました"))

    def _stderr_tail(self) -> str:
        """stderr.log の短い末尾を安全に利用者向け終了理由へ添える。"""
        try:
            with (self.run_dir / "stderr.log").open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 800))
                data = stream.read(800)
            lines = [line.strip() for line in data.decode("utf-8", errors="replace").splitlines()]
            return " ".join(line for line in lines if line)[-300:]
        except OSError:
            return ""

    def _fail(self, message: str) -> None:
        """起動失敗として子プロセスを終わらせる（hello 前・登録失敗）。"""
        if self._failing:
            return
        self._pending_failure = message
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self._kill_process()
            return
        self._finish(JobExit(start_failed=True, message=message))

    def _protocol_error(self, message: str) -> None:
        """hello 以降のプロトコルエラーとして子プロセスを終わらせる。"""
        if self._failing:
            return
        self._protocol_failure = message
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self._kill_process()
            return
        self._finish(JobExit(message=message, protocol_error=True))

    def _finish(self, outcome: JobExit) -> None:
        if self._done:
            return
        self._done = True
        self._hello_timer.stop()
        for stream in (self._stdout_log, self._stderr_log):
            if not stream.closed:
                stream.close()
        self._close_job_object()
        if self.info.on_exit is not None:
            self.info.on_exit(outcome)
        self.finished.emit(outcome)

    def _kill_process(self) -> None:
        if self._job_handle and os.name == "nt":
            # ランチャー経由の子も含めて終わらせる
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel.TerminateJobObject.restype = ctypes.c_int
            kernel.TerminateJobObject(self._job_handle, 1)
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def kill(self) -> None:
        """子プロセスを終了する（終了通知は finished で届く）。"""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self._kill_process()
