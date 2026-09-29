"""ジョブの終端判定とプロセスの照合（学習・評価で共通）。

終端判定は純粋関数 ``decide_terminal_state`` に集約する。「result.json が妥当」の
中身は学習・評価それぞれで検証し、その結果だけをここへ渡す。
プロセスの照合は PID と作成時刻の組で行い、PID の再利用を別プロセスと取り違えない。
"""

from __future__ import annotations

import ctypes
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .protocol import atomic_write_json

PROCESS_FILE = "process.json"
CREATION_TIME_TOLERANCE_S = 2.0


@dataclass(frozen=True)
class TerminalDecision:
    """確定した終端状態。

    source は判定の根拠: "existing"（保存済みの status.json）、"stop_request"、
    "result"、"start_failed"、"error"、"process_dead"。
    source が "existing" のときは status.json を書き直さない。
    message は existing のときだけ入る。error / start_failed の文言は呼び出し側が補う。
    """

    status: str
    reason: str | None
    source: str
    message: str = ""


def _resolve(value: bool | Callable[[], bool]) -> bool:
    return bool(value() if callable(value) else value)


def decide_terminal_state(
    *,
    existing_status: dict[str, Any] | None,
    stop_request: dict[str, Any] | None,
    result_valid: bool | Callable[[], bool],
    error_present: bool | Callable[[], bool],
    start_failed: bool,
    process_alive: bool | Callable[[], bool],
) -> TerminalDecision | None:
    """学習 5.2 の優先順位で終端状態を決める。None はまだ確定しない（プロセスが生きている）。

    優先順位: 保存済みの状態 → 停止要求 → 妥当な結果 → エラー/起動失敗 →
    プロセスが終了済みなら stopped/interrupted → 生きていれば None。
    result_valid・error_present・process_alive は bool か、必要になったときだけ
    評価する引数なし関数を渡せる（重い検証やプロセス照合を不要なら行わない）。
    """
    if existing_status is not None:
        return TerminalDecision(
            str(existing_status["status"]),
            existing_status.get("reason"),
            "existing",
            existing_status.get("message", ""),
        )
    if stop_request is not None:
        return TerminalDecision("stopped", stop_request.get("reason"), "stop_request")
    if _resolve(result_valid):
        return TerminalDecision("completed", None, "result")
    if start_failed:
        return TerminalDecision("failed", "start_failed", "start_failed")
    if _resolve(error_present):
        return TerminalDecision("failed", "error", "error")
    if not _resolve(process_alive):
        return TerminalDecision("stopped", "interrupted", "process_dead")
    return None


# ---- プロセスの照合（PID + 作成時刻） ----


def process_created_at(process: dict[str, Any]) -> float | None:
    """process.json の作成時刻を Unix 秒へ変換する。"""
    for key in ("creation_time", "create_time", "created_at", "process_created_at"):
        value = process.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
    return None


def _kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    return kernel32


def windows_process_handle(pid: int) -> tuple[Any, float] | None:
    """生存中の Windows プロセスのハンドルと作成時刻（Unix 秒）を返す。"""
    if os.name != "nt":
        return None

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    kernel32 = _kernel32()
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FileTime),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    created = FileTime()
    exit_code = ctypes.c_uint32()
    # GetProcessTimes の未使用の出力にも領域を渡す（None だと失敗する環境がある）
    scratch = [FileTime() for _ in range(3)]
    if not kernel32.GetProcessTimes(
        handle,
        ctypes.byref(created),
        ctypes.byref(scratch[0]),
        ctypes.byref(scratch[1]),
        ctypes.byref(scratch[2]),
    ):
        kernel32.CloseHandle(handle)
        return None
    if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
        kernel32.CloseHandle(handle)
        return None
    windows_ticks = (created.high << 32) | created.low
    created_at = windows_ticks / 10_000_000 - 11_644_473_600
    return handle, created_at


def process_creation_time(pid: int) -> float | None:
    """生存中のプロセスの作成時刻（Unix 秒）を返す。取得できなければ None。"""
    identity = windows_process_handle(pid)
    if identity is None:
        return None
    handle, created_at = identity
    _kernel32().CloseHandle(handle)
    return created_at


def parent_process_id(pid: int) -> int | None:
    """プロセスの親 PID を返す。Windows 以外や取得できないときは None。

    venv の python.exe はランチャーで、実際の Python は子プロセスとして動く。
    hello の PID が QProcess の PID の子かどうかを確かめるために使う。
    """
    if os.name != "nt":
        return None

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_uint32),
            ("cntUsage", ctypes.c_uint32),
            ("th32ProcessID", ctypes.c_uint32),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", ctypes.c_uint32),
            ("cntThreads", ctypes.c_uint32),
            ("th32ParentProcessID", ctypes.c_uint32),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_uint32),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = _kernel32()
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry)]
    kernel32.Process32FirstW.restype = ctypes.c_int
    kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry)]
    kernel32.Process32NextW.restype = ctypes.c_int
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return None
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(ProcessEntry)
        found = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            if entry.th32ProcessID == pid:
                return int(entry.th32ParentProcessID)
            found = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return None


def process_alive(process: dict[str, Any]) -> bool:
    """保存 PID と作成時刻が一致する Windows プロセスだけを生存扱いする。"""
    try:
        pid = int(process["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_created = process_created_at(process)
    if expected_created is None:
        return False
    identity = windows_process_handle(pid)
    if identity is None:
        return False
    handle, actual_created = identity
    _kernel32().CloseHandle(handle)
    return abs(expected_created - actual_created) <= CREATION_TIME_TOLERANCE_S


def terminate_process(process: dict[str, Any]) -> None:
    """PID と作成時刻が一致するプロセスを終了して待機する。"""
    try:
        pid = int(process["pid"])
    except (KeyError, TypeError, ValueError):
        return
    expected_created = process_created_at(process)
    identity = windows_process_handle(pid) if expected_created is not None else None
    if identity is None:
        return
    handle, actual_created = identity
    kernel32 = _kernel32()
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.TerminateProcess.restype = ctypes.c_int
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    if abs(expected_created - actual_created) <= CREATION_TIME_TOLERANCE_S:
        kernel32.TerminateProcess(handle, 1)
        kernel32.WaitForSingleObject(handle, 5000)
    kernel32.CloseHandle(handle)


def process_record(pid: int, creation_time: float) -> dict[str, Any]:
    """process.json に保存する内容を作る。"""
    return {"pid": int(pid), "creation_time": float(creation_time)}


def write_process_record(run_dir: str | Path, pid: int, creation_time: float) -> None:
    """run_dir/process.json を原子的に書く。"""
    atomic_write_json(Path(run_dir) / PROCESS_FILE, process_record(pid, creation_time))


def creation_time_or_now(pid: int) -> float:
    """プロセスの作成時刻を返す。取得できない環境では現在時刻で代用する。"""
    created = process_creation_time(pid)
    return created if created is not None else time.time()
