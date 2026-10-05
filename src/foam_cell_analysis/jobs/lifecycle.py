"""ジョブの終端判定とプロセスの照合（学習・評価で共通）。

終端判定は純粋関数 ``decide_terminal_state`` に集約する。「result.json が妥当」の
中身は学習・評価それぞれで検証し、その結果だけをここへ渡す。
プロセスの照合は PID と作成時刻の組で行い、PID の再利用を別プロセスと取り違えない。
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .protocol import atomic_write_json

PROCESS_FILE = "process.json"
# FILETIME is converted to Unix seconds as a float (precision is ~0.24 us).
CREATION_TIME_TOLERANCE_S = 0.000001
CREATION_TIME_TOLERANCE_TICKS = 10


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
    if existing_status is not None and not isinstance(existing_status, dict):
        raise ValueError("終端状態の形式が不正です")
    if stop_request is not None and not isinstance(stop_request, dict):
        raise ValueError("停止要求の形式が不正です")
    if existing_status is not None:
        if not isinstance(existing_status.get("status"), str) or not existing_status["status"]:
            raise ValueError("終端状態の値が不正です")
        if existing_status.get("reason") is not None and not isinstance(
            existing_status.get("reason"), str
        ):
            raise ValueError("終端理由の形式が不正です")
        return TerminalDecision(
            str(existing_status["status"]),
            existing_status.get("reason"),
            "existing",
            existing_status.get("message", ""),
        )
    if stop_request is not None:
        if stop_request.get("reason") is not None and not isinstance(
            stop_request.get("reason"), str
        ):
            raise ValueError("停止理由の形式が不正です")
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
        if type(value) in (int, float):
            result = float(value)
            return result if math.isfinite(result) and result > 0 else None
        if isinstance(value, str):
            try:
                result = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                return result if math.isfinite(result) and result > 0 else None
            except ValueError:
                continue
    return None


def _kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    return kernel32


_QUERY_LIMITED_INFORMATION = 0x1000
_PROCESS_TERMINATE = 0x0001
_SYNCHRONIZE = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def windows_process_handle(
    pid: int, access: int = _QUERY_LIMITED_INFORMATION
) -> tuple[Any, float] | None:
    """生存中の Windows プロセスのハンドルと作成時刻（Unix 秒）を返す。

    access は OpenProcess の要求権限。終了させるときは TERMINATE と SYNCHRONIZE を足す。
    """
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
    handle = kernel32.OpenProcess(access, False, pid)
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


def windows_process_creation_ticks(handle: Any) -> int | None:
    """保持した Windows process handle の作成時刻を FILETIME ticks で返す。"""
    if os.name != "nt" or not handle:
        return None

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    kernel32 = _kernel32()
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FileTime),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    created = FileTime()
    scratch = [FileTime() for _ in range(3)]
    if not kernel32.GetProcessTimes(
        handle,
        ctypes.byref(created),
        ctypes.byref(scratch[0]),
        ctypes.byref(scratch[1]),
        ctypes.byref(scratch[2]),
    ):
        return None
    return (created.high << 32) | created.low


def process_creation_time(pid: int) -> float | None:
    """生存中のプロセスの作成時刻（Unix 秒）を返す。取得できなければ None。"""
    identity = windows_process_handle(pid)
    if identity is None:
        return None
    handle, created_at = identity
    _kernel32().CloseHandle(handle)
    return created_at


def windows_process_state(pid: int, expected_created: float) -> str:
    """Windows でプロセスの不在・終了・別 PID・照合不能を明示的に分ける。"""
    if os.name != "nt":
        return "unknown"
    kernel32 = _kernel32()
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    handle = kernel32.OpenProcess(_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error in {87, 1168}:
            return "absent"
        return "unknown"
    try:
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        if kernel32.WaitForSingleObject(handle, 0) == 0:
            return "terminated"
        identity = windows_process_handle(pid, _QUERY_LIMITED_INFORMATION | _SYNCHRONIZE)
        if identity is None:
            return "unknown"
        inner, created = identity
        kernel32.CloseHandle(inner)
        if abs(created - expected_created) > CREATION_TIME_TOLERANCE_S:
            return "different"
        return "running"
    finally:
        kernel32.CloseHandle(handle)


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


def _windows_command_line_args(command_line: str) -> list[str] | None:
    """Windows の標準 CommandLineToArgvW でコマンドラインを分解する。"""
    if os.name != "nt":
        return None
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        kernel32 = _kernel32()
        shell32.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
        shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        count = ctypes.c_int()
        argv = shell32.CommandLineToArgvW(command_line, ctypes.byref(count))
        if not argv or count.value <= 0:
            return None
        try:
            return [argv[index] for index in range(count.value)]
        finally:
            kernel32.LocalFree(argv)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def process_alive(process: dict[str, Any]) -> bool:
    """保存 PID と作成時刻が一致する Windows プロセスだけを生存扱いする。"""
    try:
        if type(process["pid"]) is not int or process["pid"] <= 0:
            return False
        pid = process["pid"]
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


def terminate_process(process: dict[str, Any]) -> bool:
    """PID と作成時刻が一致するプロセスを終了して待機する。

    終了できた（もともと存在しない場合を含む）ときは True、終了できなかったときは False を返す。
    """
    try:
        if type(process["pid"]) is not int or process["pid"] <= 0:
            return False
        pid = process["pid"]
    except (KeyError, TypeError, ValueError):
        return False
    expected_created = process_created_at(process)
    if expected_created is None:
        return False
    if os.name == "nt":
        handles = None
        for _ in range(3):
            handles = _windows_process_tree_handles(pid, expected_created)
            if handles is not None:
                break
            time.sleep(0.01)
        if not handles:
            return False
        kernel32 = _kernel32()
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.TerminateProcess.restype = ctypes.c_int
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        done = True
        held = list(handles)
        known = {pid: (created, depth) for pid, _handle, created, depth in held}
        current = held
        try:
            while True:
                for _pid, handle, _created, _depth in sorted(
                    current, key=lambda item: item[3], reverse=True
                ):
                    if kernel32.WaitForSingleObject(handle, 0) != 0:
                        if not kernel32.TerminateProcess(handle, 1):
                            done = False
                for _pid, handle, _created, _depth in current:
                    if kernel32.WaitForSingleObject(handle, 5000) != 0:
                        done = False
                descendants = _windows_snapshot_descendants(known)
                if descendants is None:
                    return False
                if not descendants:
                    return done
                held.extend(descendants)
                for child_pid, _handle, created, depth in descendants:
                    known[child_pid] = (created, depth)
                current = descendants
                if len(held) > 4096:
                    return False
        finally:
            for _pid, handle, _created, _depth in held:
                kernel32.CloseHandle(handle)
    return False


def _windows_process_tree_handles(
    root_pid: int, expected_created: float
) -> list[tuple[int, Any, float, int]] | None:
    """列挙した PID を作成時刻で照合し、終了待機まで全ハンドルを保持する。"""
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
    snapshot_time = time.time()
    parents: dict[int, int] = {}
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(ProcessEntry)
        found = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            found = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    if root_pid not in parents:
        return None
    depths = {root_pid: 0}
    changed = True
    while changed:
        changed = False
        for child, parent in parents.items():
            if child not in depths and parent in depths:
                depths[child] = depths[parent] + 1
                changed = True
    result = []
    created_by_pid: dict[int, float] = {}
    for child, depth in sorted(depths.items(), key=lambda item: item[1]):
        identity = windows_process_handle(
            child,
            _QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE | _SYNCHRONIZE,
        )
        if identity is None:
            error = ctypes.get_last_error()
            if error in {87, 1168} and child != root_pid:
                # 既に自然終了した子は対象外。権限不足などは不確定として保護する。
                continue
            for _pid, handle, _created, _depth in result:
                kernel32.CloseHandle(handle)
            return None
        handle, created = identity
        if child == root_pid and abs(created - expected_created) > CREATION_TIME_TOLERANCE_S:
            kernel32.CloseHandle(handle)
            for _prior_pid, prior, _created, _depth in result:
                kernel32.CloseHandle(prior)
            # PID was reused: never terminate the new process.
            return None
        if created > snapshot_time + CREATION_TIME_TOLERANCE_S:
            kernel32.CloseHandle(handle)
            for _pid, prior, _created, _depth in result:
                kernel32.CloseHandle(prior)
            return None
        if depth:
            parent = parents[child]
            parent_created = created_by_pid.get(parent)
            actual_parent = _windows_parent_from_handle(handle)
            if (
                parent_created is None
                or actual_parent != parent
                or created + CREATION_TIME_TOLERANCE_S < parent_created
            ):
                kernel32.CloseHandle(handle)
                for _prior_pid, prior, _created, _depth in result:
                    kernel32.CloseHandle(prior)
                return None
        created_by_pid[child] = created
        result.append((child, handle, created, depth))
    return result


def _windows_snapshot_descendants(
    known: dict[int, tuple[float, int]],
) -> list[tuple[int, Any, float, int]] | None:
    """終了済みの親 PID は保持済みidentityで固定し、残存する新しい子孫を再列挙する。"""
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
    snapshot_time = time.time()
    parents: dict[int, int] = {}
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(ProcessEntry)
        found = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            found = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    depths: dict[int, int] = {}
    changed = True
    while changed:
        changed = False
        for child, parent in parents.items():
            if child in known or child in depths:
                continue
            if parent in known:
                depths[child] = known[parent][1] + 1
                changed = True
            elif parent in depths:
                depths[child] = depths[parent] + 1
                changed = True
    result: list[tuple[int, Any, float, int]] = []
    for child, depth in sorted(depths.items(), key=lambda item: item[1]):
        identity = windows_process_handle(
            child, _QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE | _SYNCHRONIZE
        )
        if identity is None:
            if ctypes.get_last_error() in {87, 1168}:
                continue
            for _pid, handle, _created, _depth in result:
                kernel32.CloseHandle(handle)
            return None
        handle, created = identity
        parent = parents[child]
        parent_created = (
            known[parent][0]
            if parent in known
            else next((value[2] for value in result if value[0] == parent), None)
        )
        if (
            created > snapshot_time + CREATION_TIME_TOLERANCE_S
            or parent_created is None
            or created + CREATION_TIME_TOLERANCE_S < parent_created
            or _windows_parent_from_handle(handle) != parent
        ):
            kernel32.CloseHandle(handle)
            for _pid, prior, _created, _depth in result:
                kernel32.CloseHandle(prior)
            return None
        result.append((child, handle, created, depth))
    return result


def _windows_parent_from_handle(handle: Any) -> int | None:
    """保持したプロセスハンドルから実際の親 PID を取得する。"""
    if os.name != "nt":
        return None

    class ProcessBasicInformation(ctypes.Structure):
        _fields_ = [
            ("reserved1", ctypes.c_void_p),
            ("peb_base_address", ctypes.c_void_p),
            ("reserved2", ctypes.c_void_p * 2),
            ("process_id", ctypes.c_void_p),
            ("inherited_from_unique_process_id", ctypes.c_void_p),
        ]

    try:
        ntdll = ctypes.WinDLL("ntdll")
        query = ntdll.NtQueryInformationProcess
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
        ]
        query.restype = ctypes.c_long
        information = ProcessBasicInformation()
        status = query(handle, 0, ctypes.byref(information), ctypes.sizeof(information), None)
        if status != 0:
            return None
        return int(information.inherited_from_unique_process_id or 0)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def settle_unrecorded_worker(
    run_dir: str | Path,
    module: str,
    alive: Callable[[dict[str, Any]], bool],
    terminator: Callable[[dict[str, Any]], Any],
) -> str:
    """process.json/hello.json がない場合、実行モジュールと run_dir でだけ OS を照合する。"""
    # 差し替えアダプターはテスト・埋込み環境のプロセス台帳を表す。
    if alive is not process_alive:
        return "dead"
    if os.name != "nt":
        return "unconfirmed"
    canonical = str(Path(run_dir).resolve()).casefold()
    command = (
        "$ErrorActionPreference='Stop'; "
        "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; "
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,Name,CommandLine,ExecutablePath,"
        "@{Name='CreationFileTime';Expression={$_.CreationDate.ToUniversalTime().ToFileTimeUtc()}} "
        "| ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            timeout=10,
            encoding="utf-8",
            errors="strict",
        )
        if completed.returncode != 0:
            return "unconfirmed"
        rows = json.loads(completed.stdout.lstrip("\ufeff") or "[]")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, UnicodeError):
        return "unconfirmed"
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return "unconfirmed"
    matching: list[dict[str, Any]] = []
    allowed_executables = {os.path.normcase(os.path.realpath(sys.executable))}
    base_executable = getattr(sys, "_base_executable", None)
    if base_executable:
        allowed_executables.add(os.path.normcase(os.path.realpath(base_executable)))
    for row in rows:
        if not isinstance(row, dict):
            return "unconfirmed"
        name = str(row.get("Name") or "").casefold()
        line_value = row.get("CommandLine")
        if not line_value and name in {"python.exe", "pythonw.exe"}:
            # 列挙した Python worker の引数を読めず、対象不在を証明できない。
            return "unconfirmed"
        if not isinstance(line_value, str):
            continue
        args = _windows_command_line_args(line_value)
        if args is None:
            return "unconfirmed"
        normalized_args = [arg.casefold() for arg in args]
        expected_module = module.casefold()
        # Python の実際の module 起動形式だけを認める。-c 等の後続引数に
        # 偽の "-m module" を埋めたプロセスを worker と誤認しない。
        if len(normalized_args) < 3 or normalized_args[1:3] != ["-m", expected_module]:
            continue
        run_arg = next(
            (
                args[index + 1]
                for index, arg in enumerate(normalized_args[:-1])
                if arg == "--run-dir"
            ),
            None,
        )
        if run_arg is None or os.path.normcase(os.path.realpath(run_arg)) != os.path.normcase(
            canonical
        ):
            continue
        executable = row.get("ExecutablePath")
        if not isinstance(executable, str):
            return "unconfirmed"
        if os.path.normcase(os.path.realpath(executable)) not in allowed_executables:
            continue
        try:
            created_ticks = row.get("CreationFileTime")
            if type(created_ticks) not in (int, float) or not math.isfinite(created_ticks):
                return "unconfirmed"
            pid = row["ProcessId"]
            if type(pid) is not int or pid <= 0:
                return "unconfirmed"
            matching.append({"pid": pid, "creation_ticks": int(created_ticks)})
        except (KeyError, TypeError, ValueError, IndexError):
            return "unconfirmed"
    for record in matching:
        pid = record["pid"]
        identity = windows_process_handle(
            pid, _QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE | _SYNCHRONIZE
        )
        if identity is None:
            expected_created = record["creation_ticks"] / 10_000_000 - 11_644_473_600
            if windows_process_state(pid, expected_created) in {
                "absent",
                "different",
                "terminated",
            }:
                continue
            return "unconfirmed"
        handle, created = identity
        try:
            actual_ticks = windows_process_creation_ticks(handle)
            if (
                actual_ticks is None
                or abs(actual_ticks - record["creation_ticks"]) > CREATION_TIME_TOLERANCE_TICKS
            ):
                expected_created = record["creation_ticks"] / 10_000_000 - 11_644_473_600
                state = windows_process_state(pid, expected_created)
                if state in {"absent", "different", "terminated"}:
                    continue
                return "unconfirmed"
            record["creation_time"] = created
            if not alive(record):
                if windows_process_state(pid, created) in {"absent", "different", "terminated"}:
                    continue
                return "unconfirmed"
            result = terminator(record)
            deadline = time.monotonic() + 5.0
            while alive(record) and time.monotonic() < deadline:
                time.sleep(0.1)
            if alive(record) or result is False:
                return "unconfirmed"
        finally:
            _kernel32().CloseHandle(handle)
    return "terminated" if matching else "dead"


def process_record(pid: int, creation_time: float) -> dict[str, Any]:
    """process.json に保存する内容を作る。"""
    return {"pid": int(pid), "creation_time": float(creation_time)}


def write_process_record(run_dir: str | Path, pid: int, creation_time: float) -> None:
    """run_dir/process.json を原子的に書く。"""
    atomic_write_json(Path(run_dir) / PROCESS_FILE, process_record(pid, creation_time))


def creation_time_or_now(pid: int) -> float:
    """プロセスの正確な作成時刻を返し、取得不能なら失敗させる。"""
    created = process_creation_time(pid)
    if created is None or not math.isfinite(created) or created <= 0:
        raise OSError("プロセスの作成時刻を確認できません")
    return created


def settle_process(
    process_path: str | Path,
    alive: Callable[[dict[str, Any]], bool],
    terminator: Callable[[dict[str, Any]], Any],
    *,
    timeout_s: float = 5.0,
) -> str:
    """前回プロセス記録を照合し、同一プロセスだけを終了確認する。

    ``missing`` はプロセス記録がない場合、``dead`` は照合済み記録の PID が
    既に存在しない場合、``terminated`` は終了確認済み、``unconfirmed`` は
    壊れた記録・照合不能・権限不足・終了確認不能を表す。
    ``alive`` / ``terminator`` は旧サービス引数の差し替え点を維持する。
    """
    path = Path(process_path)
    if not path.exists():
        return "missing"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return "unconfirmed"
        if type(record.get("pid")) is not int or record["pid"] <= 0:
            return "unconfirmed"
        pid = record["pid"]
        created = process_created_at(record)
        if pid <= 0 or created is None:
            return "unconfirmed"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return "unconfirmed"
    try:
        if not alive(record):
            if os.name == "nt" and alive is process_alive:
                state = windows_process_state(pid, created)
                if state in {"unknown", "running"}:
                    return "unconfirmed"
            return "dead"
        result = terminator(record)
        deadline = time.monotonic() + timeout_s
        while alive(record) and time.monotonic() < deadline:
            time.sleep(0.1)
        if alive(record):
            return "unconfirmed"
        if result is False:
            if os.name == "nt":
                state = windows_process_state(pid, created)
                if state == "terminated":
                    return "terminated"
                if state == "absent":
                    return "dead"
            return "unconfirmed"
        return "terminated"
    except (OSError, PermissionError, ValueError, TypeError):
        return "unconfirmed"
