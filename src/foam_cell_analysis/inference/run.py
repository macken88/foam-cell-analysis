"""評価プロセスの入口（比較・推論設計 7.2・7.3）。

``python -m foam_cell_analysis.inference.run --run-dir <絶対パス>``

run_spec を読む（不正なら終了コード 2）→ hello → go を待つ（EOF なら 3）→ started →
事前検査 → 推論 → result.json → completed（0）。例外は error.json と error を出して 1。
go 以降は stdin の EOF（親の終了）を監視し、EOF なら終了コード 3 で終わる。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO


def _timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat()


def _watch_stdin(stream: TextIO) -> None:
    """go 後の親プロセス終了を監視する（学習の training/run.py と同じ）。"""
    from foam_cell_analysis.training.run import _watch_stdin as watch

    watch(stream)


def _select_device(model_type: str) -> Any:
    """学習と同じ規則（device_request=auto）で device を決める。

    fake_numpy は torch を使わないので、torch を読み込まずに CPU とする。
    """
    if model_type == "fake_numpy":
        return "cpu"
    from foam_cell_analysis.inference.adapters import select_device

    return select_device()


def run_job(
    run_dir: str | Path,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    start_watchdog: bool = True,
    free_bytes_fn: Callable[[Path], int] | None = None,
    adapter_factory: Callable[..., Any] | None = None,
) -> int:
    """テスト用の差し替え口を持つ評価プロセスの本体。終了コードを返す。"""
    from foam_cell_analysis.inference.protocol import (
        append_event,
        read_run_spec,
        validate_run_spec,
    )
    from foam_cell_analysis.jobs.protocol import atomic_write_json

    run_path = Path(run_dir).resolve()
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    try:
        spec = read_run_spec(run_path)
        validate_run_spec(spec, run_path)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return 2

    run_id = spec["run_id"]
    hello = {"v": 1, "run_id": run_id, "time": _timestamp(), "type": "hello", "pid": os.getpid()}
    output_stream.write(json.dumps(hello, ensure_ascii=False, separators=(",", ":")) + "\n")
    output_stream.flush()
    line = input_stream.readline()
    if not line or line.strip() != "go":
        return 3
    if start_watchdog:
        threading.Thread(target=_watch_stdin, args=(input_stream,), daemon=True).start()

    sequence = 0
    phase = "starting"

    def emit(event_type: str, **fields: Any) -> dict[str, Any]:
        nonlocal sequence
        sequence += 1
        event = {
            "v": 1,
            "run_id": run_id,
            "seq": sequence,
            "time": _timestamp(),
            "type": event_type,
            **fields,
        }
        append_event(run_path / "events.jsonl", event)
        output_stream.write(
            json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        )
        output_stream.flush()
        return event

    def set_phase(name: str) -> None:
        nonlocal phase
        phase = name

    try:
        from foam_cell_analysis.inference.loop import execute_evaluation, write_json_atomic
        from foam_cell_analysis.training.preflight import _version_mismatches

        constraints = Path(__file__).resolve().parents[3] / "constraints-ml.txt"
        versions, mismatches = _version_mismatches(constraints)
        device = _select_device(spec["model_type"])
        write_json_atomic(
            run_path / "environment.json",
            {
                "schema": 1,
                "device": str(device),
                "device_request": spec["device_request"],
                "versions": versions,
                "version_mismatches": mismatches,
                "python": sys.version.split()[0],
                "pid": os.getpid(),
                "time": _timestamp(),
            },
        )
        emit("started", device=str(device), versions=versions, version_mismatches=mismatches)
        for mismatch in mismatches:
            emit("warning", message=f"依存パッケージの版が制約と異なります: {mismatch}")
        execute_evaluation(
            run_path,
            spec,
            emit,
            device=device,
            versions={key: versions[key] for key in ("torch", "torchvision", "cellpose")},
            free_bytes_fn=free_bytes_fn,
            adapter_factory=adapter_factory,
            set_phase=set_phase,
        )
        return 0
    except Exception as error:
        message = str(error) or type(error).__name__
        atomic_write_json(
            run_path / "error.json",
            {
                "schema": 1,
                "phase": phase,
                "exception_type": type(error).__name__,
                "message": message,
                "traceback": traceback.format_exc(),
                "time": _timestamp(),
            },
        )
        try:
            emit("error", phase=phase, message=message)
        except Exception:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    arguments = parser.parse_args(argv)
    return run_job(arguments.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
