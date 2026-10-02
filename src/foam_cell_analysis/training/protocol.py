"""学習プロセスとの JSON / JSONL プロトコル。torch を読み込まない。

イベント外形・JSON Lines・原子的な書き込みは ``jobs/protocol.py`` と共有する。
既存の import を保つため、共通の名前をここから再公開する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foam_cell_analysis.jobs.protocol import (
    ENVELOPE_FIELDS,
    append_jsonl,
    atomic_write_json,
    read_json,
    read_jsonl_events,
)

EVENT_FIELDS: dict[str, set[str]] = {
    "started": {"device", "versions", "version_mismatches"},
    "preflight": {"n_used", "folds", "excluded", "required_bytes", "free_bytes"},
    "phase": {"phase", "fold", "total_epochs"},
    "epoch": {"phase", "fold", "epoch", "loss", "lr"},
    "val": {"fold", "epoch", "ap", "n_images"},
    "checkpoint": {"fold", "epoch", "kind", "path"},
    "fold_done": {"fold", "last_epoch", "early_stopped", "best_epoch"},
    "oof": {"epoch", "ap", "per_fold", "n_images"},
    "selected": {"epoch", "ap", "per_class"},
    "partial_result": {"path"},
    "completed": {"path"},
    "error": {"phase", "message"},
    "warning": {"message"},
}
JSON_FILES = ("run_spec", "resolved_data", "result", "error", "status")


def validate_event(event: dict[str, Any], *, allow_hello: bool = True) -> None:
    """イベントの共通必須項目と型別必須項目を検証する。"""
    kind = event.get("type")
    if kind == "hello" and allow_hello:
        required = {"v", "run_id", "time", "type", "pid"}
        missing = required - event.keys()
        if missing or "seq" in event:
            raise ValueError("hello の共通項目が不足、または seq が指定されています")
        if (
            event["v"] != 1
            or not isinstance(event["run_id"], str)
            or not isinstance(event["time"], str)
            or not isinstance(event["pid"], int)
        ):
            raise ValueError("hello の共通項目の型が不正です")
        return
    if kind not in EVENT_FIELDS:
        raise ValueError(f"未対応イベントです: {kind}")
    required = set(ENVELOPE_FIELDS) | EVENT_FIELDS[kind]
    missing = required - event.keys()
    if missing:
        raise ValueError(f"イベント必須項目がありません: {', '.join(sorted(missing))}")
    if event["v"] != 1 or not isinstance(event["seq"], int) or event["seq"] < 1:
        raise ValueError("イベント v または seq が不正です")
    if not isinstance(event["run_id"], str) or not isinstance(event["time"], str):
        raise ValueError("イベント run_id/time が不正です")


def read_events(path: str | Path, *, expected_run_id: str | None = None) -> list[dict[str, Any]]:
    """JSONL を読み、切れた最終行を無視し seq の重複・欠落を検出する。"""
    return read_jsonl_events(
        path,
        validate=lambda event: validate_event(event, allow_hello=False),
        expected_run_id=expected_run_id,
    )


def append_event(path: str | Path, event: dict[str, Any]) -> None:
    """検証済みイベントを追記し、ディスクへ flush する。"""
    validate_event(event, allow_hello=False)
    append_jsonl(path, event)


def _read_run_file(run_dir: str | Path, name: str) -> dict[str, Any]:
    return read_json(Path(run_dir) / f"{name}.json")


def _write_run_file(run_dir: str | Path, name: str, value: dict[str, Any]) -> None:
    atomic_write_json(Path(run_dir) / f"{name}.json", value)


def read_run_spec(run_dir: str | Path) -> dict[str, Any]:
    return _read_run_file(run_dir, "run_spec")


def write_run_spec(run_dir: str | Path, value: dict[str, Any]) -> None:
    _write_run_file(run_dir, "run_spec", value)


def read_resolved_data(run_dir: str | Path) -> dict[str, Any]:
    return _read_run_file(run_dir, "resolved_data")


def write_resolved_data(run_dir: str | Path, value: dict[str, Any]) -> None:
    _write_run_file(run_dir, "resolved_data", value)


def read_result(run_dir: str | Path) -> dict[str, Any]:
    return _read_run_file(run_dir, "result")


def write_result(run_dir: str | Path, value: dict[str, Any]) -> None:
    _write_run_file(run_dir, "result", value)


def read_partial_result(run_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(run_dir) / "result.partial.json")


def write_partial_result(run_dir: str | Path, value: dict[str, Any]) -> None:
    atomic_write_json(Path(run_dir) / "result.partial.json", value)


def read_error(run_dir: str | Path) -> dict[str, Any]:
    return _read_run_file(run_dir, "error")


def write_error(run_dir: str | Path, value: dict[str, Any]) -> None:
    _write_run_file(run_dir, "error", value)


def read_status(run_dir: str | Path) -> dict[str, Any]:
    return _read_run_file(run_dir, "status")


def write_status(run_dir: str | Path, value: dict[str, Any]) -> None:
    _write_run_file(run_dir, "status", value)
