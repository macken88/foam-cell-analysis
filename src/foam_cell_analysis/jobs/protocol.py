"""子プロセスとの共通プロトコル（イベント外形・JSON Lines・原子的な書き込み）。

学習と評価で共有する部分だけを置く。各イベントの必須項目は
``training/protocol.py`` など各処理側で定義する。torch と Qt を読み込まない。
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

PROTOCOL_VERSION = 1
ENVELOPE_FIELDS = frozenset({"v", "run_id", "seq", "time", "type"})
HELLO_FIELDS = frozenset({"v", "run_id", "time", "type", "pid"})


def valid_event_time(value: Any) -> bool:
    """イベント時刻として扱える Unix 時刻または ISO 文字列か確認する。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def validate_envelope(event: dict[str, Any], *, expected_run_id: str | None = None) -> None:
    """hello 以外のイベントの外形（v, run_id, seq, time, type）を検証する。

    外形が欠けている・型が違う場合は ValueError を送出する（プロトコルエラー）。
    """
    missing = ENVELOPE_FIELDS - event.keys()
    if missing:
        raise ValueError(f"イベントの共通項目がありません: {', '.join(sorted(missing))}")
    if event["v"] != PROTOCOL_VERSION:
        raise ValueError(f"イベントの v が不正です: {event['v']!r}")
    seq = event["seq"]
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
        raise ValueError(f"イベントの seq が不正です: {seq!r}")
    if not isinstance(event["type"], str) or not event["type"]:
        raise ValueError("イベントの type が不正です")
    if not isinstance(event["run_id"], str):
        raise ValueError("イベントの run_id が不正です")
    if expected_run_id is not None and event["run_id"] != expected_run_id:
        raise ValueError(f"イベントの run_id が一致しません: {event['run_id']}")
    if not valid_event_time(event["time"]):
        raise ValueError("イベントの time が不正です")


def validate_hello(event: dict[str, Any], *, expected_run_id: str | None = None) -> None:
    """起動直後の hello（seq なし、pid あり）を検証する。"""
    missing = HELLO_FIELDS - event.keys()
    if missing or "seq" in event:
        raise ValueError("hello の共通項目が不足、または seq が指定されています")
    pid = event["pid"]
    if (
        event["v"] != PROTOCOL_VERSION
        or event["type"] != "hello"
        or not isinstance(event["run_id"], str)
        or not valid_event_time(event["time"])
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
    ):
        raise ValueError("hello の共通項目の型が不正です")
    if expected_run_id is not None and event["run_id"] != expected_run_id:
        raise ValueError(f"hello の run_id が一致しません: {event['run_id']}")


SeqCheck = Literal["next", "duplicate", "gap"]


def classify_seq(last_seq: int, seq: int) -> SeqCheck:
    """受信した seq を直前の seq と比べ、次・重複・欠落のどれかを返す。

    重複は無視してよく、欠落は events.jsonl からの再生で埋める（学習 7.3）。
    """
    if seq <= last_seq:
        return "duplicate"
    if seq > last_seq + 1:
        return "gap"
    return "next"


def read_jsonl_events(
    path: str | Path,
    *,
    validate: Callable[[dict[str, Any]], None] | None = None,
    expected_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """events.jsonl を読み、切れた最終行を無視し seq の重複・欠落を検出する。

    ``validate`` を省略すると外形（validate_envelope）だけを検証する。
    途中の未完了行、壊れた JSON、seq の重複・欠落は ValueError にする。
    """
    check = validate or validate_envelope
    raw = Path(path).read_bytes()
    lines = raw.splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.endswith((b"\n", b"\r")):
            if index == len(lines) - 1:
                break
            raise ValueError(f"events.jsonl の途中に未完了行があります: {index + 1}")
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"events.jsonl の JSON が壊れています: {index + 1}") from error
        if not isinstance(event, dict):
            raise ValueError(f"イベントは JSON object である必要があります: {index + 1}")
        check(event)
        if expected_run_id is not None and event["run_id"] != expected_run_id:
            raise ValueError("events.jsonl の run_id が一致しません")
        expected_seq = len(events) + 1
        if event["seq"] != expected_seq:
            kind = "重複" if event["seq"] < expected_seq else "欠落"
            raise ValueError(f"events.jsonl の seq に{kind}があります: {event['seq']}")
        events.append(event)
    return events


def append_jsonl(path: str | Path, event: dict[str, Any]) -> None:
    """1 行の JSON を追記し、ディスクへ flush する（検証は呼び出し側で行う）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    """同じフォルダの一時ファイルから原子的に JSON を置き換える。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def read_json(path: str | Path) -> dict[str, Any]:
    """JSON object を読む。"""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON のトップレベルは object である必要があります: {path}")
    return value


@dataclass(frozen=True)
class StdoutLine:
    """子プロセスの stdout 1 行を解釈した結果。

    kind:
      - "hello": 妥当な hello（event に内容）
      - "event": 外形が妥当なイベント（型ごとの必須項目は未検証）
      - "ignored": hello 以降の JSON object でない行（stdout.log に残して無視する）
      - "error": プロトコルエラー（message に理由）。ジョブを失敗にする
    """

    kind: Literal["hello", "event", "ignored", "error"]
    event: dict[str, Any] | None = None
    message: str = ""


def parse_stdout_line(
    line: bytes, *, run_id: str, hello_received: bool, label: str = "子プロセス"
) -> StdoutLine:
    """stdout の 1 行を比較・推論設計 7.4 の規則で解釈する。

    hello より前は厳格に解釈し、hello として妥当でなければ error（起動失敗）。
    hello 以降は、JSON object として読めない行を ignored、JSON object だが
    外形（v, run_id, seq, time, type）が欠けた・不正な行を error にする。
    """
    if not hello_received:
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return StdoutLine("error", message=f"{label}から不正な JSON を受信しました")
        if not isinstance(event, dict):
            return StdoutLine("error", message=f"{label}から不正な JSON を受信しました")
        try:
            validate_hello(event, expected_run_id=run_id)
        except ValueError:
            return StdoutLine(
                "error", message=f"{label}の hello が不正です (受信 PID: {event.get('pid')})"
            )
        return StdoutLine("hello", event=event)
    try:
        event = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return StdoutLine("ignored", message="JSON ではない行")
    if not isinstance(event, dict):
        return StdoutLine("ignored", message="JSON object ではない行")
    try:
        validate_envelope(event, expected_run_id=run_id)
    except ValueError as error:
        return StdoutLine("error", event=event, message=f"{label}のプロトコルエラー: {error}")
    return StdoutLine("event", event=event)
