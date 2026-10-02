"""子プロセスの stdout 解釈（比較・推論設計 7.4）と共通プロトコルの補助関数。"""

import json

import pytest

from foam_cell_analysis.jobs.protocol import (
    classify_seq,
    parse_stdout_line,
    read_jsonl_events,
)

RUN_ID = "exp_001/attempt_001"


def _line(value) -> bytes:
    return json.dumps(value).encode("utf-8")


def _hello(**changes):
    return {"v": 1, "run_id": RUN_ID, "time": "2026-09-30T10:00:00+09:00", "type": "hello",
            "pid": 1234, **changes}  # fmt: skip


def _event(**changes):
    return {"v": 1, "run_id": RUN_ID, "seq": 1, "time": 1.5, "type": "started", **changes}


def test_hello_is_accepted_before_hello():
    parsed = parse_stdout_line(_line(_hello()), run_id=RUN_ID, hello_received=False)
    assert parsed.kind == "hello"
    assert parsed.event["pid"] == 1234


@pytest.mark.parametrize(
    "line",
    [
        b"Loading weights...",
        b"",
        _line([1, 2]),
        _line(_event()),  # hello より前の通常イベント
        _line(_hello(run_id="exp_002/attempt_001")),
        _line(_hello(pid=0)),
        _line(_hello(seq=1)),
    ],
)
def test_lines_before_hello_are_strict(line):
    parsed = parse_stdout_line(line, run_id=RUN_ID, hello_received=False, label="学習プロセス")
    assert parsed.kind == "error"
    assert parsed.message.startswith("学習プロセス")


@pytest.mark.parametrize("line", [b"Downloading model...", b"", b"42", b"[1, 2]", b"\xff\xfe"])
def test_non_json_lines_after_hello_are_ignored(line):
    parsed = parse_stdout_line(line, run_id=RUN_ID, hello_received=True)
    assert parsed.kind == "ignored"


@pytest.mark.parametrize(
    "event",
    [
        {k: v for k, v in _event().items() if k != "seq"},
        {k: v for k, v in _event().items() if k != "time"},
        {"type": "started"},
        _event(v=2),
        _event(seq=0),
        _event(time="not-a-time"),
        _event(run_id="exp_002/attempt_001"),
        _hello(),  # hello の 2 回目は seq がないので外形不足
    ],
)
def test_json_without_envelope_after_hello_is_protocol_error(event):
    parsed = parse_stdout_line(_line(event), run_id=RUN_ID, hello_received=True)
    assert parsed.kind == "error"
    assert "プロトコルエラー" in parsed.message


def test_valid_event_after_hello_is_passed_without_type_check():
    # 型ごとの必須項目は各 runner / backend が検証する
    parsed = parse_stdout_line(_line(_event(type="image_done")), run_id=RUN_ID, hello_received=True)
    assert parsed.kind == "event"
    assert parsed.event["type"] == "image_done"


def test_classify_seq():
    assert classify_seq(3, 4) == "next"
    assert classify_seq(3, 3) == "duplicate"
    assert classify_seq(3, 1) == "duplicate"
    assert classify_seq(3, 5) == "gap"


def test_read_jsonl_events_ignores_truncated_last_line_and_detects_gaps(tmp_path):
    path = tmp_path / "events.jsonl"
    first = json.dumps(_event(seq=1))
    second = json.dumps(_event(seq=2))
    path.write_text(f'{first}\n{second}\n{{"v":1,', encoding="utf-8")
    assert [event["seq"] for event in read_jsonl_events(path)] == [1, 2]

    path.write_text(f"{first}\n{json.dumps(_event(seq=3))}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="欠落"):
        read_jsonl_events(path)
