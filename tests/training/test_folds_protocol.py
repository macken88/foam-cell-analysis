import json

import pytest

from foam_cell_analysis.services.models import DataItem
from foam_cell_analysis.training.folds import assign_folds
from foam_cell_analysis.training.protocol import read_events, validate_event
from foam_cell_analysis.training.seeds import derive


def _items():
    return [
        DataItem(
            f"i{group}{index}",
            "",
            f"folder{group}/image.png",
            ["A"],
            "A" if group % 2 else "B",
            "良",
            ["r1"],
            "r1",
        )
        for group in range(6)
        for index in range(2)
    ]


def test_folds_are_deterministic_grouped_and_balanced():
    items = _items()
    one = assign_folds(items, 3, 7)
    assert assign_folds(list(reversed(items)), 3, 7) == one
    assert len(set(one.values())) == 3
    by_group = {}
    for item in items:
        by_group.setdefault(item.source_folder, set()).add(one[item.item_id])
    assert all(len(folds) == 1 for folds in by_group.values())
    for label in ("A", "B"):
        counts = [
            sum(1 for item in items if one[item.item_id] == fold and item.classification == label)
            for fold in set(one.values())
        ]
        assert max(counts) - min(counts) <= 2
    assert derive(42, "fold", 1) == derive(42, "fold", 1)
    assert derive(42, "fold", 1) != derive(42, "fold", 2)


def test_jsonl_ignores_partial_tail_and_detects_seq_errors(tmp_path):
    event = {
        "v": 1,
        "run_id": "exp_0001/attempt_001",
        "seq": 1,
        "time": "2026-01-01T00:00:00Z",
        "type": "warning",
        "message": "ok",
    }
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event) + "\n{", encoding="utf-8")
    assert read_events(path) == [event]
    duplicate = dict(event, seq=1)
    path.write_text(
        "\n".join(json.dumps(row) for row in (event, duplicate)) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="重複"):
        read_events(path)
    missing = dict(event, seq=3)
    path.write_text(json.dumps(missing) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="欠落"):
        read_events(path)
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON が壊れ"):
        read_events(path)


def test_event_required_fields_are_checked():
    with pytest.raises(ValueError, match="必須項目"):
        validate_event({"v": 1, "run_id": "x", "seq": 1, "time": "now", "type": "warning"})
