"""共通テーマと見た目部品のテスト。"""

import re
from pathlib import Path

from PySide6.QtWidgets import QLabel

from foam_cell_analysis.gui.theme import Color, build_stylesheet
from foam_cell_analysis.gui.widgets.marks import CountChip, KeyHintBar, StatusTag, UsageTag


def test_build_stylesheet_uses_primary_theme_colors(qapp) -> None:
    """QSS に主要な意味色と横罫線色が含まれる。"""
    stylesheet = build_stylesheet()

    for color in (Color.STAGE, Color.SLIDE, Color.GRAPHITE, Color.SLATE, Color.RULE_SOFT):
        assert color in stylesheet


def test_source_hex_colors_are_centralized_in_theme() -> None:
    """theme.py 以外に 16 進色を直接記述しない。"""
    source_root = Path(__file__).parents[2] / "src"
    hex_color = re.compile(r"['\"]#[0-9A-Fa-f]{3,8}['\"]")

    offenders = [
        str(path.relative_to(source_root))
        for path in source_root.rglob("*.py")
        if path.name != "theme.py" and hex_color.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_mark_widgets_build_and_display_expected_labels(qtbot) -> None:
    """タグ、チップ、キー操作ヒントを生成して文字列を表示する。"""
    usage = UsageTag("train")
    status = StatusTag("完了")
    count = CountChip("学習", 489, "train")
    hints = KeyHintBar([("Q", "学習"), ("W", "検証")])

    for widget in (usage, status, count, hints):
        qtbot.addWidget(widget)
        widget.show()

    assert usage.text() == "学習"
    assert status.text() == "完了"
    assert count.name_label.text() == "学習"
    assert count.count_label.text() == "489"
    assert [label.text() for label in hints.findChildren(QLabel)] == ["Q", "学習", "W", "検証"]


def test_key_hint_bar_keeps_labels_whole_and_help_visible_when_narrow(qtbot, qapp):
    hints = KeyHintBar(
        [("Q", "学習"), ("Space", "次の未振り分け"), ("Enter", "連続振り分け"), ("?", "キー一覧")]
    )
    qtbot.addWidget(hints)
    hints.resize(280, hints.sizeHint().height())
    hints.show()
    qapp.processEvents()

    assert hints._groups[-1].isVisible()
    assert hints._groups[0].isVisible()
    assert not hints._groups[2].isVisible()
    for group in hints._groups:
        if group.isVisible():
            keycap, label = group.findChildren(QLabel)
            assert keycap.width() >= keycap.sizeHint().width()
            assert label.width() >= label.sizeHint().width()
