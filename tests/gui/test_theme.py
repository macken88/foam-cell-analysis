"""共通テーマと見た目部品のテスト。"""

import re
from pathlib import Path

from foam_cell_analysis.gui.theme import Color, build_stylesheet


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
