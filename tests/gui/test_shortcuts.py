"""ショートカット既定値・重複・JSON 入出力の検証。"""

from foam_cell_analysis.gui.shortcuts import DEFAULT_SHORTCUTS, ShortcutMap


def test_default_mapping_covers_data_preparation_actions(tmp_path):
    shortcuts = ShortcutMap(tmp_path / "keymap.json")
    assert shortcuts["usage_train"] == "Q"
    assert shortcuts["usage_val"] == "W"
    assert shortcuts["auto_triage"] == "Ctrl+D"
    assert shortcuts["filter_errors"] == "Alt+6"
    assert shortcuts["display_mode"] == "M"
    assert shortcuts["previous_image"] == "Left"
    assert shortcuts["next_image"] == "Right"
    assert "channel_prev" not in shortcuts.mapping
    assert "toggle_view" not in shortcuts.mapping
    assert set(shortcuts.mapping) == set(DEFAULT_SHORTCUTS)
    assert ("1–9", "分類") in shortcuts.hint_items()
    assert ("Enter", "連続振り分け") in shortcuts.hint_items()


def test_legacy_channel_keys_are_ignored_without_load_error(tmp_path):
    """旧 keymap.json のチャンネル操作は安全に読み飛ばす。"""
    path = tmp_path / "keymap.json"
    path.write_text(
        '{"channel_prev": "[", "channel_next": "]", "usage_train": "T"}',
        encoding="utf-8",
    )

    shortcuts = ShortcutMap(path)

    assert shortcuts["usage_train"] == "T"
    assert "channel_prev" not in shortcuts.mapping
    assert "channel_next" not in shortcuts.mapping


def test_hint_bar_tracks_custom_classification_keys(tmp_path):
    shortcuts = ShortcutMap(tmp_path / "keymap.json", mapping={"class_1": "F1"})

    assert ("F1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9", "分類") in shortcuts.hint_items()


def test_primary_hints_show_only_common_actions_and_keep_help_last(tmp_path):
    shortcuts = ShortcutMap(tmp_path / "keymap.json")

    assert shortcuts.hint_items() == [
        ("Q", "学習"),
        ("W", "検証"),
        ("E", "不採用"),
        ("R", "未振り分け"),
        ("1–9", "分類"),
        ("Z", "良"),
        ("X", "可"),
        ("C", "不良"),
        ("Space", "次の未振り分け"),
        ("Enter", "連続振り分け"),
        ("M", "表示形式"),
        ("?", "キー一覧"),
    ]


def test_hint_bar_normalizes_modifier_names(tmp_path):
    shortcuts = ShortcutMap(
        tmp_path / "keymap.json", mapping={"triage_view": "Control+Shift+Return"}
    )

    assert ("Ctrl+Shift+Enter", "連続振り分け") in shortcuts.hint_items()


def test_duplicate_detection_and_swap(tmp_path):
    shortcuts = ShortcutMap(tmp_path / "keymap.json")
    conflicts = shortcuts.duplicates("usage_train", "W")
    assert conflicts == ["usage_val"]
    shortcuts.assign("usage_train", "W", swap=True)
    assert shortcuts["usage_train"] == "W"
    assert shortcuts["usage_val"] == "Q"
    assert shortcuts.duplicates("usage_train", "W") == []


def test_save_load_export_and_import_round_trip(tmp_path):
    user_path = tmp_path / "config" / "keymap.json"
    exported = tmp_path / "exported.json"
    shortcuts = ShortcutMap(user_path)
    shortcuts.assign("usage_train", "T")
    shortcuts.save()
    assert ShortcutMap(user_path)["usage_train"] == "T"
    shortcuts.export_to(exported)
    restored = ShortcutMap(tmp_path / "elsewhere.json")
    restored.import_from(exported)
    assert restored.mapping == shortcuts.mapping
