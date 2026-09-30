"""ショートカット既定値・重複・JSON 入出力の検証。"""

from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from foam_cell_analysis.gui.context import AppContext, StatusBus
from foam_cell_analysis.gui.home_window import HomeWindow
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.keymap_dialog import KeymapDialog, KeymapWindow, ShortcutKeyEdit
from foam_cell_analysis.gui.modes.data_preparation.dialogs import ContinuousTriageDialog
from foam_cell_analysis.gui.navigation import Navigator, PageId
from foam_cell_analysis.gui.shortcuts import DEFAULT_SHORTCUTS, ShortcutMap
from foam_cell_analysis.gui.window_manager import WindowManager
from foam_cell_analysis.services.mock.backend import MockBackend


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


def _click_menu_action(qapp, button, action):
    """ボタンからメニューを開き、指定項目の表示位置をクリックする。"""
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    menu = button.menu()
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
    qapp.processEvents()


def test_keymap_cancel_discards_staged_changes(shell, qapp):
    shortcuts = shell.ctx.shortcuts
    original = shortcuts["usage_train"]
    dialog = KeymapDialog(None, shortcuts)
    row = dialog.actions.index("usage_train")
    dialog.table.setCurrentCell(row, 2)
    dialog.table.setFocus()
    QTest.keyClick(dialog.table, Qt.Key.Key_F2)
    editor = dialog.table.focusWidget()
    assert isinstance(editor, ShortcutKeyEdit)
    editor.setFocus()
    QTest.keyClick(editor, Qt.Key.Key_T)
    QTest.keyClick(editor, Qt.Key.Key_Return)
    qapp.processEvents()
    assert dialog.shortcuts["usage_train"] == "T"
    dialog.reject()
    assert shortcuts["usage_train"] == original


def test_keymap_editor_captures_modifier_sequence(qapp, shell):
    dialog = KeymapDialog(None, shell.ctx.shortcuts)
    row = dialog.actions.index("display_mode")
    dialog.table.setCurrentCell(row, 2)
    dialog.table.setFocus()
    QTest.keyClick(dialog.table, Qt.Key.Key_F2)
    editor = dialog.table.focusWidget()
    assert isinstance(editor, ShortcutKeyEdit)
    editor.setFocus()
    QTest.keyClick(
        editor,
        Qt.Key.Key_J,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier,
    )
    QTest.keyClick(editor, Qt.Key.Key_Return)
    qapp.processEvents()
    assert dialog.table.item(row, 2).text() == "Ctrl+Alt+J"
    assert dialog.shortcuts["display_mode"] == "Ctrl+Alt+J"
    dialog.reject()


def test_saving_keymap_by_mouse_updates_open_pages_and_triage(qapp, shell):
    data_page = shell.page(PageId.DATA_PREPARATION)
    compare_page = shell.page(PageId.MASK_COMPARISON)
    inference_page = shell.page(PageId.INFERENCE)
    dialog_results = []

    def edit_and_save(dialog):
        dialog.show()
        qapp.processEvents()
        for action_name, key, modifiers in (
            ("usage_train", Qt.Key.Key_T, Qt.KeyboardModifier.AltModifier),
            ("display_mode", Qt.Key.Key_U, Qt.KeyboardModifier.NoModifier),
        ):
            row = dialog.actions.index(action_name)
            cell = dialog.table.item(row, 2)
            dialog.table.scrollToItem(cell)
            qapp.processEvents()
            rect = dialog.table.visualItemRect(cell)
            QTest.mouseClick(dialog.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
            QTest.keyClick(dialog.table, Qt.Key.Key_F2)
            qapp.processEvents()
            editor = dialog.table.focusWidget()
            assert isinstance(editor, ShortcutKeyEdit)
            QTest.keyClick(editor, key, modifiers)
            qapp.processEvents()
        QTest.mouseClick(
            dialog.buttons.button(QDialogButtonBox.StandardButton.Save),
            Qt.MouseButton.LeftButton,
        )
        qapp.processEvents()
        dialog_results.append(dialog.result())
        return dialog.result()

    action = next(
        action
        for action in data_page.other_button.menu().actions()
        if action.text() == "キー割り当て一覧"
    )
    with patch.object(KeymapDialog, "exec", edit_and_save):
        _click_menu_action(qapp, data_page.other_button, action)
        shell.ctx.keymap_window.change_button.click()
    assert dialog_results == [QDialog.DialogCode.Accepted]
    assert shell.ctx.shortcuts["usage_train"] == "Alt+T"
    assert shell.ctx.shortcuts["display_mode"] == "U"
    assert (
        shell.ctx.keymap_window.table.item(
            shell.ctx.keymap_window.actions.index("display_mode"), 2
        ).text()
        == "U"
    )
    assert compare_page.shortcuts["display_mode"] == "U"
    assert inference_page.shortcuts["usage_train"] == "Alt+T"
    item = next(item for item in data_page.items if item.usage == "unassigned")
    dialog = ContinuousTriageDialog(
        data_page,
        [item],
        lambda item_id, **changes: shell.ctx.backend.update_item("all", item_id, **changes),
        shell.ctx.backend,
        shell.ctx.shortcuts,
    )
    assert "Alt+T" in dialog.usage_buttons["train"][0].text()
    dialog.show()
    dialog.setFocus()
    QTest.keyClick(dialog, Qt.Key.Key_T, Qt.KeyboardModifier.AltModifier)
    qapp.processEvents()
    assert item.usage == "train"
    dialog.close()


def make_context(backend=None, shortcuts=None):
    """テスト用の独立したアプリケーションコンテキストを作る。"""
    from foam_cell_analysis.gui.shortcuts import ShortcutMap

    return AppContext(
        backend or MockBackend(),
        Navigator(),
        JobManager(),
        StatusBus(),
        shortcuts or ShortcutMap(mapping={}),
    )


def test_keymap_window_is_single_non_modal_and_refreshes(qapp):
    """キー一覧は非モーダルで一つだけ開き、割り当て変更を表示する。"""
    ctx = make_context()
    home = HomeWindow(ctx, WindowManager(ctx))
    settings_menu = home.menuBar().actions()[1].menu()
    action = next(item for item in settings_menu.actions() if item.text() == "キー割り当て")
    action.trigger()
    first = ctx.keymap_window
    action.trigger()
    second = ctx.keymap_window
    assert isinstance(first, KeymapWindow)
    assert first is second
    assert not first.isModal()
    assert first.table.columnCount() == 3
    ctx.shortcuts.assign("next_image", "N")
    ctx.shortcuts.changed.emit()
    row = first.actions.index("next_image")
    assert first.table.item(row, 2).text() == "N"
    first.close()
    home.close()
