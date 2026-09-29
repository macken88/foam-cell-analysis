"""9 件の利用者要望に対する画面操作テスト。"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSizePolicy

from foam_cell_analysis.gui.context import AppContext, StatusBus
from foam_cell_analysis.gui.home_window import HomeWindow
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.keymap_dialog import KeymapWindow
from foam_cell_analysis.gui.modes.data_preparation.dialogs import (
    AutoTriageDialog,
    ContinuousTriageDialog,
    DatasetFinalizeDialog,
    ImportDialog,
)
from foam_cell_analysis.gui.modes.data_preparation.page import (
    DataPreparationPage,
    DatasetHistoryPage,
)
from foam_cell_analysis.gui.modes.training.page import TrainingPage
from foam_cell_analysis.gui.navigation import Navigator
from foam_cell_analysis.gui.theme import numeric_font
from foam_cell_analysis.gui.window_manager import WindowManager
from foam_cell_analysis.services.mock.backend import MockBackend


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


def click_row(qapp, page, item_id):
    """対象画像の行を実際の表クリックで選択する。"""
    row = next(
        index for index, item in enumerate(page.model.visible_items()) if item.item_id == item_id
    )
    page.table.show()
    qapp.processEvents()
    rect = page.table.visualRect(page.model.index(row, 0))
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    qapp.processEvents()


def test_keymap_window_is_single_non_modal_and_refreshes(qapp):
    """キー一覧は非モーダルで一つだけ開き、割り当て変更を表示する。"""
    ctx = make_context()
    home = HomeWindow(ctx, WindowManager(ctx))
    settings_menu = home.menuBar().actions()[1].menu()
    action = next(item for item in settings_menu.actions() if item.text() == "キー割り当て…")
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


def test_arrow_shortcuts_move_rows_and_ignore_search_focus(qapp):
    """左右キーで画像行を移し、検索欄では文字入力を優先する。"""
    page = DataPreparationPage(make_context())
    page.show()
    visible = page.model.visible_items()
    click_row(qapp, page, visible[1].item_id)
    QTest.keyClick(page.table, Qt.Key.Key_Left)
    assert page._selected_ids == [visible[0].item_id]
    page.search.setFocus()
    QTest.keyClick(page.search, Qt.Key.Key_Right)
    assert page._selected_ids == [visible[0].item_id]
    page.class_combo.showPopup()
    QTest.keyClick(page.class_combo, Qt.Key.Key_Left)
    assert page._selected_ids == [visible[0].item_id]
    page.class_combo.hidePopup()
    page.close()


def test_selected_auto_triage_overwrites_assigned_items_and_undoes(
    qapp, qtbot, monkeypatch, tmp_path
):
    """行クリック、Ctrl+D、選択中、実行ボタンの経路で上書きし Ctrl+Z で戻す。"""
    backend = MockBackend()
    ctx = make_context(backend)
    page = DataPreparationPage(ctx)
    page.show()
    item = next(item for item in backend.get_working_items() if item.usage == "train")
    click_row(qapp, page, item.item_id)

    def choose_and_run():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, AutoTriageDialog)
        QTest.mouseClick(dialog.target_selected, Qt.MouseButton.LeftButton)
        dialog.ratio.setValue(100)
        row = next(
            index
            for index in range(dialog.preview.rowCount())
            if dialog.preview.item(index, 0).text() == (item.classification or "未設定")
        )
        assert dialog.preview.item(row, 1).text().endswith("→ 0")
        assert dialog.preview.item(row, 2).text().endswith("→ 1")
        QTest.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(0, choose_and_run)
    QTest.keyClick(page.table, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
    assert item.usage == "val"
    page.undo_stack.undo()
    assert item.usage == "train"
    page.close()


def test_auto_triage_numeric_controls_and_preview_order(qapp):
    """数値欄を内容幅に保ち、分類定義順と数値セル書式を使う。"""
    backend = MockBackend()
    dialog = AutoTriageDialog(None, backend.get_working_items(), backend=backend)
    assert dialog.ratio.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    assert dialog.seed.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    classes = [dialog.preview.item(row, 0).text() for row in range(dialog.preview.rowCount())]
    expected = [name for name in backend.classifications if name in classes]
    if "未設定" in classes:
        expected.append("未設定")
    assert classes == expected
    if dialog.preview.rowCount():
        cell = dialog.preview.item(0, 1)
        assert cell.textAlignment() & Qt.AlignmentFlag.AlignRight
        assert cell.font().family() == numeric_font().family()
    dialog.close()


def test_single_channel_gui_and_continuous_triage_mouse_actions(qapp, qtbot):
    """取り込み画面は単一画像フォルダで、連続振り分けに用途ボタンを置く。"""
    dialog = ImportDialog()
    assert not hasattr(dialog, "channel_rows")
    assert dialog.image_edit is not None
    ctx = make_context()
    training = TrainingPage(ctx)
    assert training.fields["data.input_channels"].text() == "A（単一チャンネル）"
    assert training._collect_config()["data"]["input_channels"] == ["A"]
    training.close()
    unassigned = next(
        item for item in ctx.backend.get_working_items() if item.usage == "unassigned"
    )
    triage = ContinuousTriageDialog(
        DataPreparationPage(ctx),
        [unassigned],
        lambda item_id, **changes: ctx.backend.update_item("all", item_id, **changes),
        ctx.backend,
        ctx.shortcuts,
        ctx.display,
    )
    triage.show()
    qapp.processEvents()
    assert set(triage.usage_buttons) == {"train", "val", "excluded", "unassigned"}
    initial_usage = unassigned.usage
    QTest.mouseClick(triage.usage_buttons["train"][0], Qt.MouseButton.LeftButton)
    assert unassigned.usage != initial_usage
    QTest.mouseClick(triage.target_filtered, Qt.MouseButton.LeftButton)
    QTest.keyClick(triage, Qt.Key.Key_R)
    assert (
        next(
            item for item in ctx.backend.get_working_items() if item.item_id == unassigned.item_id
        ).usage
        == "unassigned"
    )
    triage.close()


def test_dataset_history_opens_read_only_thumbnail_window(qapp):
    """版履歴の選択とボタン操作で、版の全画像を非モーダル表示する。"""
    ctx = make_context()
    page = DatasetHistoryPage(ctx)
    page.show()
    page.table.selectRow(0)
    qapp.processEvents()
    QTest.mouseClick(page.thumbnail_button, Qt.MouseButton.LeftButton)
    version = page.model.versions[0]
    window = page.thumbnail_windows[version.version]
    assert not window.isModal()
    model = window.models[version.version]
    assert model.rowCount() == version.n_images
    if version.purpose == "train" and version.base_validation_version:
        assert window.tabs.count() == 2
        assert version.base_validation_version in window.tabs.tabText(1)
    index = page.model.index(0, 0)
    QTest.mouseDClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.table.visualRect(index).center(),
    )
    assert page.thumbnail_windows[version.version] is window
    window.close()
    page.close()


def test_finalize_reason_button_filters_error_rows(qapp):
    """確定不可の理由ボタンからエラー行だけに絞り込める。"""
    backend = MockBackend()
    item = next(item for item in backend.get_working_items() if item.usage == "train")
    backend.update_item("all", item.item_id, classification=None)
    page = DataPreparationPage(make_context(backend))
    page.show()
    assert not page.finalize_error_button.isHidden()
    QTest.mouseClick(page.finalize_error_button, Qt.MouseButton.LeftButton)
    assert page.model.errors_only
    assert all(
        item.item_id in {error.item_id for error in backend.validate_items().errors}
        for item in page.model.visible_items()
    )
    dialog = DatasetFinalizeDialog(None, backend)
    assert not dialog.error_summary_button.isHidden()
    dialog.close()
    page.close()


def test_display_toggle_ignores_arrow_keys_in_continuous_triage(qapp):
    """連続振り分けで ← → は画像移動だけに効き、表示切り替えは変わらない。"""
    page = DataPreparationPage(make_context())
    dialog = ContinuousTriageDialog(
        page,
        page.model.visible_items(),
        lambda *_args, **_kwargs: None,
        page.ctx.backend,
        page.ctx.shortcuts,
    )
    dialog.show()
    qapp.processEvents()
    toggle = dialog.display_toggle
    before = toggle.is_alternate
    QTest.mouseClick(toggle.raw_button, Qt.MouseButton.LeftButton)
    assert not toggle.raw_button.hasFocus()
    QTest.keyClick(dialog, Qt.Key.Key_Right)
    QTest.keyClick(dialog, Qt.Key.Key_Left)
    assert toggle.is_alternate is False
    assert toggle.raw_button.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert before is True
    dialog.close()
    page.close()
