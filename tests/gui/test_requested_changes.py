"""9 件の利用者要望に対する画面操作テスト。"""

from dataclasses import replace
from datetime import datetime
from time import perf_counter

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QHeaderView, QSizePolicy

from foam_cell_analysis.gui.context import AppContext, DisplayPreference, StatusBus
from foam_cell_analysis.gui.home_window import HomeWindow
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.keymap_dialog import KeymapWindow
from foam_cell_analysis.gui.modes.data_preparation import dialogs as data_dialogs
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
from foam_cell_analysis.gui.modes.inference.page import InferencePage
from foam_cell_analysis.gui.modes.training.page import TrainingPage
from foam_cell_analysis.gui.navigation import Navigator
from foam_cell_analysis.gui.theme import numeric_font
from foam_cell_analysis.gui.widgets.marks import DisplayToggle
from foam_cell_analysis.gui.window_manager import WindowManager
from foam_cell_analysis.services.mock.backend import MockBackend
from foam_cell_analysis.services.models import DatasetVersion


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


def test_display_preference_is_persisted_and_controls_are_two_choice(qapp, tmp_path):
    """表示設定が保存され、各画面のトグル名と描画が追従する。"""
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    preference = DisplayPreference(settings)
    toggle = DisplayToggle(preference)
    changes = []
    preference.changed.connect(changes.append)
    preference.set_value("二値マスク")
    assert settings.value("display/alternate") == "二値マスク"
    assert changes == ["二値マスク"]
    assert toggle.buttons["二値マスク"].text() == "二値マスク"
    ctx = make_context()
    ctx.display = DisplayPreference(settings)
    home = HomeWindow(ctx, WindowManager(ctx))
    home.display_actions["インスタンスラベル"].trigger()
    assert settings.value("display/alternate") == "インスタンスラベル"
    inference = InferencePage(ctx)
    assert len(inference.display_toggle.buttons) == 2
    data_page = DataPreparationPage(ctx)
    assert data_page.display_toggle.buttons["インスタンスラベル"].text() == "インスタンスラベル"
    data_page.display_actions["二値マスク"].trigger()
    assert inference.display_toggle.buttons["二値マスク"].text() == "二値マスク"
    assert data_page.display_toggle.buttons["二値マスク"].text() == "二値マスク"
    home.close()
    data_page.close()


def test_keymap_window_is_single_non_modal_and_refreshes(qapp):
    """キー一覧は非モーダルで一つだけ開き、割り当て変更を表示する。"""
    ctx = make_context()
    home = HomeWindow(ctx, WindowManager(ctx))
    settings_menu = home.menuBar().actions()[1].menu()
    action = next(item for item in settings_menu.actions() if item.text() == "キー割り当て一覧…")
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


def test_work_table_interactive_width_is_saved_and_restored(qapp, tmp_path):
    """作業表の各列幅を変更でき、次の画面生成で復元する。"""
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    page = DataPreparationPage(make_context())
    header = page.table.horizontalHeader()
    assert all(
        header.sectionResizeMode(column) == QHeaderView.ResizeMode.Interactive
        for column in range(page.model.columnCount())
    )
    header.resizeSection(0, 333)
    page.close()
    restored = DataPreparationPage(make_context())
    assert restored.table.columnWidth(0) == 333
    restored.close()


def test_work_table_first_launch_sizes_data_columns_and_uses_slack(qapp, tmp_path):
    """保存幅のない初回表示で内容幅を確保し、余白をフォルダ列に配る。"""
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    page = DataPreparationPage(make_context())
    page.resize(1440, 800)
    page.show()
    qapp.processEvents()
    header = page.table.horizontalHeader()
    for column in range(1, page.model.columnCount()):
        assert page.table.columnWidth(column) >= header.sectionSizeHint(column)
    assert page.table.columnWidth(3) >= 96
    assert sum(page.table.columnWidth(column) for column in range(page.model.columnCount())) >= (
        page.table.viewport().width()
    )
    page.close()


def test_work_table_ignores_extremely_narrow_saved_width(qapp, tmp_path):
    """24px 未満を含む保存幅は初回内容幅へ戻す。"""
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    page = DataPreparationPage(make_context())
    page.show()
    qapp.processEvents()
    page.table.horizontalHeader().resizeSection(1, 10)
    page._save_column_widths()
    page.close()
    restored = DataPreparationPage(make_context())
    restored.resize(1440, 800)
    restored.show()
    qapp.processEvents()
    assert restored.table.columnWidth(1) >= restored.table.horizontalHeader().sectionSizeHint(1)
    restored.close()


def test_selected_auto_triage_overwrites_assigned_items_and_undoes(
    qapp, qtbot, monkeypatch, tmp_path
):
    """行クリック、Ctrl+D、選択中、実行ボタンの経路で上書きし Ctrl+Z で戻す。"""
    monkeypatch.setattr(
        data_dialogs,
        "QSettings",
        lambda *_args: QSettings(str(tmp_path / "auto-triage.ini"), QSettings.Format.IniFormat),
    )
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
    assert "data.input_channels" not in training.fields
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


def test_label_combo_pairs_are_compact_and_display_channel_rows_are_removed(qapp):
    """メタデータはラベルとコンボを組にし、表示・チャンネル行を置かない。"""
    page = DataPreparationPage(make_context())
    assert len(page.edit_combos) == 4
    assert not hasattr(page, "display_combo")
    assert not hasattr(page, "channel_combo")
    for combo in page.edit_combos.values():
        assert combo.sizePolicy().horizontalPolicy().name == "Fixed"
    page.close()


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


def test_dataset_version_thumbnail_initial_view_is_fast_for_one_thousand_items(qapp):
    """1,000件の版履歴ウィンドウを0.5秒以内に初期表示する。"""
    backend = MockBackend()
    base = backend.get_working_items()[0]
    items = [
        replace(
            base,
            item_id=f"history_perf_{index:04d}",
            source_filename=f"history_{index:04d}.tif",
            seed=7000 + index,
            usage="train",
        )
        for index in range(1000)
    ]
    version = DatasetVersion(
        "train_perf",
        "train",
        "train_v003",
        datetime.now().astimezone(),
        [item.item_id for item in items],
        len(items),
        comment="性能確認",
        base_validation_version="val_v003",
    )
    backend.versions.append(version)
    backend._version_items[version.version] = items
    page = DatasetHistoryPage(make_context(backend))
    row = next(
        index
        for index, record in enumerate(page.model.versions)
        if record.version == version.version
    )
    page.table.selectRow(row)
    started = perf_counter()
    page.open_thumbnails()
    elapsed = perf_counter() - started
    window = page.thumbnail_windows[version.version]
    assert elapsed < 0.5
    assert window.models[version.version].rowCount() == 1000
    assert window.tabs.count() == 2
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
