"""Review 12 regression checks through visible GUI operations."""

from collections import Counter

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from foam_cell_analysis.gui.navigation import ModeId, PageId


def _menus(window):
    pending = [action.menu() for action in window.menuBar().actions() if action.menu()]
    while pending:
        menu = pending.pop()
        yield menu
        pending.extend(action.menu() for action in menu.actions() if action.menu())


def _select_rows_by_click(page, qtbot, rows):
    page.table.clearSelection()
    for position, row in enumerate(rows):
        page.table.scrollTo(page.model.index(row, 0))
        qtbot.wait(10)
        rect = page.table.visualRect(page.model.index(row, 0))
        modifier = (
            Qt.KeyboardModifier.NoModifier if position == 0 else Qt.KeyboardModifier.ControlModifier
        )
        QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, modifier, rect.center())
    page.table.setFocus()
    qtbot.wait(10)


def test_all_mode_menus_have_unique_sibling_item_names(shell, qapp):
    mode_for_page = {
        PageId.DATA_PREPARATION: ModeId.DATA_PREPARATION,
        PageId.DATASET_HISTORY: ModeId.DATA_PREPARATION,
        PageId.TRAINING: ModeId.TRAINING,
        PageId.TRAINING_QUEUE: ModeId.TRAINING,
        PageId.EXPERIMENTS: ModeId.TRAINING,
        PageId.CANDIDATES: ModeId.COMPARISON,
        PageId.MASK_COMPARISON: ModeId.COMPARISON,
        PageId.RELEASED_MODELS: ModeId.COMPARISON,
        PageId.INFERENCE: ModeId.INFERENCE,
    }
    violations = []
    for page_id, mode_id in mode_for_page.items():
        shell.navigate(page_id)
        qapp.processEvents()
        window = shell.manager.window(mode_id)
        for menu in _menus(window):
            names = [
                action.text().split("\t", 1)[0]
                for action in menu.actions()
                if not action.isSeparator()
            ]
            duplicates = [name for name, count in Counter(names).items() if count > 1]
            if duplicates:
                violations.append((page_id, menu.title(), duplicates))
    assert violations == []


def test_changed_filter_menu_has_one_action_and_lower_entry_is_safe(shell, qapp):
    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    window = shell.manager.window(ModeId.DATA_PREPARATION)
    filter_menu_action = next(
        action
        for menu in _menus(window)
        for action in menu.actions()
        if action.menu() and action.text().split("\t", 1)[0] == "絞り込み"
    )
    changed = [
        action
        for action in filter_menu_action.menu().actions()
        if action.text().split("\t", 1)[0] == "変更あり"
    ]
    assert len(changed) == 1
    before = page.model.changed_only
    changed[0].trigger()
    qapp.processEvents()
    assert page.model.changed_only is not before


def test_undo_redo_shortcuts_only_apply_to_table_or_preview(shell, qapp, qtbot):
    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    window = shell.manager.window(ModeId.DATA_PREPARATION)
    window.show()
    window.activateWindow()
    window.raise_()
    qtbot.waitUntil(window.isActiveWindow)
    item = page.items[0]
    old_quality = item.quality
    new_quality = "可" if old_quality != "可" else "良"
    page._change([item.item_id], quality=new_quality)
    stack_index = page.undo_stack.index()

    page.source_combo.setFocus()
    qtbot.waitUntil(page.source_combo.hasFocus)
    QTest.keyClick(page.source_combo, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert page.undo_stack.index() == stack_index
    assert item.quality == new_quality

    page.class_filter.setFocus()
    qtbot.waitUntil(page.class_filter.hasFocus)
    QTest.keyClick(page.class_filter, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert page.undo_stack.index() == stack_index
    assert item.quality == new_quality

    page.table.setFocus()
    QTest.keyClick(page.table, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert page.undo_stack.index() == stack_index - 1
    assert item.quality == old_quality
    QTest.keyClick(page.table, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert page.undo_stack.index() == stack_index


def test_non_dialog_page_shortcuts_do_not_override_text_input(shell, qapp, qtbot):
    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    page.search.setText("needle")
    page.search.setFocus()
    before = page.table.selectionModel().selectedRows()
    QTest.keyClick(page.search, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert page.search.selectionLength() == len("needle")
    assert page.table.selectionModel().selectedRows() == before

    invocations = []
    page.auto_triage = lambda: invocations.append("auto")
    page.export_excel = lambda: invocations.append("export")
    page.finalize = lambda: invocations.append("finalize")
    page.search.setFocus()
    QTest.keyClick(page.search, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(page.search, Qt.Key.Key_E, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(page.search, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert invocations == ["auto", "export", "finalize"]


def test_filter_excluding_all_selected_rows_leaves_zero_selection(shell, qapp, qtbot):
    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    selected = page.model.visible_items()[:3]
    classes = page.ctx.backend.classifications
    selected_class = classes[0]
    excluded_class = classes[1]
    for item in selected:
        shell.ctx.backend.update_item("all", item.item_id, classification=selected_class)
    page.refresh()
    _select_rows_by_click(
        page,
        qtbot,
        [
            next(
                row
                for row, item in enumerate(page.model.visible_items())
                if item.item_id == target.item_id
            )
            for target in selected
        ],
    )
    assert len(page.table.selectionModel().selectedRows()) == 3
    page.class_filter.showPopup()
    qtbot.wait(10)
    excluded_index = page.class_filter.findData(excluded_class)
    rect = page.class_filter.view().visualRect(page.class_filter.model().index(excluded_index, 0))
    QTest.mouseClick(
        page.class_filter.view().viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(rect.left() + 10, rect.center().y()),
    )
    qapp.processEvents()
    assert page.model.classifications == {excluded_class}
    assert page.table.selectionModel().selectedRows() == []
    assert page.selection_note.text() == "選択中の 0 件を変更"
    assert page.preview_meta.text() == "画像を選択してください"
    assert not page.usage_combo.isEnabled()


def test_usage_change_removing_selected_rows_leaves_zero_selection(shell, qapp, qtbot):
    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    train_rows = [
        row for row, item in enumerate(page.model.visible_items()) if item.usage == "train"
    ][:3]
    _select_rows_by_click(page, qtbot, train_rows)
    QTest.mouseClick(page.chips["val"], Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert page.table.selectionModel().selectedRows() == []
    assert page.selection_note.text() == "選択中の 0 件を変更"
