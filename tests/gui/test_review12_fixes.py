"""Review 12 regression checks through visible GUI operations."""

from collections import Counter

from PySide6.QtCore import QEvent, QItemSelectionModel, QPoint, Qt
from PySide6.QtTest import QTest

from foam_cell_analysis.gui.navigation import ModeId, PageId
from foam_cell_analysis.gui.widgets.form import FormSection


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


def _choose_combo_item(qtbot, combo, value):
    combo.showPopup()
    qtbot.wait(10)
    index = combo.findData(value)
    rect = combo.view().visualRect(combo.model().index(index, 0))
    QTest.mouseClick(combo.view().viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
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


def test_current_row_and_scroll_survive_activation_edit_undo_redo(shell, qapp, qtbot):
    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    window = shell.manager.window(ModeId.DATA_PREPARATION)
    rows = [18, 19, 20]
    _select_rows_by_click(page, qtbot, rows)
    selected = [page.model.visible_items()[row].item_id for row in rows]
    page.table.selectionModel().setCurrentIndex(
        page.model.index(rows[-1], 0), QItemSelectionModel.SelectionFlag.NoUpdate
    )
    current_id = selected[-1]
    page.table.verticalScrollBar().setValue(12)
    scroll_value = page.table.verticalScrollBar().value()
    window.raise_()
    window.activateWindow()
    QTest.qWait(20)
    qapp.sendEvent(window, QEvent(QEvent.Type.WindowDeactivate))
    qapp.sendEvent(window, QEvent(QEvent.Type.WindowActivate))
    qapp.processEvents()
    assert [
        page.model.item_at(index.row()).item_id
        for index in page.table.selectionModel().selectedRows()
    ] == selected
    assert page.model.item_at(page.table.currentIndex().row()).item_id == current_id
    assert page.table.verticalScrollBar().value() == scroll_value

    item = next(item for item in page.items if item.item_id == current_id)
    previous = item.quality
    target = "可" if previous != "可" else "良"
    page.quality_combo.setCurrentIndex(page.quality_combo.findData(target))
    qapp.processEvents()
    for expected_quality in (target, previous, target):
        assert item.quality == expected_quality
        assert [
            page.model.item_at(index.row()).item_id
            for index in page.table.selectionModel().selectedRows()
        ] == selected
        assert page.model.item_at(page.table.currentIndex().row()).item_id == current_id
        assert page.table.verticalScrollBar().value() == scroll_value
        if expected_quality == target and item.quality == target:
            if page.undo_stack.canUndo():
                page.undo_stack.undo()
                qapp.processEvents()
                expected_quality = previous
                assert item.quality == previous
                assert page.model.item_at(page.table.currentIndex().row()).item_id == current_id
                page.undo_stack.redo()
                qapp.processEvents()
                break


def test_narrow_window_shows_default_columns_and_wraps_preview_controls(shell, qapp):
    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    page.resize(900, 650)
    page.show()
    qapp.processEvents()
    table_width = page.table.viewport().width()
    default_width = sum(page.table.columnWidth(column) for column in range(5))
    assert default_width <= table_width
    assert page.table.horizontalScrollBar().maximum() == 0
    controls = [page.usage_combo, page.class_combo, page.quality_combo, page.mask_combo]
    top_y = sorted(
        {combo.mapTo(page.preview_panel, combo.rect().topLeft()).y() for combo in controls}
    )
    assert len(top_y) == 2


def test_training_section_headings_have_prominent_theme_fonts(shell, qapp):
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    sections = [widget for widget, _column in page._form_widgets if isinstance(widget, FormSection)]
    assert sections
    for section in sections:
        assert section.heading_label.font().pointSizeF() >= qapp.font().pointSizeF() + 2, (
            section.title(),
            section.heading_label.font().toString(),
        )
        assert section.heading_label.font().bold(), section.title()
        assert section.heading_rule.isVisible()


def test_training_columns_stack_independently_and_dataset_note_stays_with_cv(shell, qapp):
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    page.preview_action.setChecked(False)
    page._update_form_columns(1300)
    left_widgets = [page.left_column.itemAt(i).widget() for i in range(page.left_column.count())]
    right_widgets = [page.right_column.itemAt(i).widget() for i in range(page.right_column.count())]
    assert left_widgets and right_widgets
    assert left_widgets[-1] is page.fields["training.epochs"].parentWidget()
    cv_section = page.fields["data.cv.n_folds"].parentWidget()
    assert page.dataset_note.parentWidget() is cv_section
    assert len(left_widgets) != len(right_widgets)


def test_training_augmentation_button_is_compact_and_next_to_profile(shell, qapp):
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    profile = page.fields["augmentation.profile"]
    qapp.processEvents()
    assert page.profile_edit_button.text() == "データ拡張を設定…"
    assert page.profile_edit_button.width() <= page.profile_edit_button.sizeHint().width() + 8
    assert profile.geometry().center().y() == page.profile_edit_button.geometry().center().y()
    assert not hasattr(page, "profile_preview_button")
