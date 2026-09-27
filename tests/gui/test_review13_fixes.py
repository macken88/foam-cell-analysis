"""Review 13 regression checks through visible training settings operations."""

import copy

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from foam_cell_analysis.gui.navigation import ModeId, PageId
from foam_cell_analysis.gui.widgets.form import CollapsibleSection


def _enqueue_waiting_row(backend, *, model_type="mask_rcnn", epochs=17):
    config = backend.default_experiment_config(model_type)
    config["experiment"]["id"] = backend.next_experiment_id()
    config["training"]["epochs"] = epochs
    return backend.add_training_queue_item(config)


def _open_queue_editor(shell, queued, qapp):
    shell.navigate(PageId.TRAINING_QUEUE)
    qapp.processEvents()
    queue = shell.page(PageId.TRAINING_QUEUE)
    queue.table.selectRow(queue.model.row_for_id(queued.experiment_id))
    queue.queue_actions["edit"].trigger()
    qapp.processEvents()
    return shell.page(PageId.TRAINING)


@pytest.mark.parametrize("finish", ["save", "cancel"])
def test_queue_edit_finish_restores_previous_form_and_view(shell, qapp, qtbot, finish):
    backend = shell.ctx.backend
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    window = shell.manager.window(ModeId.TRAINING)
    window.resize(1050, 720)
    window.show()
    qapp.processEvents()

    page.model_type.setCurrentIndex(page.model_type.findData("cellpose"))
    page.fields["training.epochs"].setValue(33)
    page.description_edit.setText("編集中の新規設定")
    page.preview_action.setChecked(True)
    qapp.processEvents()
    maximum = page.scroll.verticalScrollBar().maximum()
    before_scroll = min(80, maximum)
    page.scroll.verticalScrollBar().setValue(before_scroll)
    before_yaml = page.preview_action.isChecked()
    before_config = copy.deepcopy(page._collect_config())
    before_actions = {
        name: (action.isEnabled(), action.toolTip())
        for name, action in page.training_actions.items()
    }

    queued = _enqueue_waiting_row(backend)
    page = _open_queue_editor(shell, queued, qapp)
    page.fields["training.epochs"].setValue(21)
    page.preview_action.setChecked(not before_yaml)
    page.scroll.verticalScrollBar().setValue(0)
    if finish == "save":
        QTest.mouseClick(page.save_button, Qt.MouseButton.LeftButton)
        assert (
            backend.get_experiment(queued.experiment_id).config.values["training"]["epochs"] == 21
        )
    else:
        QTest.mouseClick(page.cancel_queue_edit_button, Qt.MouseButton.LeftButton)
        assert (
            backend.get_experiment(queued.experiment_id).config.values["training"]["epochs"] == 17
        )
    qapp.processEvents()

    shell.navigate(PageId.TRAINING)
    qapp.processEvents()
    assert page.fields["training.epochs"].value() == 33
    assert page.description_edit.text() == "編集中の新規設定"
    assert page.model_type.currentData() == "cellpose"
    assert page.preview_action.isChecked() is before_yaml
    assert page.scroll.verticalScrollBar().value() == before_scroll
    assert page.queue_edit_banner.isHidden()
    assert page.validate_button.isVisible()
    assert page.queue_button.isVisible()
    assert page.start_button.isVisible()
    assert page.cancel_queue_edit_button.isHidden()
    assert {
        name: (action.isEnabled(), action.toolTip())
        for name, action in page.training_actions.items()
    } == before_actions
    assert page._collect_config()["training"]["epochs"] == before_config["training"]["epochs"]


def test_queue_edit_disables_actions_and_execution_methods(shell, qapp, monkeypatch):
    backend = shell.ctx.backend
    queued = _enqueue_waiting_row(backend)
    page = _open_queue_editor(shell, queued, qapp)
    waiting_before = len(backend.list_training_queue())
    calls = []
    monkeypatch.setattr(
        backend, "validate_experiment_config", lambda _config: calls.append("validate") or []
    )

    for name in ("validate", "queue", "start"):
        action = page.training_actions[name]
        assert not action.isEnabled()
        assert "キューの行を編集中" in action.toolTip()
        action.trigger()
    assert page.validate_config() == []
    assert page.enqueue_config() is None
    assert page.start_training(confirm=False) is None
    assert calls == []
    assert len(backend.list_training_queue()) == waiting_before
    assert not shell.ctx.jobs.training_jobs


def test_reflow_preserves_right_column_focus_cursor_and_yaml_highlight(shell, qapp, qtbot):
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    window = shell.manager.window(ModeId.TRAINING)
    window.resize(1800, 850)
    window.show()
    page.preview_action.setChecked(False)
    qapp.processEvents()
    page._update_form_columns(1500)
    assert page._two_columns, f"scroll width={page.scroll.width()}, window={window.width()}"

    anchor = page._model_widgets["mask_rcnn"]["model.anchors.sizes"]
    anchor.setFocus()
    QTest.keyClick(anchor, Qt.Key.Key_Left)
    cursor_position = anchor.cursorPosition()
    config_before = copy.deepcopy(page._collect_config())
    yaml_before = page.yaml_preview.toPlainText()

    page.preview_action.trigger()
    qapp.processEvents()
    assert not page._two_columns
    assert anchor.hasFocus()
    assert anchor.cursorPosition() == cursor_position
    assert [item.cursor.selectedText().strip() for item in page.yaml_preview.extraSelections()] == [
        "sizes:"
    ]

    page.preview_action.trigger()
    qapp.processEvents()
    page._update_form_columns(1500)
    assert page._two_columns
    assert anchor.hasFocus()
    assert anchor.cursorPosition() == cursor_position
    window.resize(900, 650)
    qapp.processEvents()
    assert not page._two_columns
    assert anchor.hasFocus()
    assert anchor.cursorPosition() == cursor_position
    window.resize(1500, 850)
    qapp.processEvents()
    page._update_form_columns(1500)
    assert page._two_columns
    assert anchor.hasFocus()
    assert anchor.cursorPosition() == cursor_position
    assert page._collect_config() == config_before
    assert page.yaml_preview.toPlainText() == yaml_before


def test_roi_yaml_highlight_matches_full_nested_path(shell, qapp, qtbot):
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    window = shell.manager.window(ModeId.TRAINING)
    window.show()
    window.activateWindow()
    page.preview_action.setChecked(True)
    for section in page.model_stack.widget(0).findChildren(CollapsibleSection):
        if not section.button.isChecked():
            QTest.mouseClick(section.button, Qt.MouseButton.LeftButton)
    roi = page._model_widgets["mask_rcnn"]["model.roi.fg_iou_thresh"]
    page.scroll.ensureWidgetVisible(roi)
    qapp.processEvents()
    roi.setValue(0.61)
    QTest.mouseClick(roi.lineEdit(), Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert roi.lineEdit().hasFocus(), (
        f"visible={roi.isVisible()}, window={window.isVisible()}, "
        f"stack={page.model_stack.currentIndex()}, parent={roi.parentWidget().isVisible()}"
    )
    highlighted = [
        selection.cursor.selectedText().strip() for selection in page.yaml_preview.extraSelections()
    ]
    assert highlighted == ["fg_iou_thresh: 0.61"]
    assert page._collect_config()["model"]["roi"]["fg_iou_thresh"] == pytest.approx(0.61)


def test_summary_tracks_cv_stratification_and_source_folder_grouping(shell, qapp, qtbot):
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    stratify = page.fields["data.cv.stratify_by_classification"]
    group = page.fields["data.cv.group_by_source_folder"]
    stratify.setChecked(False)
    group.setChecked(False)
    qapp.processEvents()
    initial = page.summary_label.text()
    page.scroll.ensureWidgetVisible(stratify)
    qapp.processEvents()
    stratify.setFocus()
    QTest.keyClick(stratify, Qt.Key.Key_Space)
    qapp.processEvents()
    assert stratify.isChecked()
    assert page.summary_label.text() != initial
    assert "層別" in page.summary_label.text()
    page.scroll.ensureWidgetVisible(group)
    qapp.processEvents()
    group.setFocus()
    QTest.keyClick(group, Qt.Key.Key_Space)
    qapp.processEvents()
    assert "フォルダ単位" in page.summary_label.text()
    stratify.setFocus()
    QTest.keyClick(stratify, Qt.Key.Key_Space)
    qapp.processEvents()
    assert "層別" not in page.summary_label.text()
    assert "フォルダ単位" in page.summary_label.text()
    assert "group_by_source_folder: true" in page.yaml_preview.toPlainText()
