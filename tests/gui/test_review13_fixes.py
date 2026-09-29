"""Review 13 regression checks through visible training settings operations."""

import copy

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from foam_cell_analysis.gui.navigation import ModeId, PageId


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


@pytest.mark.parametrize("finish", ["save", "cancel"])
def test_opening_another_queue_row_keeps_form_from_before_queue_edit(shell, qapp, finish):
    """キュー編集中に別の行を開いても、最初のキュー編集前の設定に戻り、未保存の変更は移らない。"""
    backend = shell.ctx.backend
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    page.fields["training.epochs"].setValue(33)
    page.description_edit.setText("元の新規設定")
    qapp.processEvents()

    first = _enqueue_waiting_row(backend, epochs=17)
    second = _enqueue_waiting_row(backend, epochs=27)
    page = _open_queue_editor(shell, first, qapp)
    page.fields["training.epochs"].setValue(18)
    page = _open_queue_editor(shell, second, qapp)
    assert page.fields["training.epochs"].value() == 27

    if finish == "save":
        page.fields["training.epochs"].setValue(30)
        QTest.mouseClick(page.save_button, Qt.MouseButton.LeftButton)
    else:
        QTest.mouseClick(page.cancel_queue_edit_button, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    assert backend.get_experiment(first.experiment_id).config.values["training"]["epochs"] == 17
    expected_second = 30 if finish == "save" else 27
    assert (
        backend.get_experiment(second.experiment_id).config.values["training"]["epochs"]
        == expected_second
    )
    shell.navigate(PageId.TRAINING)
    qapp.processEvents()
    assert page.fields["training.epochs"].value() == 33
    assert page.description_edit.text() == "元の新規設定"
    assert page.queue_edit_banner.isHidden()
