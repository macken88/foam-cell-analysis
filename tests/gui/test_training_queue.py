import copy
import time

import pytest
from PySide6.QtCore import QItemSelection, QItemSelectionModel, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDoubleSpinBox, QMessageBox, QSpinBox

from foam_cell_analysis.gui.navigation import PageId
from foam_cell_analysis.gui.theme import Color


def queue_value(queue, row, column):
    return queue.model.data(queue.model.index(row, column), Qt.ItemDataRole.DisplayRole)


def test_training_queue_tab_and_enqueue_flow(shell, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
    )
    page = shell.page(PageId.TRAINING)
    assert hasattr(page, "queue_button")
    for _ in range(3):
        page.queue_button.click()
    queue = shell.page(PageId.TRAINING_QUEUE)
    assert queue.model.rowCount() == 3
    ids = [queue_value(queue, row, 2) for row in range(3)]
    assert len(set(ids)) == 3
    assert all(item.status == "queued" for item in shell.ctx.backend.list_training_queue())


def test_queue_edit_validation_and_order_operations(shell):
    queue = shell.page(PageId.TRAINING_QUEUE)
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["experiment"]["id"] = backend.next_experiment_id()
    backend.add_training_queue_item(config)
    queue.refresh()
    assert queue.model.rowCount() == 1
    assert queue_value(queue, 0, 0) == "1"
    assert queue.edit_cell(0, "training.epochs", 0) is True
    queue.refresh()
    assert "設定エラー" in queue_value(queue, 0, 1)


def test_queue_edit_model_defaults_duplicate_move_and_delete(shell, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
    )
    training = shell.page(PageId.TRAINING)
    queue = shell.page(PageId.TRAINING_QUEUE)
    shell.navigate(PageId.TRAINING)
    QTest.mouseClick(training.queue_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(training.queue_button, Qt.MouseButton.LeftButton)
    queue.refresh()
    assert queue.model.rowCount() == 2
    queue.edit_cell(0, "training.epochs", 9)
    queue.edit_cell(0, "training.learning_rate", 0.00025)
    first = queue_value(queue, 0, 2)
    updated = shell.ctx.backend.get_experiment(first)
    assert updated.config.values["training"]["epochs"] == 9
    assert "learning_rate: 0.00025" in updated.config.to_yaml()
    queue.edit_cell(0, "model.type", "cellpose")
    assert shell.ctx.backend.get_experiment(first).config.values["model"]["type"] == "cellpose"
    assert (
        "pretrained_weights" not in shell.ctx.backend.get_experiment(first).config.values["model"]
    )
    queue.table.selectRow(0)
    second = queue_value(queue, 1, 2)
    queue.duplicate_selected()
    assert queue.model.rowCount() == 3
    assert queue_value(queue, 1, 2) != second
    queue.table.selectRow(2)
    queue.move_selected(-1)
    assert queue_value(queue, 1, 2) != queue_value(queue, 2, 2)
    queue.table.selectRow(2)
    queue.delete_selected()
    assert queue.model.rowCount() == 2


def test_queue_row_opens_training_editor_save_and_cancel(shell):
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["experiment"]["id"] = backend.next_experiment_id()
    queued = backend.add_training_queue_item(config)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    queue.edit_row(0)
    editor = shell.page(PageId.TRAINING)
    assert editor.queue_edit_banner.text() == f"キューの {queued.experiment_id} を編集中"
    assert editor.training_actions["save"].text() == "キューに保存"
    assert editor.cancel_queue_edit_button.text() == "キャンセル"
    assert editor.validate_button.isHidden()
    assert editor.queue_button.isHidden()
    assert editor.start_button.isHidden()
    editor.fields["training.epochs"].setValue(7)
    editor._save_button_clicked()
    assert backend.get_experiment(queued.experiment_id).config.values["training"]["epochs"] == 7
    editor.on_enter({"edit_queue": queued.experiment_id})
    editor.fields["training.epochs"].setValue(8)
    editor._cancel_queue_edit()
    assert backend.get_experiment(queued.experiment_id).config.values["training"]["epochs"] == 7


def test_queue_refresh_restores_multi_selection(shell):
    backend = shell.ctx.backend
    for _ in range(3):
        config = backend.default_experiment_config("mask_rcnn")
        config["experiment"]["id"] = backend.next_experiment_id()
        backend.add_training_queue_item(config)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    selection = QItemSelection()
    for row in (0, 2):
        selection.select(
            queue.model.index(row, 0), queue.model.index(row, queue.model.columnCount() - 1)
        )
    queue.table.selectionModel().select(
        selection,
        QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
    )
    selected_before = set(queue._selected_ids())
    queue.refresh()
    assert set(queue._selected_ids()) == selected_before


def test_queue_failure_does_not_block_following_item(shell):
    from PySide6.QtTest import QTest

    backend = shell.ctx.backend
    training = shell.page(PageId.TRAINING)
    for _ in range(2):
        training.queue_button.click()
    entries = backend.list_training_queue()
    for item in entries:
        config = item.config.values.copy()
        config["training"] = config["training"].copy()
        config["training"]["epochs"] = 1
        backend.update_training_queue_item(item.experiment_id, config)
    backend.fail_training_ids.add(entries[0].experiment_id)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    queue.run_button.click()
    statuses = [backend.get_experiment(item.experiment_id).status for item in entries]
    for _ in range(100):
        if statuses == ["failed", "completed"]:
            break
        QTest.qWait(100)
        statuses = [backend.get_experiment(item.experiment_id).status for item in entries]
    assert statuses == ["failed", "completed"]
    for _ in range(100):
        if not shell.ctx.queue_controller.executing:
            break
        QTest.qWait(10)
    assert not shell.ctx.queue_controller.executing
    failed_id = entries[0].experiment_id
    queue.refresh()
    # 終わった行（失敗・完了）は自動でキューの表から外れ、実験の記録は残る
    assert queue.model.rowCount() == 0
    assert backend.get_experiment(failed_id).status == "failed"


@pytest.mark.slow
def test_queue_stop_finishes_active_item_and_leaves_next_waiting(shell, qtbot):
    backend = shell.ctx.backend
    for _ in range(2):
        config = backend.default_experiment_config("mask_rcnn")
        config["experiment"]["id"] = backend.next_experiment_id()
        config["training"]["epochs"] = 30
        backend.add_training_queue_item(config)
    entries = backend.list_training_queue()
    controller = shell.ctx.queue_controller
    controller.start()
    QTest.qWait(5)
    controller.stop()
    qtbot.waitUntil(
        lambda: (
            backend.get_experiment(entries[0].experiment_id).status == "completed"
            and not controller.executing
        ),
        timeout=60_000,
    )
    assert backend.get_experiment(entries[0].experiment_id).status == "completed"
    assert backend.get_experiment(entries[1].experiment_id).status == "queued"


def test_queue_keeps_running_and_selection_when_window_reopens(shell):
    from PySide6.QtTest import QTest

    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["experiment"]["id"] = backend.next_experiment_id()
    config["training"]["epochs"] = 80
    item = backend.add_training_queue_item(config)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    queue.table.selectRow(0)
    shell.manager.window("training").close()
    shell.ctx.queue_controller.start()
    QTest.qWait(5)
    shell.navigate(PageId.TRAINING_QUEUE)
    assert shell.ctx.queue_controller.executing
    assert queue.model.entries[queue.table.currentIndex().row()].experiment_id == item.experiment_id
    shell.ctx.queue_controller.stop()
    job = shell.ctx.jobs.find(f"training:{item.experiment_id}")
    if job:
        job.cancel()
    QTest.qWait(10)


def test_training_start_during_queue_adds_without_starting_second_job(shell, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
    )
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["experiment"]["id"] = backend.next_experiment_id()
    config["training"]["epochs"] = 80
    first = backend.add_training_queue_item(config)
    controller = shell.ctx.queue_controller
    controller.start()
    assert controller.active_id == first.experiment_id
    training = shell.page(PageId.TRAINING)
    queued = training.start_training(confirm=False)
    assert queued is not None
    assert len(backend.list_training_queue()) == 2
    assert shell.ctx.jobs.running_count == 1
    controller.stop()
    job = shell.ctx.jobs.find(f"training:{first.experiment_id}")
    if job:
        job.cancel()


def test_experiment_list_copies_checked_experiments_into_queue(shell):
    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    experiment = shell.ctx.backend.list_experiments()[0]
    original_id = experiment.config.values["experiment"]["id"]
    for row in range(page.table.rowCount()):
        if page.table.item(row, 1).text() == experiment.experiment_id:
            page.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
            break
    page.action_map["queue_copy"].trigger()
    queued = shell.ctx.backend.list_training_queue()
    assert len(queued) == 1
    assert queued[0].experiment_id != experiment.experiment_id
    assert experiment.config.values["experiment"]["id"] == original_id


def test_experiment_list_copies_legacy_checked_experiment_into_queue(shell):
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["data"].pop("cv")
    config["data"]["split_id"] = "split_001"
    config["checkpoint"].pop("save_fold_models")
    config["checkpoint"]["save_best"] = True
    config["checkpoint"]["save_last"] = True
    experiment = backend.save_experiment_draft(config)
    experiment.config.values["data"].pop("cv")
    experiment.config.values["data"]["split_id"] = "split_001"
    experiment.config.values["checkpoint"].pop("save_fold_models")
    experiment.config.values["checkpoint"]["save_best"] = True
    experiment.config.values["checkpoint"]["save_last"] = True
    original = copy.deepcopy(experiment.config.values)

    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    page.refresh()
    row = next(
        row
        for row in range(page.table.rowCount())
        if page.table.item(row, 1).text() == experiment.experiment_id
    )
    rect = page.table.visualItemRect(page.table.item(row, 0))
    checkbox_pos = rect.topLeft() + QPoint(12, rect.height() // 2)
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=checkbox_pos)
    menu_bar = page.window().menuBar()
    top = next(action for action in menu_bar.actions() if action.text().startswith("学習"))
    QTest.mouseClick(menu_bar, Qt.MouseButton.LeftButton, pos=menu_bar.actionGeometry(top).center())
    menu = top.menu()
    action = page.action_map["queue_copy"]
    assert action.isEnabled()
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())

    queued = shell.ctx.backend.list_training_queue()
    assert len(queued) == 1
    assert queued[0].experiment_id != experiment.experiment_id
    assert queued[0].config.values["data"]["cv"]["n_folds"] == 5
    assert queued[0].config.values["checkpoint"]["save_fold_models"] is True
    assert experiment.config.values == original


def test_experiment_list_refuses_protected_experiment_queue_copy(shell):
    backend = shell.ctx.backend
    experiment = backend.list_experiments()[0]
    original = copy.deepcopy(experiment.config.values)
    experiment.recovery_state = "termination_unknown"
    experiment.recovery_reason = "終了状態を確認できません"

    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    row = next(
        row
        for row in range(page.table.rowCount())
        if page.table.item(row, 1).text() == experiment.experiment_id
    )
    rect = page.table.visualItemRect(page.table.item(row, 0))
    checkbox_pos = rect.topLeft() + QPoint(12, rect.height() // 2)
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=checkbox_pos)
    menu_bar = page.window().menuBar()
    top = next(action for action in menu_bar.actions() if action.text().startswith("学習"))
    QTest.mouseClick(menu_bar, Qt.MouseButton.LeftButton, pos=menu_bar.actionGeometry(top).center())
    menu = top.menu()
    action = page.action_map["queue_copy"]

    assert menu.isVisible()
    assert not action.isEnabled()
    assert backend.list_training_queue() == []
    assert experiment.config.values == original


def test_training_queue_order_and_page_navigation(shell):
    queue = shell.page(PageId.TRAINING_QUEUE)
    assert queue is not None
    shell.navigate(PageId.TRAINING_QUEUE)
    assert shell.manager.current_page_id("training") == PageId.TRAINING_QUEUE
    assert queue.empty_label.text().startswith("キューは空です")


def test_terminal_save_failure_stops_queue_and_reports_reason(shell, qapp, qtbot, monkeypatch):
    backend = shell.ctx.backend
    first_config = backend.default_experiment_config("mask_rcnn")
    first_config["training"]["epochs"] = 1
    first_config["data"]["cv"]["n_folds"] = 2
    first = backend.add_training_queue_item(first_config)
    second_config = backend.default_experiment_config("mask_rcnn")
    second = backend.add_training_queue_item(second_config)
    original_conclude = backend.conclude_training_run

    def fail_first_conclusion(experiment_id, attempt, job_exit):
        if experiment_id == first.experiment_id:
            raise OSError("status.json の保存に失敗")
        return original_conclude(experiment_id, attempt, job_exit)

    monkeypatch.setattr(backend, "conclude_training_run", fail_first_conclusion)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    with qtbot.waitSignal(shell.ctx.training_runner.ended, timeout=5_000):
        QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    assert not shell.ctx.queue_controller.executing
    assert queue.model.rowCount() == 2
    assert backend.get_experiment(second.experiment_id).status == "queued"
    assert "終端状態を保存できないため、キューを停止しました" in shell.status_text.text()
    assert "status.json の保存に失敗" in shell.status_text.text()


def _open_context_menu(view, pos):
    """右クリックで届くのと同じ QContextMenuEvent を表へ送る。"""
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QApplication

    viewport = view.viewport()
    QTest.mouseClick(viewport, Qt.MouseButton.RightButton, pos=pos)
    QApplication.sendEvent(
        viewport,
        QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos, viewport.mapToGlobal(pos)),
    )


def _menu_action(menu, text):
    return next(action for action in menu.actions() if action.text().split("\t")[0] == text)


def test_start_button_queues_config_runs_it_and_removes_finished_row(
    shell, qapp, qtbot, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    from foam_cell_analysis.gui.modes.training.queue_model import PROGRESS_ROLE
    from foam_cell_analysis.gui.navigation import ModeId

    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
    )
    backend = shell.ctx.backend
    training = shell.page(PageId.TRAINING)
    training.fields["training.epochs"].setValue(3)
    training.fields["data.cv.n_folds"].setValue(2)
    QTest.mouseClick(training.start_button, Qt.MouseButton.LeftButton)

    rows = backend.list_training_queue()
    assert len(rows) == 1
    experiment_id = rows[0].experiment_id
    assert rows[0].status == "running"
    assert shell.ctx.queue_controller.executing
    assert shell.ctx.training_runner.experiment_id == experiment_id

    queue = shell.page(PageId.TRAINING_QUEUE)
    window = shell.manager.window(ModeId.TRAINING)
    qtbot.waitUntil(lambda: backend.get_experiment(experiment_id).current_epoch >= 1, timeout=5_000)
    queue.refresh()
    status = queue.model.index(0, 1)
    assert status.data() == "実行中"
    assert 0.0 <= status.data(PROGRESS_ROLE) <= 1.0
    assert "エポック" in status.data(Qt.ItemDataRole.ToolTipRole)
    assert not window.training_progress_bar.isHidden()
    assert window.training_progress_label.text().startswith(f"{experiment_id} 学習中：")

    qtbot.waitUntil(
        lambda: backend.get_experiment(experiment_id).status == "completed", timeout=5_000
    )
    assert backend.get_experiment(experiment_id).status == "completed"
    qapp.processEvents()
    assert backend.list_training_queue() == []
    assert queue.model.rowCount() == 0
    assert window.training_progress_bar.isHidden()


def test_queue_context_menu_removes_finished_rows_and_keeps_experiments(shell, qapp, monkeypatch):
    from foam_cell_analysis.gui.navigation import ModeId

    backend = shell.ctx.backend
    finished = backend.add_training_queue_item(backend.default_experiment_config("mask_rcnn"))
    waiting = backend.add_training_queue_item(backend.default_experiment_config("mask_rcnn"))
    finished.status = "stopped"
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    queue.refresh()
    window = shell.manager.window(ModeId.TRAINING)
    window.show()
    shown = []
    monkeypatch.setattr(queue.context_menu, "exec", lambda *_args: shown.append(queue.context_menu))
    _open_context_menu(queue.table, queue.table.visualRect(queue.model.index(0, 2)).center())
    assert shown == [queue.context_menu]
    menu = queue.context_menu
    action = _menu_action(menu, "終了・中断した行を削除")
    assert action.toolTip().startswith(
        "キューの表から外すだけです。実験の記録（実験一覧）は残ります。"
    )
    menu.popup(queue.table.viewport().mapToGlobal(queue.table.viewport().rect().center()))
    qapp.processEvents()
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
    qapp.processEvents()

    assert [row.experiment_id for row in backend.list_training_queue()] == [waiting.experiment_id]
    assert backend.get_experiment(finished.experiment_id).status == "stopped"
    window.hide()


def test_failed_retry_preparation_does_not_conclude_previous_attempt(shell, monkeypatch):
    """完了後の再学習が準備で失敗しても、前回の試行を「完了」として扱い直さない。"""
    backend = shell.ctx.backend
    training = shell.page(PageId.TRAINING)
    config = training._collect_config()
    config["training"]["epochs"] = 1
    config["data"]["cv"]["n_folds"] = 2
    experiment = backend.add_training_queue_item(config)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    for _ in range(200):
        if backend.get_experiment(experiment.experiment_id).status == "completed":
            break
        QTest.qWait(20)
    for _ in range(100):
        if not shell.ctx.queue_controller.executing and not shell.ctx.training_runner.is_busy:
            break
        QTest.qWait(10)
    assert backend.get_experiment(experiment.experiment_id).status == "completed"

    concluded = []
    original_conclude = backend.conclude_training_run
    monkeypatch.setattr(
        backend,
        "conclude_training_run",
        lambda *args, **kwargs: concluded.append(args) or original_conclude(*args, **kwargs),
    )

    def fail_prepare(*_args, **_kwargs):
        raise ValueError("データセットが初回試行と異なります")

    monkeypatch.setattr(backend, "prepare_training_run", fail_prepare)
    outcomes = []
    shell.ctx.training_runner.ended.connect(outcomes.append)
    reservation = backend.add_training_retry_reservation(experiment.experiment_id)
    shell.ctx.training_runner.start(experiment.experiment_id, reservation.queue_id, retry=True)

    assert concluded == []
    assert [(item.status, item.reason) for item in outcomes] == [("failed", "prepare_failed")]
    assert backend.get_experiment(experiment.experiment_id).status == "completed"
    queue.refresh()
    assert any(
        entry.queue_id == reservation.queue_id and entry.status == "failed"
        for entry in backend.list_training_queue()
    )


def test_invalid_queue_config_is_saved_and_reported_instead_of_raising(shell):
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["experiment"]["id"] = backend.next_experiment_id()
    config["data"]["dataset_version"] = "missing_train_v999"
    item = backend.add_training_queue_item(config)
    assert item.config.values["data"]["used_item_ids"] == []
    issues = backend.validate_experiment_config(item.config.values)
    assert any(issue["level"] == "error" for issue in issues)


def test_validation_excludes_the_same_experiment_but_checks_duplicate_queue_rows(shell):
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["experiment"]["id"] = backend.next_experiment_id()
    config["experiment"]["description"] = "identical config"
    first = backend.add_training_queue_item(config)
    same_row_issues = backend.validate_experiment_config(first.config.values)
    assert not any("同一設定" in issue["message"] for issue in same_row_issues)
    duplicate = copy.deepcopy(first.config.values)
    duplicate["experiment"]["id"] = backend.next_experiment_id()
    backend.add_training_queue_item(duplicate)
    issues = backend.validate_experiment_config(duplicate)
    assert any("同一設定" in issue["message"] for issue in issues)


def test_queue_table_cell_edit_to_invalid_value_is_handled_without_slot_exception(shell):
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["experiment"]["id"] = backend.next_experiment_id()
    config["data"]["classification"] = "C"
    queued = backend.add_training_queue_item(config)
    shell.navigate(PageId.TRAINING_QUEUE)
    page = shell.page(PageId.TRAINING_QUEUE)
    row = page.model.row_for_id(queued.experiment_id)
    index = page.model.index(row, page.model.column_for_path("data.cv.n_folds"))
    page.table.setCurrentIndex(index)
    QTest.keyClick(page.table, Qt.Key.Key_F2)
    spin = page.table.findChild(QSpinBox)
    assert spin is not None
    spin.setValue(10)
    other = page.model.index(row, page.model.column_for_path("training.epochs"))
    QTest.mouseClick(
        page.table.viewport(), Qt.MouseButton.LeftButton, pos=page.table.visualRect(other).center()
    )
    assert backend.get_experiment(queued.experiment_id).config.values["data"]["cv"]["n_folds"] == 10
    assert page.model.data(page.model.index(row, 1), Qt.ItemDataRole.DisplayRole) == "設定エラー"
    assert page.model.data(page.model.index(row, 0), Qt.ItemDataRole.BackgroundRole).name() == (
        Color.ERROR_BG.lower()
    )


def add_queue_item(backend, *, epochs=30, model="mask_rcnn", config=None, folds=None):
    values = config or backend.default_experiment_config(model)
    values["experiment"]["id"] = backend.next_experiment_id()
    values["training"]["epochs"] = epochs
    if folds is not None:
        values["data"]["cv"]["n_folds"] = folds
    return backend.add_training_queue_item(values)


def wait_for(qapp, condition, timeout=3000):
    deadline = time.monotonic() + timeout / 1000
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        QTest.qWait(5)
    return condition()


def training_jobs(shell):
    return shell.ctx.jobs.training_jobs


def test_second_single_training_is_queued_while_first_runs(shell, monkeypatch, qapp):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    training = shell.page(PageId.TRAINING)
    training.fields["training.epochs"].setValue(100)
    QTest.mouseClick(training.start_button, Qt.MouseButton.LeftButton)
    first_id = next(
        item.experiment_id
        for item in shell.ctx.backend.list_experiments()
        if item.status == "running"
    )
    shell.navigate(PageId.TRAINING)
    QTest.mouseClick(training.start_button, Qt.MouseButton.LeftButton)
    assert len(training_jobs(shell)) == 1
    assert shell.ctx.queue_controller.executing
    # 学習開始はキュー経由になったため、1 件目も実行中の行としてキューに並ぶ
    queue_rows = shell.ctx.backend.list_training_queue()
    assert [row.status for row in queue_rows] == ["running", "queued"]
    assert queue_rows[0].experiment_id == first_id
    queue = shell.page(PageId.TRAINING_QUEUE)
    assert f"実行中 {first_id}" in queue.status_line.text()
    # 後片付け: 「今すぐ停止」でキューごと止め、2 件目を走らせない
    shell.navigate(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.stop_now_button, Qt.MouseButton.LeftButton)
    assert wait_for(qapp, lambda: not shell.ctx.queue_controller.executing)
    assert [row.status for row in shell.ctx.backend.list_training_queue()] == ["queued"]


def test_retry_during_queue_execution_is_reserved_without_rewriting_history(
    shell, monkeypatch, qapp
):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    backend = shell.ctx.backend
    current = add_queue_item(backend, epochs=100)
    retry = backend.get_experiment("exp_0044")
    retry.status = "stopped"
    original_config = retry.config.to_yaml()
    shell.ctx.queue_controller.start()
    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    row = next(
        row
        for row in range(page.table.rowCount())
        if page.table.item(row, 1).text() == retry.experiment_id
    )
    page.table.setCurrentCell(row, 1)
    page.action_map["retry"].trigger()
    queued = backend.list_training_queue()
    assert len(training_jobs(shell)) == 1
    assert len(queued) == 2
    assert queued[-1].experiment_id == retry.experiment_id
    assert queued[-1].queue_is_retry
    assert queued[-1].queue_retry_attempt == len(retry.runs) + 1
    assert queued[-1].status == "queued"
    assert retry.status == "stopped"
    assert retry.config.to_yaml() == original_config
    job = training_jobs(shell)[0]
    job.cancel()
    wait_for(qapp, lambda: not shell.ctx.queue_controller.executing)
    assert backend.get_experiment(current.experiment_id).status == "stopped"


def test_detail_editor_becomes_read_only_if_queued_row_starts(shell, qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    backend = shell.ctx.backend
    # 1 件目は短く終わらせ、2 件目が実行中になる瞬間を編集画面で観察する
    first = add_queue_item(backend, epochs=1, folds=2)
    second = add_queue_item(backend, epochs=100)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    row = queue.model.row_for_id(second.experiment_id)
    index = queue.model.index(row, 2)
    queue.table.scrollTo(index)
    QTest.qWait(5)
    QTest.mouseClick(
        queue.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=queue.table.visualRect(index).center(),
    )
    QTest.mouseDClick(
        queue.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=queue.table.visualRect(index).center(),
    )
    editor = shell.page(PageId.TRAINING)
    assert editor._queue_edit_id == second.experiment_id
    assert editor.save_button.isEnabled()
    shell.navigate(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    assert wait_for(qapp, lambda: backend.get_experiment(second.experiment_id).status == "running")
    assert backend.get_experiment(first.experiment_id).status == "completed"
    assert second.experiment_id in editor.queue_edit_banner.text()
    assert "編集を保存できません" in editor.queue_edit_banner.text()
    assert not editor.save_button.isEnabled()
    assert editor.cancel_queue_edit_button.text() == "閉じる"
    before = backend.get_experiment(second.experiment_id).config.to_yaml()
    QTest.mouseClick(editor.save_button, Qt.MouseButton.LeftButton)
    assert backend.get_experiment(second.experiment_id).config.to_yaml() == before
    QTest.mouseClick(editor.cancel_queue_edit_button, Qt.MouseButton.LeftButton)
    assert editor._queue_edit_id is None
    assert editor.start_button.isVisible()
    shell.navigate(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.stop_now_button, Qt.MouseButton.LeftButton)
    assert wait_for(qapp, lambda: not shell.ctx.queue_controller.executing)


def test_detail_editor_reports_if_row_is_deleted_and_catches_save_errors(shell, monkeypatch):
    backend = shell.ctx.backend
    item = add_queue_item(backend)
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    queue.edit_row(queue.model.row_for_id(item.experiment_id))
    editor = shell.page(PageId.TRAINING)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue.table.selectRow(queue.model.row_for_id(item.experiment_id))
    queue.queue_actions["delete"].trigger()
    assert "削除されたため" in editor.queue_edit_banner.text()
    assert not editor.save_button.isEnabled()
    editor._restore_before_queue_edit()

    another = add_queue_item(backend)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue.refresh()
    queue.edit_row(queue.model.row_for_id(another.experiment_id))
    editor = shell.page(PageId.TRAINING)
    monkeypatch.setattr(
        backend,
        "update_training_queue_item",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("保存競合")),
    )
    QTest.mouseClick(editor.save_button, Qt.MouseButton.LeftButton)
    assert warnings and warnings[-1][1] == "保存競合"
    assert editor._queue_edit_id == another.experiment_id


def test_stop_reservation_is_cleared_after_current_training(shell, qapp):
    first = add_queue_item(shell.ctx.backend, epochs=1, folds=2)
    second = add_queue_item(shell.ctx.backend, epochs=1, folds=2)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(queue.stop_after_button, Qt.MouseButton.LeftButton)
    assert wait_for(qapp, lambda: not shell.ctx.queue_controller.executing)
    assert first.status != "queued"
    assert shell.ctx.backend.get_experiment(second.experiment_id).status == "queued"
    assert not shell.ctx.queue_controller.stop_requested
    assert "今の学習が終わったら停止します" not in queue.status_line.text()


def test_move_buttons_are_enabled_only_for_their_direction(shell):
    backend = shell.ctx.backend
    for _ in range(3):
        add_queue_item(backend)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    queue.table.selectRow(0)
    assert not queue.up_button.isEnabled()
    assert queue.down_button.isEnabled()
    queue.table.selectRow(2)
    assert queue.up_button.isEnabled()
    assert not queue.down_button.isEnabled()


def test_queue_status_line_reports_latest_active_progress(shell, qapp):
    item = add_queue_item(shell.ctx.backend, epochs=100)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    assert wait_for(
        qapp, lambda: shell.ctx.backend.get_experiment(item.experiment_id).current_epoch >= 3
    )
    assert wait_for(
        qapp,
        lambda: (
            (
                f"エポック {shell.ctx.backend.get_experiment(item.experiment_id).current_epoch}/"
                f"{shell.ctx.backend.get_experiment(item.experiment_id).total_epochs}"
            )
            in queue.status_line.text()
        ),
    )
    job = training_jobs(shell)[0]
    job.cancel()


def test_double_clicking_nonwaiting_fixed_column_opens_experiment_list(shell, qapp):
    item = add_queue_item(shell.ctx.backend, epochs=100)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    assert wait_for(
        qapp, lambda: shell.ctx.backend.get_experiment(item.experiment_id).status == "running"
    )
    row = queue.model.row_for_id(item.experiment_id)
    index = queue.model.index(row, 0)
    rect = queue.table.visualRect(index)
    QTest.mouseClick(queue.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    QTest.mouseDClick(queue.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert shell.manager.current_page_id("training") == PageId.EXPERIMENTS
    experiments = shell.page(PageId.EXPERIMENTS)
    assert experiments._current_experiment().experiment_id == item.experiment_id
    job = training_jobs(shell)[0]
    job.cancel()


def test_queue_check_cell_click_changes_setting(shell):
    """チェック欄をクリックすると、キュー行の設定が実際に変わる。"""
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["experiment"]["id"] = backend.next_experiment_id()
    row = backend.add_training_queue_item(config)
    shell.navigate(PageId.TRAINING_QUEUE)
    page = shell.page(PageId.TRAINING_QUEUE)
    column = page.model.column_for_path("data.cv.stratify_by_classification")
    index = page.model.index(0, column)
    assert page.model.data(index, Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
    rect = page.table.visualRect(index)
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    saved = backend.get_experiment(row.experiment_id).config.values
    assert saved["data"]["cv"]["stratify_by_classification"] is False


def test_queue_edit_survives_live_progress_and_is_saved(shell, qapp):
    """学習の進捗で表が更新されても、編集中の入力が消えずに保存できる。"""
    backend = shell.ctx.backend
    first_config = backend.default_experiment_config("mask_rcnn")
    first_config["experiment"]["id"] = backend.next_experiment_id()
    first_config["training"]["epochs"] = 200
    first = backend.add_training_queue_item(first_config)
    waiting_config = backend.default_experiment_config("mask_rcnn")
    waiting_config["experiment"]["id"] = backend.next_experiment_id()
    waiting = backend.add_training_queue_item(waiting_config)
    shell.navigate(PageId.TRAINING_QUEUE)
    page = shell.page(PageId.TRAINING_QUEUE)
    QTest.mouseClick(page.run_button, Qt.MouseButton.LeftButton)
    assert shell.ctx.queue_controller.active_id == first.experiment_id

    waiting_row = next(
        row
        for row, entry in enumerate(page.model.entries)
        if entry.experiment_id == waiting.experiment_id
    )
    index = page.model.index(waiting_row, page.model.column_for_path("training.learning_rate"))
    QTest.mouseClick(
        page.table.viewport(), Qt.MouseButton.LeftButton, pos=page.table.visualRect(index).center()
    )
    QTest.keyClick(page.table, Qt.Key.Key_F2)
    qapp.processEvents()
    editor = page.table.focusWidget()
    assert isinstance(editor, QDoubleSpinBox) and editor.isVisible()
    editor.selectAll()
    QTest.keyClicks(editor, "3.7e-4")
    # 学習の進捗による表の更新が、編集の途中に何度も起きる
    started = time.monotonic()
    while time.monotonic() - started < 0.3:
        qapp.processEvents()
    assert shell.ctx.backend.get_experiment(first.experiment_id).current_epoch > 0
    assert editor.isVisible()
    QTest.keyClick(editor, Qt.Key.Key_Enter)
    qapp.processEvents()
    saved = backend.get_experiment(waiting.experiment_id).config.values
    assert saved["training"]["learning_rate"] == pytest.approx(3.7e-4)
    shell.ctx.queue_controller.stop()
    shell.ctx.training_runner.request_stop(timeout_ms=5000)


def test_queue_waits_while_evaluation_holds_compute_and_starts_after_release(shell, qapp, qtbot):
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["training"]["epochs"] = 1
    config["data"]["cv"]["n_folds"] = 2
    item = backend.add_training_queue_item(config)
    compute = shell.ctx.compute
    evaluation = compute.request("evaluation", "評価 cand_001", lambda: None)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    queue.refresh()

    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    QTest.qWait(50)

    controller = shell.ctx.queue_controller
    assert controller.executing and controller.waiting_for_compute
    assert not shell.ctx.training_runner.is_busy
    assert backend.get_experiment(item.experiment_id).status == "queued"
    assert "評価の終了を待っています" in queue.status_line.text()

    with qtbot.waitSignal(shell.ctx.training_runner.ended, timeout=4_000) as signal:
        compute.release(evaluation)
    assert signal.args[0].experiment_id == item.experiment_id
    assert signal.args[0].status == "completed"
    qtbot.waitUntil(lambda: not controller.executing, timeout=2_000)
    assert not compute.is_busy
    assert "評価の終了を待っています" not in queue.status_line.text()
