import pytest
from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt
from PySide6.QtTest import QTest

from foam_cell_analysis.gui.navigation import PageId


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
