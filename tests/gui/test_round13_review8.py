"""回帰テスト: 学習キューの再レビュー指摘。"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from foam_cell_analysis.gui.navigation import PageId
from foam_cell_analysis.gui.theme import Color


def add_queue_item(backend, *, epochs=30, model="mask_rcnn", config=None):
    values = config or backend.default_experiment_config(model)
    values["experiment"]["id"] = backend.next_experiment_id()
    values["training"]["epochs"] = epochs
    return backend.add_training_queue_item(values)


def wait_for(qapp, condition, timeout=3000):
    elapsed = 0
    while elapsed < timeout:
        qapp.processEvents()
        if condition():
            return True
        QTest.qWait(5)
        elapsed += 5
    return condition()


def training_jobs(shell):
    return shell.ctx.jobs.training_jobs


def test_training_runner_owns_the_single_training_slot(shell):
    backend = shell.ctx.backend
    experiment = backend.start_training(backend.default_experiment_config("mask_rcnn"))
    shell.ctx.training_runner.start(experiment.experiment_id)

    with pytest.raises(RuntimeError, match="別の学習"):
        shell.ctx.training_runner.start("exp_other")

    assert shell.ctx.jobs.has_training_job
    assert shell.ctx.jobs.jobs() == []
    shell.ctx.training_runner.request_stop()


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
    job = training_jobs(shell)[0]
    job.cancel()
    wait_for(qapp, lambda: not shell.ctx.queue_controller.executing)


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


def test_detail_editor_becomes_read_only_if_queued_row_starts(shell, qapp):
    backend = shell.ctx.backend
    first = add_queue_item(backend, epochs=160)
    second = add_queue_item(backend, epochs=100)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
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
    assert wait_for(qapp, lambda: backend.get_experiment(second.experiment_id).status == "running")
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
    first_job = shell.ctx.jobs.find(f"training:{first.experiment_id}")
    if first_job:
        first_job.cancel()
    second_job = shell.ctx.jobs.find(f"training:{second.experiment_id}")
    if second_job:
        second_job.cancel()


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


def test_experiment_queue_copy_refreshes_open_training_identifier(shell, monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    training = shell.page(PageId.TRAINING)
    shell.navigate(PageId.TRAINING)
    stale = training.experiment_id.text()
    shell.navigate(PageId.EXPERIMENTS)
    experiments = shell.page(PageId.EXPERIMENTS)
    source = shell.ctx.backend.list_experiments()[0]
    row = next(
        row
        for row in range(experiments.table.rowCount())
        if experiments.table.item(row, 1).text() == source.experiment_id
    )
    experiments.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
    experiments.action_map["queue_copy"].trigger()
    queued = shell.ctx.backend.list_training_queue()[-1]
    assert queued.experiment_id == stale
    assert training.experiment_id.text() != queued.experiment_id
    assert training.experiment_id.text() == shell.ctx.backend.next_experiment_id()


def test_stop_reservation_is_cleared_after_current_training(shell, qapp):
    first = add_queue_item(shell.ctx.backend, epochs=120)
    second = add_queue_item(shell.ctx.backend, epochs=10)
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


def test_experiment_list_hides_terminal_progress_and_localizes_epoch(shell):
    experiment = add_queue_item(shell.ctx.backend)
    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)

    experiment.status = "running"
    experiment.phase = "cross_validation"
    experiment.current_epoch = 2
    page.refresh()
    row = next(
        row
        for row in range(page.table.rowCount())
        if page.table.item(row, 1).text() == experiment.experiment_id
    )
    assert "エポック 2/" in page.table.item(row, 7).text()
    assert "epoch" not in page.table.item(row, 7).text()

    experiment.status = "completed"
    page.refresh()
    row = next(
        row
        for row in range(page.table.rowCount())
        if page.table.item(row, 1).text() == experiment.experiment_id
    )
    assert page.table.item(row, 7).text() == "—"


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


def test_error_background_and_inapplicable_values_are_visible(shell):
    backend = shell.ctx.backend
    invalid_config = backend.default_experiment_config("mask_rcnn")
    invalid_config["experiment"]["id"] = backend.next_experiment_id()
    invalid_config["training"]["epochs"] = 0
    invalid = backend.add_training_queue_item(invalid_config)
    cellpose = add_queue_item(backend, model="cellpose")
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    invalid_row = queue.model.row_for_id(invalid.experiment_id)
    cellpose_row = queue.model.row_for_id(cellpose.experiment_id)
    assert queue.model.data(queue.model.index(invalid_row, 1), Qt.ItemDataRole.DisplayRole) == (
        "設定エラー"
    )
    background = queue.model.data(queue.model.index(invalid_row, 0), Qt.ItemDataRole.BackgroundRole)
    assert background.name().lower() == Color.ERROR_BG.lower()
    weights_column = queue.model.column_for_path("model.pretrained_weights")
    weights = queue.model.index(cellpose_row, weights_column)
    assert queue.model.data(weights, Qt.ItemDataRole.DisplayRole) == "—"
    assert queue.model.data(weights, Qt.ItemDataRole.BackgroundRole).name().lower() == (
        Color.IDLE_BG.lower()
    )


def test_queue_choice_labels_match_training_form_and_expand_short_class_codes(shell):
    from foam_cell_analysis.gui.labels import classification_label

    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["experiment"]["id"] = backend.next_experiment_id()
    config["data"]["classification"] = "C"
    config["data"]["quality_filter"] = "good_only"
    item = backend.add_training_queue_item(config)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    training = shell.page(PageId.TRAINING)
    row = queue.model.row_for_id(item.experiment_id)
    for path in (
        "data.quality_filter",
        "model.pretrained_weights",
        "augmentation.profile",
    ):
        value = item.config.values
        for part in path.split("."):
            value = value[part]
        combo = training.fields.get(path) or training._model_widgets["mask_rcnn"][path]
        queue_value = queue.model.data(
            queue.model.index(row, queue.model.column_for_path(path)),
            Qt.ItemDataRole.DisplayRole,
        )
        assert queue_value == combo.itemText(combo.findData(value))
    assert classification_label("C") == "分類C"
    assert (
        queue.model.data(
            queue.model.index(row, queue.model.column_for_path("data.classification")),
            Qt.ItemDataRole.DisplayRole,
        )
        == "分類C"
    )
