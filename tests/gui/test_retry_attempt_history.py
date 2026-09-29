"""再試行を別の実行試行として保存する GUI 回帰テスト。"""

import copy

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from foam_cell_analysis.gui.navigation import PageId


def wait_for(qapp, condition, timeout=5000):
    elapsed = 0
    while elapsed < timeout:
        qapp.processEvents()
        if condition():
            return True
        QTest.qWait(5)
        elapsed += 5
    return condition()


def _select_experiment(page, experiment_id):
    row = next(
        row
        for row in range(page.table.rowCount())
        if page.table.item(row, 1).text() == experiment_id
    )
    page.table.setCurrentCell(row, 1)
    return row


def test_idle_retry_keeps_previous_attempt_results_visible(shell, monkeypatch, qapp):
    backend = shell.ctx.backend
    original = backend.get_experiment("exp_0044")
    previous_history = list(original.history)
    previous_folds = {fold: list(points) for fold, points in original.fold_histories.items()}
    previous_checkpoints = list(original.checkpoints)
    previous_config = original.config.to_yaml()
    previous_runs = len(original.runs)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    row = _select_experiment(page, original.experiment_id)
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.table.visualRect(page.table.model().index(row, 1)).center(),
    )
    page.action_map["retry"].trigger()

    assert len(original.runs) == previous_runs + 1
    assert original.runs[-2].history == previous_history
    assert original.runs[-2].fold_histories == previous_folds
    assert original.runs[-2].oof_evaluation is not None
    assert original.runs[-2].used_item_ids
    assert original.runs[-2].fold_assignments
    assert original.runs[-2].checkpoints == previous_checkpoints
    assert original.config.to_yaml() == previous_config

    page.refresh()
    row = _select_experiment(page, original.experiment_id)
    page._current_changed()
    assert page.run_table.rowCount() == previous_runs + 1
    assert page.run_table.item(previous_runs - 1, 4).text() == (
        f"{original.runs[-2].oof_evaluation.overall_map:.3f}"
    )
    assert page.run_table.item(previous_runs - 1, 5).text() != "—"
    assert page.table.item(row, 6).text() in {"実行中", "完了"}

    job = shell.ctx.jobs.find(f"training:{original.experiment_id}")
    if job:
        job.cancel()


def test_active_retry_is_read_only_reservation_for_same_experiment(shell, monkeypatch, qapp):
    backend = shell.ctx.backend
    original = backend.get_experiment("exp_0044")
    run_count = len(original.runs)
    ids_before = {item.experiment_id for item in backend.list_experiments()}
    other_config = backend.default_experiment_config("mask_rcnn")
    other_config["training"]["epochs"] = 120
    other = backend.add_training_queue_item(other_config)
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    assert wait_for(qapp, lambda: shell.ctx.jobs.has_training_job)

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    row = _select_experiment(page, original.experiment_id)
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.table.visualRect(page.table.model().index(row, 1)).center(),
    )
    page.action_map["retry"].trigger()

    queue = shell.page(PageId.TRAINING_QUEUE)
    queue.refresh()
    retry_row = next(
        row
        for row, entry in enumerate(queue.model.entries)
        if "再試行" in queue.model.data(queue.model.index(row, 2), Qt.ItemDataRole.DisplayRole)
    )
    retry_entry = queue.model.entries[retry_row]
    assert retry_entry.experiment_id == original.experiment_id
    assert queue.model.data(queue.model.index(retry_row, 2), Qt.ItemDataRole.DisplayRole) == (
        f"{original.experiment_id}（再試行 {run_count + 1}）"
    )
    assert not (queue.model.flags(queue.model.index(retry_row, 3)) & Qt.ItemFlag.ItemIsEditable)
    queue.table.selectRow(retry_row)
    assert not queue.edit_button.isEnabled()
    assert shell.ctx.backend.list_experiments()[-1].experiment_id == other.experiment_id
    assert {item.experiment_id for item in backend.list_experiments()} == ids_before | {
        other.experiment_id
    }

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    queue.queue_actions["delete"].trigger()
    assert backend.get_experiment(original.experiment_id) is original
    assert len(original.runs) == run_count

    for job in shell.ctx.jobs.training_jobs:
        job.cancel()


def test_active_retry_runs_as_another_attempt_without_new_experiment(shell, monkeypatch, qapp):
    backend = shell.ctx.backend
    original = backend.get_experiment("exp_0044")
    original.status = "stopped"
    # やり直しが最後まで走るのを待つので、短い学習にしておく
    original.config.values["training"]["epochs"] = 1
    run_count = len(original.runs)
    ids_before = {item.experiment_id for item in backend.list_experiments()}
    config = backend.default_experiment_config("mask_rcnn")
    config["training"]["epochs"] = 100
    active = backend.add_training_queue_item(config)
    queue = shell.page(PageId.TRAINING_QUEUE)
    shell.navigate(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    assert wait_for(qapp, lambda: shell.ctx.jobs.has_training_job)

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    row = _select_experiment(page, original.experiment_id)
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.table.visualRect(page.table.model().index(row, 1)).center(),
    )
    page.action_map["retry"].trigger()

    first_job = shell.ctx.jobs.find(f"training:{active.experiment_id}")
    assert first_job is not None
    first_job.cancel()
    assert wait_for(qapp, lambda: len(original.runs) == run_count + 1)
    assert wait_for(qapp, lambda: original.status == "completed")
    assert {item.experiment_id for item in backend.list_experiments()} == ids_before | {
        active.experiment_id
    }
    # 終わった行（元の学習とやり直しの予約）は、結果が実験一覧に残るためキューから外れる
    assert wait_for(qapp, lambda: backend.list_training_queue() == [])
    for job in shell.ctx.jobs.training_jobs:
        job.cancel()


def test_retry_candidate_keeps_source_attempt_and_oof_snapshot(shell, monkeypatch, qapp):
    backend = shell.ctx.backend
    experiment = backend.get_experiment("exp_0042")
    candidate = next(
        item for item in backend.list_candidates() if item.experiment_id == experiment.experiment_id
    )
    source_map = candidate.oof_evaluation.overall_map
    source_attempt = candidate.source_attempt_number
    source_reference = candidate.checkpoint_reference
    assert source_attempt == len(experiment.runs)
    assert "試行" in candidate.checkpoint_reference

    experiment.status = "stopped"
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    row = _select_experiment(page, experiment.experiment_id)
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.table.visualRect(page.table.model().index(row, 1)).center(),
    )
    page.action_map["retry"].trigger()

    assert candidate.oof_evaluation.overall_map == source_map
    assert candidate.source_attempt_number == source_attempt
    assert candidate.checkpoint_reference == source_reference
    released = next(
        model
        for model in backend.list_released_models()
        if model.candidate_id == candidate.candidate_id
    )
    assert released.source_attempt_number == source_attempt
    assert released.oof_evaluation.overall_map == source_map
    job = shell.ctx.jobs.find(f"training:{experiment.experiment_id}")
    if job:
        job.cancel()


def test_rereserving_after_cancel_keeps_retry_reservations_distinct(shell, monkeypatch, qapp):
    """予約を取り消して再予約しても、予約が重複せず、1 件の削除で 1 件だけ消える。"""
    backend = shell.ctx.backend
    original = backend.get_experiment("exp_0044")
    run_count = len(original.runs)
    active = backend.add_training_queue_item(backend.default_experiment_config("mask_rcnn"))
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    assert wait_for(qapp, lambda: shell.ctx.jobs.has_training_job)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    def reserve_retry():
        shell.navigate(PageId.EXPERIMENTS)
        page = shell.page(PageId.EXPERIMENTS)
        row = _select_experiment(page, original.experiment_id)
        QTest.mouseClick(
            page.table.viewport(),
            Qt.MouseButton.LeftButton,
            pos=page.table.visualRect(page.table.model().index(row, 1)).center(),
        )
        page.action_map["retry"].trigger()

    def retry_rows():
        queue.refresh()
        return [
            row
            for row, entry in enumerate(queue.model.entries)
            if entry.queue_is_retry and entry.status == "queued"
        ]

    def retry_labels():
        return [
            queue.model.data(queue.model.index(row, 2), Qt.ItemDataRole.DisplayRole)
            for row in retry_rows()
        ]

    reserve_retry()
    reserve_retry()
    shell.navigate(PageId.TRAINING_QUEUE)
    assert retry_labels() == [
        f"exp_0044（再試行 {run_count + 1}）",
        f"exp_0044（再試行 {run_count + 2}）",
    ]

    # 先の予約を取り消し、もう一度予約する
    queue.table.selectRow(retry_rows()[0])
    queue.queue_actions["delete"].trigger()
    reserve_retry()
    shell.navigate(PageId.TRAINING_QUEUE)
    ids = [queue.model.entries[row].queue_id for row in retry_rows()]
    assert len(ids) == len(set(ids)) == 2
    assert retry_labels() == [
        f"exp_0044（再試行 {run_count + 1}）",
        f"exp_0044（再試行 {run_count + 2}）",
    ]

    # 1 件だけ削除すると、もう 1 件は残る
    queue.table.selectRow(retry_rows()[0])
    queue.queue_actions["delete"].trigger()
    assert retry_labels() == [f"exp_0044（再試行 {run_count + 1}）"]
    assert len(original.runs) == run_count
    assert active.experiment_id in {entry.experiment_id for entry in backend.list_training_queue()}
    for job in shell.ctx.jobs.training_jobs:
        job.cancel()


def test_legacy_retry_api_rejects_without_migrating_existing_config(shell):
    backend = shell.ctx.backend
    experiment = backend.get_experiment("exp_0044")
    config = experiment.config.values
    config["data"].pop("cv", None)
    config["data"]["split_id"] = "split_001"
    config["checkpoint"].pop("save_fold_models", None)
    config["checkpoint"]["best_metric"] = "instance_map"
    config["checkpoint"]["best_mode"] = "min"
    experiment.status = "stopped"
    saved = copy.deepcopy(config)
    with pytest.raises(ValueError, match="旧形式"):
        backend.retry_experiment(experiment.experiment_id)
    assert experiment.config.values == saved
