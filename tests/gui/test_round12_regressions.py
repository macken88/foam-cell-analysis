"""Round 12 regression coverage for the queue table, delayed validation, and review findings."""

import copy
import re

import pytest
import yaml
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView, QDoubleSpinBox, QSpinBox, QTableView

from foam_cell_analysis.gui.modes.training.dialogs import ExperimentCompareDialog
from foam_cell_analysis.gui.navigation import PageId
from foam_cell_analysis.gui.theme import Color


def test_queue_uses_delegate_editors_not_permanent_cell_widgets(shell):
    shell.navigate(PageId.TRAINING_QUEUE)
    page = shell.page(PageId.TRAINING_QUEUE)
    assert isinstance(page.table, QTableView)
    assert page.table.editTriggers() == (
        QAbstractItemView.EditTrigger.SelectedClicked
        | QAbstractItemView.EditTrigger.EditKeyPressed
        | QAbstractItemView.EditTrigger.AnyKeyPressed
    )
    assert page.model is page.table.model()


def test_queue_boolean_is_an_unlabelled_check_state(shell):
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["experiment"]["id"] = backend.next_experiment_id()
    row = backend.add_training_queue_item(config)
    shell.navigate(PageId.TRAINING_QUEUE)
    page = shell.page(PageId.TRAINING_QUEUE)
    column = page.model.column_for_path("data.cv.stratify_by_classification")
    index = page.model.index(0, column)
    assert page.model.data(index, Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
    assert page.model.data(index, Qt.ItemDataRole.DisplayRole) in (None, "")
    rect = page.table.visualRect(index)
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert (
        backend.get_experiment(row.experiment_id).config.values["data"]["cv"][
            "stratify_by_classification"
        ]
        is False
    )


def test_queue_live_progress_keeps_active_numeric_editor_open(shell, qapp):
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
    controller = shell.ctx.queue_controller
    controller.start()
    assert controller.active_id == first.experiment_id
    index = page.model.index(1, page.model.column_for_path("training.learning_rate"))
    QTest.mouseClick(
        page.table.viewport(), Qt.MouseButton.LeftButton, pos=page.table.visualRect(index).center()
    )
    QTest.keyClick(page.table, Qt.Key.Key_F2)
    qapp.processEvents()
    editor = page.table.findChild(QDoubleSpinBox)
    assert editor is not None and editor.isVisible()
    for _ in range(3):
        controller.changed.emit()
        qapp.processEvents()
    assert editor.isVisible()
    editor.setValue(0.00037)
    QTest.keyClick(editor, Qt.Key.Key_Enter)
    qapp.processEvents()
    assert backend.get_experiment(waiting.experiment_id).config.values["training"][
        "learning_rate"
    ] == pytest.approx(0.00037)
    controller.stop()
    job = shell.ctx.jobs.find(f"training:{first.experiment_id}")
    if job:
        job.cancel()
    qapp.processEvents()


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


def test_missing_dataset_version_is_a_validation_error_and_value_error(shell):
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["data"]["dataset_version"] = "missing_train_v999"
    issues = backend.validate_experiment_config(config)
    assert any(issue["level"] == "error" for issue in issues)
    with pytest.raises(ValueError, match="missing_train_v999"):
        backend.estimate_training_items(dataset_version="missing_train_v999")


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


def test_working_table_preserves_saved_widths_when_one_column_is_narrow(shell, qapp):
    from foam_cell_analysis.gui.modes.data_preparation.page import DataPreparationPage

    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    page.resize(1400, 800)
    qapp.processEvents()
    header = page.table.horizontalHeader()
    column = 0
    before = page.table.columnWidth(column)
    x = header.sectionViewportPosition(column) + before - 1
    point = QPoint(x, header.height() // 2)
    QTest.mousePress(header.viewport(), Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseMove(header.viewport(), QPoint(x + 50, point.y()), delay=20)
    QTest.mouseRelease(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(x + 50, point.y()))
    expected = page.table.columnWidth(column)
    header.resizeSection(page.model.columnCount() - 1, 19)
    reloaded = DataPreparationPage(shell.ctx)
    reloaded.show()
    qapp.processEvents()
    assert reloaded._has_saved_column_widths
    assert reloaded.table.columnWidth(column) == expected
    reloaded.close()


def test_compare_dialog_keeps_manually_resized_columns_after_filter_toggle(shell, qapp):
    experiments = shell.ctx.backend.list_experiments()[:2]
    dialog = ExperimentCompareDialog(experiments)
    dialog.show()
    qapp.processEvents()
    header = dialog.table.horizontalHeader()
    column = 1
    before = dialog.table.columnWidth(column)
    x = header.sectionViewportPosition(column) + before - 1
    point = QPoint(x, header.height() // 2)
    QTest.mousePress(header.viewport(), Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseMove(header.viewport(), QPoint(x + 70, point.y()), delay=20)
    QTest.mouseRelease(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(x + 70, point.y()))
    resized = dialog.table.columnWidth(column)
    QTest.mouseClick(dialog.differences_only, Qt.MouseButton.LeftButton)
    assert dialog.table.columnWidth(column) == resized
    dialog.close()


def test_training_spec_yaml_code_blocks_are_parseable():
    from pathlib import Path

    path = (
        Path(__file__).parents[2]
        / "docs/specs/foam_cell_segmentation_model_training_spec_ja_v2.html"
    )
    source = path.read_text(encoding="utf-8")
    blocks = re.findall(r"<pre>(.*?)</pre>", source, re.DOTALL)
    config_blocks = [block for block in blocks if "best_metric:" in block]
    assert config_blocks
    for block in config_blocks:
        yaml.safe_load(block)
