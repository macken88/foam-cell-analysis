"""Round 12 regression coverage for the queue table, delayed validation, and review findings."""

import copy

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QSpinBox

from foam_cell_analysis.gui.navigation import PageId
from foam_cell_analysis.gui.theme import Color


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
