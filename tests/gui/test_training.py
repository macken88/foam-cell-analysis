"""モデル学習ページと拡張プロファイルの画面テスト。"""

import copy
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QLineEdit, QMessageBox

from foam_cell_analysis.gui.modes.training.augmentation_dialog import AugmentationDialog
from foam_cell_analysis.gui.navigation import ModeId, PageId
from foam_cell_analysis.gui.theme import numeric_font


def test_training_start_navigates_and_records_final_model(shell, qapp):
    page = shell.page(PageId.TRAINING)
    assert f"id: {page.experiment_id.text()}" in page.yaml_preview.toPlainText()
    assert "used_item_ids: 学習開始時に確定" in page.yaml_preview.toPlainText()
    assert "data.used_item_ids" not in page.fields
    page.fields["training.epochs"].setValue(2)
    page.fields["checkpoint.validation_interval"].setValue(1)

    experiment_id = page.start_training(confirm=False)

    assert experiment_id
    assert shell.current_page() is shell.page(PageId.EXPERIMENTS)
    experiment = shell.ctx.backend.get_experiment(experiment_id)
    assert experiment.status == "running"
    for _ in range(100):
        qapp.processEvents()
        experiment = shell.ctx.backend.get_experiment(experiment_id)
        if experiment.status == "completed":
            break
        QTest.qWait(2)
    assert experiment.status == "completed"
    assert any(checkpoint.name == "final.pt" for checkpoint in experiment.checkpoints)
    assert experiment.oof_evaluation is not None
    assert experiment.selected_epoch is not None
    assert page.experiment_id.text() != experiment_id


def test_cv_fold_spin_updates_yaml_and_estimate_through_keyboard(shell, qapp):
    from PySide6.QtCore import Qt

    page = shell.page(PageId.TRAINING)
    folds = page.fields["data.cv.n_folds"]
    folds.setFocus()
    QTest.keyClick(folds, Qt.Key.Key_Up)
    qapp.processEvents()
    assert "n_folds: 6" in page.yaml_preview.toPlainText()
    assert "6 分割" in page.estimate_label.text()


def test_qtest_training_click_finishes_cv_and_final_model(shell, qapp, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog

    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    page = shell.page(PageId.TRAINING)
    page.fields["training.epochs"].setValue(2)
    page.fields["checkpoint.validation_interval"].setValue(1)
    QTest.mouseClick(page.start_button, Qt.MouseButton.LeftButton)

    assert shell.current_page() is shell.page(PageId.EXPERIMENTS)
    experiment = next(
        item
        for item in shell.ctx.backend.list_experiments()
        if item.status == "running" and item.experiment_id.startswith("exp_00")
    )
    for _ in range(100):
        qapp.processEvents()
        experiment = shell.ctx.backend.get_experiment(experiment.experiment_id)
        if experiment.status == "completed":
            break
        QTest.qWait(2)
    assert experiment.status == "completed"
    assert experiment.oof_evaluation is not None
    assert experiment.selected_epoch is not None
    assert any(checkpoint.name == "final.pt" for checkpoint in experiment.checkpoints)

    results = shell.page(PageId.EXPERIMENTS)
    row = next(
        row
        for row in range(results.table.rowCount())
        if results.table.item(row, 1).text() == experiment.experiment_id
    )
    results.table.setCurrentCell(row, 1)
    results.refresh()
    assert results.details.tabText(0) == "概要"
    assert results.oof_table.item(0, 1).text() != "—"

    monkeypatch.setattr(
        "foam_cell_analysis.gui.modes.training.experiment_list.SendToCandidatesDialog.exec",
        lambda _dialog: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        "foam_cell_analysis.gui.modes.comparison.candidates_page.CandidateDialog.exec",
        lambda _dialog: QDialog.DialogCode.Accepted,
    )
    QTest.mouseClick(results.button_map["send"], Qt.MouseButton.LeftButton)
    qapp.processEvents()
    candidates = shell.page(PageId.CANDIDATES)
    qapp.processEvents()
    candidate = next(
        item
        for item in shell.ctx.backend.list_candidates()
        if item.experiment_id == experiment.experiment_id
    )
    candidate_row = next(
        row
        for row in range(candidates.table.rowCount())
        if candidates.table.item(row, 1).text() == candidate.candidate_id
    )
    assert candidates.table.horizontalHeaderItem(6).text() == "検証 mAP"
    assert candidates.table.horizontalHeaderItem(7).text() == "OOF mAP"
    assert candidates.table.item(candidate_row, 7).text() != "—"


def test_training_return_allocates_new_experiment_id(shell, monkeypatch, qapp):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    page = shell.page(PageId.TRAINING)
    page.fields["training.epochs"].setValue(2)
    first_id = page.start_training(confirm=False)

    assert first_id
    first = shell.ctx.backend.get_experiment(first_id)
    next_id = page.experiment_id.text()
    assert next_id != first_id

    shell.navigate(PageId.TRAINING)
    assert page.experiment_id.text() == next_id
    second_id = page.start_training(confirm=False)

    assert second_id == next_id
    assert shell.ctx.backend.get_experiment(first_id) is first
    assert len([job for job in shell.ctx.jobs.training_jobs]) == 1
    queued = shell.ctx.backend.get_experiment(second_id)
    assert queued.status == "queued"
    shell.ctx.jobs.find(f"training:{first_id}").cancel()
    for _ in range(500):
        qapp.processEvents()
        if not shell.ctx.queue_controller.executing:
            break
        QTest.qWait(2)
    assert queued.status == "completed"


def test_existing_non_draft_experiment_id_is_rejected(shell, monkeypatch):
    page = shell.page(PageId.TRAINING)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: QMessageBox.StandardButton.Ok)
    page.experiment_id.setText("exp_0042")

    assert page.start_training(confirm=False) is None
    assert shell.ctx.backend.get_experiment("exp_0042").status != "running"


def test_training_options_refresh_and_preserve_selection(shell):
    page = shell.page(PageId.TRAINING)
    dataset_box = page.fields["data.dataset_version"]
    previous_dataset = dataset_box.currentData()
    profile_box = page.fields["augmentation.profile"]
    previous_profile = profile_box.currentData()
    created = shell.ctx.backend.save_augmentation_profile(
        shell.ctx.backend.get_augmentation_profile(previous_profile)
    )

    shell.navigate(PageId.EXPERIMENTS)
    shell.navigate(PageId.TRAINING)

    assert dataset_box.currentData() == previous_dataset
    assert profile_box.currentData() == previous_profile
    assert profile_box.findData(created.profile_id) >= 0


def test_model_type_switch_changes_model_specific_controls(shell):
    page = shell.page(PageId.TRAINING)

    page.model_type.setFocus()
    QTest.keyClick(page.model_type, Qt.Key.Key_End)

    assert page.config["model"]["type"] == "cellpose"
    assert "model.pretrained_model" in page._model_widgets["cellpose"]
    assert "model.backbone" not in page._model_widgets["cellpose"]
    page.model_type.setFocus()
    QTest.keyClick(page.model_type, Qt.Key.Key_Home)
    assert page.config["model"]["type"] == "mask_rcnn"
    assert "model.backbone" in page._model_widgets["mask_rcnn"]


def test_model_switch_updates_only_untouched_shared_defaults(shell, qapp):
    page = shell.page(PageId.TRAINING)
    page.model_type.setFocus()
    QTest.keyClick(page.model_type, Qt.Key.Key_End)
    qapp.processEvents()
    assert page.config["training"]["batch_size"] == 1
    assert page.config["training"]["learning_rate"] == 1e-5
    assert page.config["training"]["weight_decay"] == 0.1

    page.fields["training.learning_rate"].setValue(0.0003)
    page.model_type.setFocus()
    QTest.keyClick(page.model_type, Qt.Key.Key_Home)
    qapp.processEvents()
    assert page.config["training"]["learning_rate"] == 0.0003
    assert page.config["training"]["batch_size"] == 2
    assert page.config["training"]["weight_decay"] == 0.0001
    assert "既定値と異なる値を保持しました" in shell.status_text.text()


def test_mask_rcnn_normalization_is_read_only_and_tracks_pretrained_weights(shell, qapp):
    """重み選択を操作すると固定正規化値が表示と YAML に反映される。"""
    from PySide6.QtCore import Qt

    page = shell.page(PageId.TRAINING)
    weights = page._model_widgets["mask_rcnn"]["model.pretrained_weights"]
    weights.setFocus()
    QTest.keyClick(weights, Qt.Key.Key_End)
    qapp.processEvents()

    assert weights.currentData() == "imagenet"
    assert "model.input.image_mean" not in page._model_widgets["mask_rcnn"]
    assert "model.input.image_std" not in page._model_widgets["mask_rcnn"]
    assert page.model_normalization_note.font().family() == numeric_font().family()
    assert "0.485, 0.456, 0.406" in page.model_normalization_note.text()
    assert "0.229, 0.224, 0.225" in page.model_normalization_note.text()
    yaml = page.yaml_preview.toPlainText()
    assert "image_mean:\n    - 0.485\n    - 0.456\n    - 0.406" in yaml
    assert "image_std:\n    - 0.229\n    - 0.224\n    - 0.225" in yaml


def test_backend_warns_when_api_config_overrides_weight_normalization(mock_backend):
    """API から異なる正規化値が渡されたとき検証警告を返す。"""
    config = mock_backend.default_experiment_config("mask_rcnn")
    config["model"]["input"]["image_mean"] = [0.1, 0.2, 0.3]
    warnings = mock_backend.validate_experiment_config(config)

    assert any(
        item["level"] == "warning" and "画像平均・標準偏差" in item["message"] for item in warnings
    )


@pytest.mark.slow
def test_qtest_final_training_uses_selected_epoch_for_first_run_and_retry(shell, qapp, monkeypatch):
    """選択エポックを最終学習・進捗・再試行の終了条件に使う。"""
    from PySide6.QtCore import Qt

    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    page = shell.page(PageId.TRAINING)
    page.fields["data.seed"].setValue(5)
    page.fields["training.epochs"].setValue(100)
    page.fields["checkpoint.validation_interval"].setValue(1)
    QTest.mouseClick(page.start_button, Qt.MouseButton.LeftButton)
    results = shell.page(PageId.EXPERIMENTS)
    experiment = next(
        item for item in shell.ctx.backend.list_experiments() if item.status == "running"
    )
    for _ in range(5000):
        qapp.processEvents()
        if experiment.phase == "final_training":
            break
        QTest.qWait(2)
    assert experiment.phase == "final_training"
    expected_steps = 5 * experiment.total_epochs + experiment.selected_epoch
    first_job = shell.ctx.jobs.find(f"training:{experiment.experiment_id}")
    assert first_job.total_steps == expected_steps
    results.refresh()
    assert (
        results.table.item(
            next(
                row
                for row in range(results.table.rowCount())
                if results.table.item(row, 1).text() == experiment.experiment_id
            ),
            7,
        )
        .text()
        .startswith("最終学習・エポック ")
    )
    assert (
        results.table.item(
            next(
                row
                for row in range(results.table.rowCount())
                if results.table.item(row, 1).text() == experiment.experiment_id
            ),
            7,
        )
        .text()
        .endswith(f"/{experiment.selected_epoch}")
    )

    results.table.setCurrentCell(
        next(
            row
            for row in range(results.table.rowCount())
            if results.table.item(row, 1).text() == experiment.experiment_id
        ),
        1,
    )
    results.action_map["stop"].trigger()
    assert experiment.status == "stopped"
    results.action_map["retry"].trigger()
    for _ in range(5000):
        qapp.processEvents()
        if experiment.status == "completed":
            break
        QTest.qWait(2)
    assert experiment.status == "completed"
    assert experiment.final_history[-1].epoch == experiment.selected_epoch
    assert len(experiment.final_history) == experiment.selected_epoch
    final = next(item for item in experiment.checkpoints if item.name == "final.pt")
    assert final.epoch == experiment.selected_epoch


def test_epoch_selection_is_read_only_oof_map_maximum(shell, qapp, monkeypatch):
    """学習フォームは OOF mAP 最大固定を表示し、設定へ旧方向キーを出さない。"""
    from PySide6.QtCore import Qt

    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    page = shell.page(PageId.TRAINING)
    assert "checkpoint.best_mode" not in page.fields
    selection_label = next(
        label
        for label in page.findChildren(QLabel)
        if label.text() == "エポック選択の指標：OOF 平均適合率（AP）・最大"
    )
    assert selection_label
    assert "checkpoint.best_metric" not in page.fields
    assert "best_metric: oof_instance_map" in page.yaml_preview.toPlainText()
    assert "best_mode" not in page.yaml_preview.toPlainText()
    page.fields["training.epochs"].setValue(3)
    page.fields["checkpoint.validation_interval"].setValue(1)
    QTest.mouseClick(page.start_button, Qt.MouseButton.LeftButton)
    experiment_id = next(
        item.experiment_id
        for item in shell.ctx.backend.list_experiments()
        if item.status == "running"
    )
    for _ in range(1000):
        qapp.processEvents()
        experiment = shell.ctx.backend.get_experiment(experiment_id)
        if experiment.status == "completed":
            break
        QTest.qWait(2)
    assert experiment.status == "completed"
    assert experiment.config.values["checkpoint"]["best_metric"] == "oof_instance_map"
    assert "best_mode" not in experiment.config.values["checkpoint"]
    assert (
        experiment.selected_epoch == max(experiment.oof_history, key=lambda point: point.map).epoch
    )


def test_legacy_draft_edit_and_copy_migrate_via_experiment_actions(shell, qapp):
    """下書き編集と複製で旧キーを移行してからフォームを組み立てる。"""
    from PySide6.QtCore import Qt

    backend = shell.ctx.backend

    def legacy_config():
        config = backend.default_experiment_config("mask_rcnn")
        config["data"].pop("cv")
        config["data"]["split_id"] = "split_001"
        config["checkpoint"].pop("save_fold_models")
        config["checkpoint"]["save_best"] = True
        config["checkpoint"]["save_last"] = True
        return config

    edited = backend.save_experiment_draft(legacy_config())
    copied = backend.save_experiment_draft(legacy_config())

    def store_legacy(experiment):
        config = experiment.config.values
        config["data"].pop("cv")
        config["data"]["split_id"] = "split_001"
        config["checkpoint"].pop("save_fold_models")
        config["checkpoint"]["save_best"] = True
        config["checkpoint"]["save_last"] = True

    store_legacy(edited)
    store_legacy(copied)
    shell.navigate(PageId.EXPERIMENTS)
    results = shell.page(PageId.EXPERIMENTS)
    results.refresh()

    def select(experiment_id):
        row = next(
            row
            for row in range(results.table.rowCount())
            if results.table.item(row, 1).text() == experiment_id
        )
        rect = results.table.visualItemRect(results.table.item(row, 1))
        QTest.mouseClick(results.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
        return row

    row = select(edited.experiment_id)
    results.table.setCurrentCell(row, 1)
    results.action_map["edit"].trigger()
    page = shell.page(PageId.TRAINING)
    assert "split_id" not in page.config["data"]
    assert page.config["data"]["cv"]["n_folds"] == 5
    assert page.config["checkpoint"]["save_fold_models"] is True
    assert page.normalization_reset_note.text().find("旧形式") >= 0
    assert "split_001" not in page.yaml_preview.toPlainText()
    assert "設定を入力してください" not in page.yaml_preview.toPlainText()
    assert page._read_control("model.rescale", QLineEdit("auto")) == "auto"

    shell.navigate(PageId.EXPERIMENTS)
    row = select(copied.experiment_id)
    results.table.setCurrentCell(row, 1)
    results.action_map["copy"].trigger()
    copied_page = shell.page(PageId.TRAINING)
    assert "split_id" not in copied_page.config["data"]
    assert copied_page.config["data"]["cv"]["n_folds"] == 5
    assert copied_page.config["checkpoint"]["save_fold_models"] is True


def test_legacy_stopped_experiment_retry_is_blocked_without_mutating_record(
    shell, qapp, monkeypatch
):
    """旧形式の再試行を案内し、保存済み設定を維持する。"""
    from PySide6.QtCore import Qt

    experiment = shell.ctx.backend.get_experiment("exp_0044")
    legacy = experiment.config.values
    legacy["data"].pop("cv", None)
    legacy["data"]["split_id"] = "split_001"
    legacy["checkpoint"].pop("save_fold_models", None)
    legacy["checkpoint"]["save_best"] = True
    legacy["checkpoint"]["save_last"] = True
    legacy["training"]["epochs"] = 2
    experiment.total_epochs = 2
    experiment.status = "stopped"
    shell.navigate(PageId.EXPERIMENTS)
    results = shell.page(PageId.EXPERIMENTS)
    results.refresh()
    row = next(
        row
        for row in range(results.table.rowCount())
        if results.table.item(row, 1).text() == experiment.experiment_id
    )
    rect = results.table.visualItemRect(results.table.item(row, 1))
    QTest.mouseClick(results.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    results.table.setCurrentCell(row, 1)
    saved_config = copy.deepcopy(experiment.config.values)
    notices = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: notices.append(message),
    )
    results.action_map["retry"].trigger()

    assert notices == [
        "この実験は旧形式の設定で記録されているため再試行できません。"
        "『設定を複製して新規実験』で、現在の形式に移した設定から始めてください。"
    ]
    assert experiment.config.values == saved_config
    assert experiment.status == "stopped"


def test_used_augmentation_profile_is_saved_as_new_version(mock_backend, qapp):
    qapp.processEvents()
    source = mock_backend.get_augmentation_profile("aug_v003")
    original_probability = source.transforms[0].probability
    dialog = AugmentationDialog(mock_backend, "aug_v003")
    dialog.eight.setChecked(True)
    assert len(dialog._preview_images) == 8
    dialog.controls[source.transforms[0].key][1].setValue(0.8)

    saved = mock_backend.save_augmentation_profile(dialog.build_profile())

    assert saved.profile_id == "aug_v004"
    assert saved.base_profile == "aug_v003"
    assert saved.transforms[0].probability == 0.8
    assert (
        mock_backend.get_augmentation_profile("aug_v003").transforms[0].probability
        == original_probability
    )
    assert "exp_0042" in mock_backend.get_augmentation_profile("aug_v003").used_by_experiments


def test_augmentation_manual_order_collision_keeps_dense_pipeline(shell, qapp):
    dialog = AugmentationDialog(shell.ctx.backend, "aug_v001")
    dialog.show()
    qapp.processEvents()

    enabled = dialog.controls["horizontal_flip"][0]
    QTest.mouseClick(enabled, Qt.MouseButton.LeftButton)
    checked = {key for key, (checkbox, *_rest) in dialog.controls.items() if checkbox.isChecked()}
    assert sorted(dialog.order_controls[key].value() for key in checked) == list(
        range(1, len(checked) + 1)
    )
    unchecked = set(dialog.order_controls) - checked
    assert all(dialog.order_controls[key].value() == 0 for key in unchecked)
    assert all(not dialog.order_controls[key].isEnabled() for key in unchecked)

    moved = dialog.order_controls["rotation"]
    moved.setFocus()
    QTest.keyClick(moved, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(moved, "1")
    QTest.keyClick(moved, Qt.Key.Key_Return)
    qapp.processEvents()
    assert moved.value() == 1
    assert sorted(dialog.order_controls[key].value() for key in checked) == list(
        range(1, len(checked) + 1)
    )

    profile = dialog.build_profile()
    enabled_keys = [item.key for item in profile.transforms if item.enabled]
    assert profile.order[: len(enabled_keys)] == sorted(
        enabled_keys, key=lambda key: dialog.order_controls[key].value()
    )
    dialog.close()


def test_augmentation_preview_uses_selected_released_training_version(qapp, tmp_path):
    import shutil

    from foam_cell_analysis.gui.modes.training.augmentation_dialog import AugmentationDialog
    from foam_cell_analysis.services.hybrid_backend import HybridBackend
    from tests.training.test_training_process import _workspace

    _workspace(tmp_path, n_items=4)
    shutil.rmtree(tmp_path / "experiments")
    backend = HybridBackend(tmp_path)
    dialog = AugmentationDialog(backend, "aug_v001")

    assert dialog.dataset.currentText() == "train_v000"
    assert dialog.items
    assert dialog.sample.currentText() in {item.item_id for item in dialog.items}
    assert any(
        label.text() == "簡易プレビュー（学習時の変換とは一致しません）"
        for label in dialog.findChildren(QLabel)
    )
    selected = next(item for item in dialog.items if item.item_id == dialog.sample.currentText())
    expected = backend.get_dataset_item_image("train_v000", selected.item_id, selected.channels[0])
    assert expected.shape == dialog._preview_images[0].shape
    QTest.mouseClick(dialog.random_sample, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert dialog.sample.currentText() in {item.item_id for item in dialog.items}
    dialog.close()


def test_training_summary_tracks_dataset_cv_model_and_epochs(shell, qapp):
    page = shell.page(PageId.TRAINING)
    assert hasattr(page, "summary_label")
    assert page.summary_label.text().find("データ") >= 0
    dataset = page.fields["data.dataset_version"]
    dataset.setCurrentIndex(min(1, dataset.count() - 1))
    folds = page.fields["data.cv.n_folds"]
    folds.setValue(folds.value() + 1)
    epochs = page.fields["training.epochs"]
    epochs.setValue(epochs.value() + 1)
    page.model_type.setCurrentIndex(1)
    qapp.processEvents()
    assert dataset.currentText() in page.summary_label.text()
    assert f"{folds.value()} 分割" in page.summary_label.text()
    assert f"{epochs.value()} エポック" in page.summary_label.text()
    assert "Cellpose" in page.summary_label.text()


def test_training_yaml_preview_toggles_and_persists(shell, qapp):
    from foam_cell_analysis.gui.settings import app_settings

    page = shell.page(PageId.TRAINING)
    assert page.preview_panel.isHidden()
    QTest.mouseClick(page.preview_button, Qt.MouseButton.LeftButton)
    assert not page.preview_panel.isHidden()
    assert page.preview_action.isChecked()
    assert page.preview_button.isChecked()
    page.preview_action.setChecked(False)
    assert page.preview_panel.isHidden()
    assert not page.preview_button.isChecked()
    assert app_settings().value("training/yamlPreview", False, type=bool) is False


def test_training_anchor_fields_remain_editable_and_sync_yaml(shell, qapp):
    page = shell.page(PageId.TRAINING)
    page.preview_action.setChecked(True)
    sizes = page._model_widgets["mask_rcnn"]["model.anchors.sizes"]
    assert isinstance(sizes, QLineEdit)
    sizes.setText("16, 32, 64")
    qapp.processEvents()
    assert "sizes:\n    - 16\n    - 32\n    - 64" in page.yaml_preview.toPlainText()


def test_training_validation_warning_is_clickable_and_clears_after_edit(shell, monkeypatch):
    page = shell.page(PageId.TRAINING)
    dialogs = []
    monkeypatch.setattr(
        "foam_cell_analysis.gui.modes.training.page.QMessageBox.information",
        lambda _parent, title, text: dialogs.append((title, text)),
    )
    warning = {"level": "warning", "message": "確認用の警告"}
    error = {"level": "error", "message": "確認用のエラー"}
    results = [warning, error]
    monkeypatch.setattr(shell.ctx.backend, "validate_experiment_config", lambda _config: results)

    assert page.validate_config() == results
    assert not page.validation_result_button.isHidden()
    assert page.validation_result_button.text() == "⚠ 警告 1 件　⚠ エラー 1 件"
    dialogs.clear()
    QTest.mouseClick(page.validation_result_button, Qt.MouseButton.LeftButton)
    assert dialogs == [("設定の検証", "warning: 確認用の警告\nerror: 確認用のエラー")]

    page.fields["training.epochs"].setValue(page.fields["training.epochs"].value() + 1)
    assert page.validation_result_button.isHidden()
    assert page._validation_results == []


def _wheel(widget, delta=-120):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    center = QPointF(widget.rect().center())
    event = QWheelEvent(
        center,
        QPointF(widget.mapToGlobal(widget.rect().center())),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def test_wheel_never_changes_inputs_and_number_fields_have_no_arrows(shell, qapp):
    from PySide6.QtWidgets import QAbstractSpinBox

    from foam_cell_analysis.gui.navigation import ModeId

    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    window = shell.manager.window(ModeId.TRAINING)
    window.resize(1000, 600)
    window.show()
    qapp.processEvents()
    folds = page.fields["data.cv.n_folds"]
    epochs = page.fields["training.epochs"]
    learning_rate = page.fields["training.learning_rate"]
    before = page._collect_config()
    model_index = page.model_type.currentIndex()
    scroll = page.scroll.verticalScrollBar()
    scroll.setValue(0)
    for widget in (folds, epochs, learning_rate):
        assert widget.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
        widget.setFocus()
        qapp.processEvents()
        _wheel(widget.lineEdit(), -120)
        _wheel(widget, -120)
    # 値は変わらず、ホイールは外側のフォームをスクロールする
    assert scroll.maximum() == 0 or scroll.value() > 0
    page.model_type.setFocus()
    _wheel(page.model_type, -120)
    qapp.processEvents()

    after = page._collect_config()
    assert after["data"]["cv"]["n_folds"] == before["data"]["cv"]["n_folds"]
    assert after["training"]["epochs"] == before["training"]["epochs"]
    assert after["training"]["learning_rate"] == before["training"]["learning_rate"]
    assert page.model_type.currentIndex() == model_index
    window.hide()


def test_learning_rate_accepts_exponent_input_in_form_and_queue_table(shell, qapp):
    from PySide6.QtCore import Qt

    backend = shell.ctx.backend
    page = shell.page(PageId.TRAINING)
    field = page.fields["training.learning_rate"]
    assert "e-" in field.text()
    field.setFocus()
    field.lineEdit().selectAll()
    QTest.keyClicks(field.lineEdit(), "1.0e-5")
    QTest.keyClick(field.lineEdit(), Qt.Key.Key_Return)
    assert page._collect_config()["training"]["learning_rate"] == 1e-5
    assert field.text() == "1e-05"

    queued = backend.get_experiment(page.enqueue_config())
    assert queued.config.values["training"]["learning_rate"] == 1e-5
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    index = queue.model.index(0, queue.model.column_for_path("training.learning_rate"))
    assert index.data() == "1e-05"
    queue.table.setCurrentIndex(index)
    queue.table.edit(index)
    qapp.processEvents()
    editor = queue.table.findChild(type(field))
    editor.lineEdit().selectAll()
    QTest.keyClicks(editor.lineEdit(), "2.5e-4")
    QTest.keyClick(editor.lineEdit(), Qt.Key.Key_Return)
    qapp.processEvents()
    assert backend.get_experiment(queued.experiment_id).config.values["training"][
        "learning_rate"
    ] == pytest.approx(2.5e-4)
    assert queue.model.index(0, index.column()).data() == "2.5e-04"


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


def test_start_training_mouse_click_keeps_confirmation(shell, qapp):
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as ask:
        QTest.mouseClick(page.start_button, Qt.MouseButton.LeftButton)
        qapp.processEvents()
    assert ask.call_count == 1
    assert shell.ctx.jobs.running_count == 0


def test_missing_dataset_version_is_a_validation_error_and_value_error(shell):
    backend = shell.ctx.backend
    config = backend.default_experiment_config("mask_rcnn")
    config["data"]["dataset_version"] = "missing_train_v999"
    issues = backend.validate_experiment_config(config)
    assert any(issue["level"] == "error" for issue in issues)
    with pytest.raises(ValueError, match="missing_train_v999"):
        backend.estimate_training_items(dataset_version="missing_train_v999")


def test_training_runner_owns_the_single_training_slot(shell):
    backend = shell.ctx.backend
    experiment = backend.start_training(backend.default_experiment_config("mask_rcnn"))
    shell.ctx.training_runner.start(experiment.experiment_id)

    with pytest.raises(RuntimeError, match="別の学習"):
        shell.ctx.training_runner.start("exp_other")

    assert shell.ctx.jobs.has_training_job
    assert shell.ctx.jobs.jobs() == []
    shell.ctx.training_runner.request_stop()
