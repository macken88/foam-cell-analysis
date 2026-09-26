"""モデル学習ページと拡張プロファイルの画面テスト。"""

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from foam_cell_analysis.gui.modes.training.augmentation_dialog import AugmentationDialog
from foam_cell_analysis.gui.navigation import PageId


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


def test_training_return_allocates_new_experiment_id(shell):
    page = shell.page(PageId.TRAINING)
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

    page.model_type.setCurrentIndex(1)

    assert page.config["model"]["type"] == "cellpose"
    assert "model.pretrained_model" in page._model_widgets["cellpose"]
    assert "model.backbone" not in page._model_widgets["cellpose"]
    page.model_type.setCurrentIndex(0)
    assert page.config["model"]["type"] == "mask_rcnn"
    assert "model.backbone" in page._model_widgets["mask_rcnn"]


def test_used_augmentation_profile_is_saved_as_new_version(mock_backend):
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
