"""モデル学習ページと拡張プロファイルの画面テスト。"""

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from foam_cell_analysis.gui.modes.training.augmentation_dialog import AugmentationDialog
from foam_cell_analysis.gui.navigation import PageId


def test_training_start_navigates_and_records_best_checkpoint(main_window, qapp):
    page = main_window.pages[PageId.TRAINING]
    assert f"id: {page.experiment_id.text()}" in page.yaml_preview.toPlainText()
    assert "used_item_ids: 学習開始時に確定" in page.yaml_preview.toPlainText()
    assert "data.used_item_ids" not in page.fields
    page.fields["training.epochs"].setValue(2)
    page.fields["checkpoint.validation_interval"].setValue(1)

    experiment_id = page.start_training(confirm=False)

    assert experiment_id
    assert main_window.stack.currentWidget() is main_window.pages[PageId.EXPERIMENTS]
    experiment = main_window.ctx.backend.get_experiment(experiment_id)
    assert experiment.status == "running"
    for _ in range(100):
        qapp.processEvents()
        experiment = main_window.ctx.backend.get_experiment(experiment_id)
        if experiment.status == "completed":
            break
        QTest.qWait(2)
    assert experiment.status == "completed"
    assert any(checkpoint.name == "best.pt" for checkpoint in experiment.checkpoints)
    assert page.experiment_id.text() != experiment_id


def test_training_return_allocates_new_experiment_id(main_window):
    page = main_window.pages[PageId.TRAINING]
    first_id = page.start_training(confirm=False)

    assert first_id
    first = main_window.ctx.backend.get_experiment(first_id)
    next_id = page.experiment_id.text()
    assert next_id != first_id

    main_window.navigate(PageId.TRAINING)
    assert page.experiment_id.text() == next_id
    second_id = page.start_training(confirm=False)

    assert second_id == next_id
    assert main_window.ctx.backend.get_experiment(first_id) is first


def test_existing_non_draft_experiment_id_is_rejected(main_window, monkeypatch):
    page = main_window.pages[PageId.TRAINING]
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: QMessageBox.StandardButton.Ok)
    page.experiment_id.setText("exp_0042")

    assert page.start_training(confirm=False) is None
    assert main_window.ctx.backend.get_experiment("exp_0042").status != "running"


def test_training_options_refresh_and_preserve_selection(main_window):
    page = main_window.pages[PageId.TRAINING]
    dataset_box = page.fields["data.dataset_version"]
    previous_dataset = dataset_box.currentData()
    profile_box = page.fields["augmentation.profile"]
    previous_profile = profile_box.currentData()
    created = main_window.ctx.backend.save_augmentation_profile(
        main_window.ctx.backend.get_augmentation_profile(previous_profile)
    )

    main_window.navigate(PageId.EXPERIMENTS)
    main_window.navigate(PageId.TRAINING)

    assert dataset_box.currentData() == previous_dataset
    assert profile_box.currentData() == previous_profile
    assert profile_box.findData(created.profile_id) >= 0


def test_model_type_switch_changes_model_specific_controls(main_window):
    page = main_window.pages[PageId.TRAINING]

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
