"""モデル学習ページと拡張プロファイルの画面テスト。"""

import copy

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QLineEdit, QMessageBox

from foam_cell_analysis.gui.modes.training.augmentation_dialog import AugmentationDialog
from foam_cell_analysis.gui.navigation import PageId
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

    page.model_type.setCurrentIndex(1)

    assert page.config["model"]["type"] == "cellpose"
    assert "model.pretrained_model" in page._model_widgets["cellpose"]
    assert "model.backbone" not in page._model_widgets["cellpose"]
    page.model_type.setCurrentIndex(0)
    assert page.config["model"]["type"] == "mask_rcnn"
    assert "model.backbone" in page._model_widgets["mask_rcnn"]


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
        .startswith("最終学習・epoch ")
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
        if label.text() == "エポック選択の指標：OOF 平均適合率（mAP）・最大"
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
    QTest.mouseClick(results.button_map["copy"], Qt.MouseButton.LeftButton)
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
