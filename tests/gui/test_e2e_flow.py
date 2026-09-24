"""MainWindow を通した画面間 E2E フロー。"""

from PySide6.QtCore import Qt

from foam_cell_analysis.gui.modes.comparison.dialogs import CandidateDialog
from foam_cell_analysis.gui.modes.data_preparation.dialogs import DatasetFinalizeDialog
from foam_cell_analysis.gui.navigation import PageId


def _wait_for(qtbot, predicate, timeout=5000):
    qtbot.waitUntil(predicate, timeout=timeout)


def test_dataset_finalize_appears_in_training_choices(main_window, qtbot):
    backend = main_window.ctx.backend
    page = main_window.pages[PageId.DATA_PREPARATION]
    dataset = backend.get_working_dataset("train")
    for item in dataset.items:
        if item.included and (not item.classification or not item.quality):
            backend.update_item(
                "train",
                item.item_id,
                classification=item.classification or "分類A",
                quality=item.quality or "良",
            )
    page.refresh()
    page.validate_button.click()
    _wait_for(qtbot, lambda: dataset.state == "VALIDATED")
    dialog = DatasetFinalizeDialog(page, backend, "train")
    version = dialog.apply()
    page.refresh()
    assert any(row.version == version.version for row in backend.list_dataset_versions("train"))
    assert version.version == "train_v004"
    main_window.navigate(PageId.TRAINING)
    choices = main_window.pages[PageId.TRAINING].fields["data.dataset_version"]
    assert choices.findText(version.version) >= 0


def test_training_candidate_release_and_routing_flow(main_window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox

    monkeypatch.setattr(
        CandidateDialog,
        "exec",
        lambda dialog: QDialog.DialogCode.Accepted,
    )

    training = main_window.pages[PageId.TRAINING]
    training.fields["training.epochs"].setValue(2)
    experiment_id = training.start_training(confirm=False)
    assert main_window.stack.currentWidget() is main_window.pages[PageId.EXPERIMENTS]
    assert main_window.ctx.jobs.jobs()
    _wait_for(
        qtbot, lambda: main_window.ctx.backend.get_experiment(experiment_id).status == "completed"
    )
    experiment = main_window.ctx.backend.get_experiment(experiment_id)
    assert any(checkpoint.name == "best.pt" for checkpoint in experiment.checkpoints)

    experiments = main_window.pages[PageId.EXPERIMENTS]
    row = next(
        index
        for index in range(experiments.table.rowCount())
        if experiments.table.item(index, 1).text() == experiment_id
    )
    experiments.table.setCurrentCell(row, 1)
    monkeypatch.setattr(
        "foam_cell_analysis.gui.modes.training.experiment_list.SendToCandidatesDialog.exec",
        lambda dialog: QDialog.DialogCode.Accepted,
    )
    experiments.send_selected()
    assert main_window.stack.currentWidget() is main_window.pages[PageId.CANDIDATES]
    candidates = main_window.pages[PageId.CANDIDATES]
    qtbot.waitUntil(
        lambda: any(
            item.experiment_id == experiment_id
            for item in main_window.ctx.backend.list_candidates()
        )
    )
    candidate = next(
        item
        for item in main_window.ctx.backend.list_candidates()
        if item.experiment_id == experiment_id
    )
    assert candidate.checkpoint == "best.pt"
    candidates.refresh()
    row = next(
        index
        for index in range(candidates.table.rowCount())
        if candidates.table.item(index, 1).text() == candidate.candidate_id
    )
    candidates.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
    candidates._evaluate()
    _wait_for(
        qtbot, lambda: candidate.status == "candidate" and "val_v003" in candidate.evaluations
    )
    candidates.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
    monkeypatch.setattr(
        "foam_cell_analysis.gui.modes.comparison.candidates_page.ReleaseDialog.exec",
        lambda dialog: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )
    candidates._release()
    released = main_window.ctx.backend.list_released_models()[-1]
    assert main_window.stack.currentWidget() is main_window.pages[PageId.RELEASED_MODELS]
    assert main_window.pages[PageId.RELEASED_MODELS].select_model(released.model_id)
    routing = main_window.pages[PageId.RELEASED_MODELS]
    control = routing._routing_controls["分類A"]
    control.setCurrentIndex(control.findData(released.model_id))
    routing.apply_pending_changes()
    main_window.navigate(PageId.INFERENCE)
    inference = main_window.pages[PageId.INFERENCE]
    inference.add_images(["e2e.png"])
    assert inference.inputs[0].model_id == released.model_id


def test_candidate_compare_keeps_sidebar_and_returns(main_window):
    candidates = main_window.pages[PageId.CANDIDATES]
    for row in (0, 1):
        candidates.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
    candidates._compare()
    assert main_window.stack.currentWidget() is main_window.pages[PageId.MASK_COMPARISON]
    assert main_window.sidebar.currentItem().data(256) == PageId.CANDIDATES
    main_window.pages[PageId.MASK_COMPARISON].back.click()
    assert main_window.stack.currentWidget() is candidates


def test_copy_experiment_allocates_new_id_and_copies_config(main_window):
    experiments = main_window.pages[PageId.EXPERIMENTS]
    row = next(
        index
        for index in range(experiments.table.rowCount())
        if experiments.table.item(index, 1).text() == "exp_0042"
    )
    experiments.table.setCurrentCell(row, 1)
    config = main_window.ctx.backend.get_experiment("exp_0042").config.values
    experiments.copy_selected()
    training = main_window.pages[PageId.TRAINING]
    assert main_window.stack.currentWidget() is training
    assert training.config["experiment"]["id"] != "exp_0042"
    assert training.config["model"] == config["model"]


def test_unassigned_inference_routes_to_released_models(main_window):
    inference = main_window.pages[PageId.INFERENCE]
    main_window.ctx.backend.apply_routing({"分類A": None})
    inference.refresh_routing()
    inference.add_images(["unassigned.png"])
    assert not inference.route_button.isHidden()
    inference.route_button.click()
    assert main_window.stack.currentWidget() is main_window.pages[PageId.RELEASED_MODELS]


def test_all_page_entries_and_job_status(main_window, qtbot):
    for page_id in PageId:
        main_window.navigate(page_id)
        assert main_window.stack.currentWidget() is main_window.pages[page_id]
    training = main_window.pages[PageId.TRAINING]
    training.fields["training.epochs"].setValue(20)
    experiment_id = training.start_training(confirm=False)
    assert "1" in main_window.job_count.text()
    _wait_for(
        qtbot,
        lambda: main_window.ctx.backend.get_experiment(experiment_id).status == "completed",
        10000,
    )
    assert main_window.job_count.text() == "実行中ジョブ: 0"
