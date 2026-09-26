"""比較・リリース画面の動作確認。"""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QHeaderView

from foam_cell_analysis.gui.context import AppContext
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.modes.comparison.candidates_page import CandidatesPage
from foam_cell_analysis.gui.modes.comparison.dialogs import (
    CandidateDialog,
    EvaluationDialog,
    MaskExportDialog,
)
from foam_cell_analysis.gui.modes.comparison.mask_compare import MaskComparisonPage
from foam_cell_analysis.gui.navigation import Navigator
from foam_cell_analysis.gui.widgets.image_convert import DisplayMode
from foam_cell_analysis.services.mock.backend import MockBackend


def make_context() -> AppContext:
    return AppContext(MockBackend(), Navigator(), JobManager())


def test_candidate_add_evaluate_release_and_route(qtbot):
    ctx = make_context()
    experiment = ctx.backend.get_experiment("exp_0042")
    config = ctx.backend.list_inference_configs("mask_rcnn")[0]
    candidate = ctx.backend.add_candidate(
        "exp_0042", experiment.checkpoints[-1].name, config.config_id
    )
    ctx.backend.start_evaluation([candidate.candidate_id], "val_v003")
    ctx.backend.evaluate_candidate(candidate.candidate_id, "val_v003")
    released = ctx.backend.release_candidate(candidate.candidate_id, "テスト", "val_v003")
    assert candidate.status == "released"
    assert released.candidate_id == candidate.candidate_id
    assert released.model_id in [model.model_id for model in ctx.backend.list_released_models()]


def test_duplicate_candidate_raises():
    ctx = make_context()
    experiment = ctx.backend.get_experiment("exp_0042")
    config = ctx.backend.list_inference_configs("mask_rcnn")[0]
    checkpoint = experiment.checkpoints[-1].name
    ctx.backend.add_candidate("exp_0042", checkpoint, config.config_id)
    try:
        ctx.backend.add_candidate("exp_0042", checkpoint, config.config_id)
    except ValueError as error:
        assert "登録済み" in str(error)
    else:
        raise AssertionError("重複候補が拒否されませんでした")


def test_mask_comparison_slots_and_navigation(qtbot):
    ctx = make_context()
    page = MaskComparisonPage(ctx)
    qtbot.addWidget(page)
    page.on_enter({"validation_version": "val_v003", "candidate_ids": ["RC-001", "RC-002"]})
    assert len(page.views) == 3
    assert len(page.items) > 1
    original = page.index
    page._move(1)
    assert page.index == (original + 1) % len(page.items)
    assert not page.views[0].scene().items() == []
    assert all(view.horizontalScrollBarPolicy().name == "ScrollBarAlwaysOff" for view in page.views)


def test_mask_comparison_shortcuts_update_every_slot_and_status(qtbot):
    ctx = make_context()
    page = MaskComparisonPage(ctx)
    qtbot.addWidget(page)
    page.on_enter({"candidate_ids": ["RC-001", "RC-002"]})
    messages = []
    ctx.status.message.connect(messages.append)
    page.views[0].setFocus()
    QTest.keyClick(page.views[0], Qt.Key.Key_M)
    assert page.mode_buttons[DisplayMode.INSTANCE_LABEL].isChecked()
    assert messages[-1] == "表示形式: インスタンスラベル"
    QTest.keyClick(page.views[0], Qt.Key.Key_BracketRight)
    assert page._channel == "B"
    assert all(not view.scene().items() == [] for view in page.views)


def test_candidate_table_defaults_to_latest_and_uses_checkboxes(qtbot):
    ctx = make_context()
    page = CandidatesPage(ctx)
    qtbot.addWidget(page)
    assert page.validation.currentText() == "val_v003"
    assert page.table.columnCount() == 10
    assert page.table.horizontalHeaderItem(0).text() == "選択"
    page.resize(1200, 700)
    page.show()
    QTest.qWait(50)
    rect = page.table.visualItemRect(page.table.item(0, 0))
    position = rect.topLeft() + QPoint(12, rect.height() // 2)
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert len(page._selected()) == 1


def test_release_button_explains_missing_evaluation(qtbot):
    ctx = make_context()
    page = CandidatesPage(ctx)
    qtbot.addWidget(page)
    row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "RC-003"
    )

    page.table.item(row, 0).setCheckState(Qt.CheckState.Checked)

    assert not page.buttons["release"].isEnabled()
    assert "評価を完了" in page.buttons["release"].toolTip()


def test_candidate_dialog_builds_model_specific_fields(qtbot):
    ctx = make_context()
    dialog = CandidateDialog(ctx)
    qtbot.addWidget(dialog)
    assert dialog.model_type.text() == "Mask R-CNN"
    assert set(dialog.fields) == {
        "box_score_thresh",
        "box_nms_thresh",
        "box_detections_per_img",
    }
    assert not dialog.fields["box_score_thresh"].isEnabled()
    dialog.use_new.setChecked(True)
    assert dialog.fields["box_score_thresh"].isEnabled()
    dialog.experiment.setCurrentText("exp_0043")
    assert dialog.model_type.text() == "Cellpose"
    assert set(dialog.fields) == {"cellprob_threshold", "flow_threshold"}


def test_detail_evaluation_and_mask_export_dialogs(qtbot):
    ctx = make_context()
    evaluation = EvaluationDialog(ctx, ctx.backend.get_candidate("RC-001"), "val_v003")
    qtbot.addWidget(evaluation)
    assert (evaluation.minimumWidth(), evaluation.minimumHeight()) == (800, 640)
    assert evaluation.metrics.rowCount() == 2
    assert evaluation.metrics.columnCount() == 5
    assert evaluation.metrics.item(0, 1).text() == "0.910"
    assert evaluation.metrics.height() <= 100
    assert all(
        evaluation.metrics.horizontalHeader().sectionResizeMode(column)
        == QHeaderView.ResizeMode.Stretch
        for column in range(1, 5)
    )
    assert (
        evaluation.results.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    )
    assert (
        evaluation.results.horizontalHeader().sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
    )
    assert (
        evaluation.results.horizontalHeader().sectionResizeMode(2)
        == QHeaderView.ResizeMode.ResizeToContents
    )
    evaluation.show()
    QTest.qWait(50)
    assert evaluation.add_row.y() < evaluation.results.y()
    assert evaluation.add_row.width() < evaluation.width() // 2
    export = MaskExportDialog()
    qtbot.addWidget(export)
    assert export.browse.text() == "参照…"
    assert not export.classification.isEnabled()
    export.scope.setCurrentIndex(1)
    assert export.classification.isEnabled()
