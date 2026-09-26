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
from foam_cell_analysis.gui.navigation import Navigator, PageId
from foam_cell_analysis.services.mock.backend import MockBackend


def make_context() -> AppContext:
    return AppContext(MockBackend(), Navigator(), JobManager())


def test_candidate_add_evaluate_release_and_route(qtbot):
    ctx = make_context()
    experiment = ctx.backend.get_experiment("exp_0042")
    config = ctx.backend.create_inference_config("mask_rcnn", {"box_score_thresh": 0.3})
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
    config = ctx.backend.create_inference_config("mask_rcnn", {"box_score_thresh": 0.3})
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
    assert not page.display_toggle.is_alternate
    start = page.index
    QTest.keyClick(page.views[0], Qt.Key.Key_Right)
    assert page.index == (start + 1) % len(page.items)
    assert all(not view.scene().items() == [] for view in page.views)


def test_candidate_table_defaults_to_latest_and_uses_checkboxes(qtbot):
    ctx = make_context()
    page = CandidatesPage(ctx)
    qtbot.addWidget(page)
    assert page.validation.currentText() == "val_v003"
    assert page.table.columnCount() == 11
    assert page.table.horizontalHeaderItem(0).text() == "選択"
    page.resize(1200, 700)
    page.show()
    QTest.qWait(50)
    rect = page.table.visualItemRect(page.table.item(0, 0))
    position = rect.topLeft() + QPoint(12, rect.height() // 2)
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert len(page._selected()) == 1


def test_seeded_candidates_show_inference_specific_oof_in_gui(shell):
    page = shell.page(PageId.CANDIDATES)
    page.resize(1400, 800)
    page.show()
    rows = {}
    for row in range(page.table.rowCount()):
        candidate_id = page.table.item(row, 1).text()
        if candidate_id in {"RC-001", "RC-003"}:
            rows[candidate_id] = row
    assert set(rows) == {"RC-001", "RC-003"}
    for _candidate_id, row in rows.items():
        cell = page.table.visualItemRect(page.table.item(row, 7))
        QTest.mouseClick(
            page.table.viewport(),
            Qt.MouseButton.LeftButton,
            pos=cell.center(),
        )
        assert page.table.item(row, 7).text()
    assert page.table.item(rows["RC-001"], 7).text() != page.table.item(rows["RC-003"], 7).text()


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
    assert evaluation.metrics.rowCount() == 4
    assert evaluation.metrics.columnCount() == 5
    assert evaluation.metrics.item(0, 1).text() == "0.910"
    assert evaluation.metrics.height() <= 160
    assert all(
        evaluation.metrics.horizontalHeader().sectionResizeMode(column)
        == QHeaderView.ResizeMode.Interactive
        for column in range(evaluation.metrics.columnCount())
    )
    assert all(
        evaluation.results.horizontalHeader().sectionResizeMode(column)
        == QHeaderView.ResizeMode.Interactive
        for column in range(evaluation.results.columnCount())
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
