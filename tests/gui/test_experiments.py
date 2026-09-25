"""実験一覧の選択条件とモデル比較への遷移テスト。"""

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog

from foam_cell_analysis.gui.modes.training.dialogs import (
    ExperimentCompareDialog,
    SendToCandidatesDialog,
)
from foam_cell_analysis.gui.navigation import PageId
from foam_cell_analysis.gui.theme import SERIES


def test_send_completed_experiment_transitions_with_candidate_parameters(shell, monkeypatch):
    page = shell.page(PageId.EXPERIMENTS)
    target_row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0042"
    )
    page.table.setCurrentCell(target_row, 1)
    received = []
    monkeypatch.setattr(
        shell.ctx.navigator,
        "navigate",
        lambda page_id, **params: received.append((page_id, params)),
    )
    monkeypatch.setattr(SendToCandidatesDialog, "exec", lambda _self: QDialog.DialogCode.Accepted)

    page.send_selected()

    assert received[-1][0] == PageId.CANDIDATES
    assert received[-1][1]["action"] == "add_candidate"
    assert received[-1][1]["experiment_id"] == "exp_0042"
    assert received[-1][1]["checkpoint"] in {"best.pt", "best"}


def test_compare_dialog_has_one_value_column_per_experiment(mock_backend):
    experiments = [
        mock_backend.get_experiment("exp_0042"),
        mock_backend.get_experiment("exp_0043"),
    ]

    dialog = ExperimentCompareDialog(experiments)

    assert dialog.table.columnCount() == 3
    assert dialog.table.horizontalHeaderItem(1).text() == "exp_0042"
    assert dialog.table.horizontalHeaderItem(2).text() == "exp_0043"
    row = next(
        index
        for index in range(dialog.table.rowCount())
        if dialog.table.item(index, 0).toolTip() == "data.used_item_ids"
    )
    assert dialog.table.item(row, 1).text() == "40 件"
    assert dialog.table.item(row, 2).text() == "40 件"


def test_learning_curves_use_separate_map_and_loss_charts(shell):
    page = shell.page(PageId.EXPERIMENTS)
    row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0042"
    )

    page.table.setCurrentCell(row, 1)

    assert [series[0] for series in page.chart_map.series] == ["mAP"]
    assert [series[0] for series in page.chart_loss.series] == ["loss"]
    assert page.chart_map.best is not None


def test_overview_has_empty_hint_and_two_column_selected_details(shell):
    page = shell.page(PageId.EXPERIMENTS)
    assert page.overview_placeholder.text() == "実験を選ぶと詳細が表示されます"
    row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0042"
    )

    page.table.setCurrentCell(row, 1)

    assert page.overview_table.columnCount() == 2
    assert page.overview_table.rowCount() > 0
    assert page.overview_table.item(0, 0).text() == "実験群"


def test_comparison_series_colors_keep_theme_order_after_refresh(mock_backend):
    experiments = [mock_backend.get_experiment(key) for key in ("exp_0042", "exp_0043", "exp_0044")]
    dialog = ExperimentCompareDialog(experiments)
    expected = [QColor(color) for color in SERIES]

    dialog.differences_only.setChecked(True)

    assert [series[1] for series in dialog.chart.series] == expected[:3]
