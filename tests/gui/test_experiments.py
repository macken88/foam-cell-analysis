"""実験一覧の選択条件とモデル比較への遷移テスト。"""

from PySide6.QtWidgets import QDialog

from foam_cell_analysis.gui.modes.training.dialogs import (
    ExperimentCompareDialog,
    SendToCandidatesDialog,
)
from foam_cell_analysis.gui.navigation import PageId


def test_send_completed_experiment_transitions_with_candidate_parameters(main_window, monkeypatch):
    page = main_window.pages[PageId.EXPERIMENTS]
    target_row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0042"
    )
    page.table.setCurrentCell(target_row, 1)
    received = []
    monkeypatch.setattr(
        main_window.ctx.navigator,
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
