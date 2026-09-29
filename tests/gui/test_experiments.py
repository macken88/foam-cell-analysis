"""実験一覧の選択条件とモデル比較への遷移テスト。"""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

from foam_cell_analysis.gui.modes.training.dialogs import (
    ExperimentCompareDialog,
    SendToCandidatesDialog,
)
from foam_cell_analysis.gui.navigation import PageId


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
    assert received[-1][1]["checkpoint"] == "final.pt"


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
    assert dialog.table.item(row, 1).text() == "59 件"
    assert dialog.table.item(row, 2).text() == "59 件"


def test_overview_translates_all_cv_config_keys_and_model_details_distinguish_sections(shell):
    """概要の交差検証キーを日本語にし、Mask R-CNN の詳細区分を区別する。"""
    from foam_cell_analysis.gui.widgets.form import CollapsibleSection

    page = shell.page(PageId.EXPERIMENTS)
    row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0042"
    )
    rect = page.table.visualItemRect(page.table.item(row, 1))
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    experiment = shell.ctx.backend.get_experiment("exp_0042")
    experiment.oof_evaluation = shell.ctx.backend._oof_evaluation(experiment, 0.91)
    page._current_changed()
    oof_headers = [
        page.oof_table.horizontalHeaderItem(column).text()
        for column in range(page.oof_table.columnCount())
    ]
    assert "未分類" in oof_headers
    assert int(page.oof_table.item(1, 1).text()) == len(experiment.used_item_ids)
    labels = {
        page.overview_table.item(row, 0).text() for row in range(page.overview_table.rowCount())
    }
    assert "画像分類で層別" in labels
    assert "取り込み元フォルダ単位で分割" in labels
    assert all("_" not in label and "." not in label for label in labels), labels
    assert all(
        page.overview_table.item(row, 0).text()
        not in {"stratify_by_classification", "group_by_source_folder"}
        for row in range(page.overview_table.rowCount())
    )

    training = shell.page(PageId.TRAINING)
    sections = {
        section.button.text()
        for section in training.findChildren(CollapsibleSection)
        if "詳細設定" in section.button.text()
    }
    assert "RPN 詳細設定 ▶" in sections
    assert "ROI 詳細設定 ▶" in sections
    assert "詳細設定 ▶" not in sections


def _click_experiment(page, experiment_id):
    row = next(
        row
        for row in range(page.table.rowCount())
        if page.table.item(row, 1).text() == experiment_id
    )
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.table.visualItemRect(page.table.item(row, 1)).center(),
    )
    return row


def _open_row_menu(page, row, monkeypatch):
    """右クリックで届く QContextMenuEvent を送り、表示されたメニューを返す。"""
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QApplication

    shown = []
    monkeypatch.setattr(page.context_menu, "exec", lambda *_args: shown.append(True))
    viewport = page.table.viewport()
    pos = page.table.visualItemRect(page.table.item(row, 1)).center()
    QApplication.sendEvent(
        viewport,
        QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos, viewport.mapToGlobal(pos)),
    )
    assert shown
    page.context_menu.aboutToShow.emit()
    return page.context_menu


def _action(menu, text):
    return next(action for action in menu.actions() if action.text() == text)


def test_stopped_experiment_menu_offers_retry_first_and_deletes_after_confirmation(
    shell, qapp, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    from foam_cell_analysis.gui.navigation import ModeId
    from foam_cell_analysis.services.models import Candidate

    backend = shell.ctx.backend
    backend.candidates["RC-009"] = Candidate("RC-009", "exp_0044", "final.pt", "infer_v005")
    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    window = shell.manager.window(ModeId.TRAINING)
    window.show()
    row = _click_experiment(page, "exp_0044")

    menu = _open_row_menu(page, row, monkeypatch)
    items = [action for action in menu.actions() if not action.isSeparator()]
    assert items[0].text() == "同じ設定でやり直す"
    assert items[0].isEnabled()
    delete = _action(menu, "実験を削除…")
    assert not delete.isEnabled()
    assert "比較候補 RC-009 がこの実験を参照しています" in delete.toolTip()

    backend.reject_candidate("RC-009")
    menu = _open_row_menu(page, row, monkeypatch)
    delete = _action(menu, "実験を削除…")
    assert delete.isEnabled()
    prompts = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda _parent, _title, text, *_args: (
            prompts.append(text) or QMessageBox.StandardButton.Yes
        ),
    )
    menu.popup(page.table.viewport().mapToGlobal(QPoint(10, 10)))
    qapp.processEvents()
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(delete).center())
    qapp.processEvents()

    assert "exp_0044 を削除しますか？" in prompts[0]
    assert "取り消せません" in prompts[0]
    assert "exp_0044" not in {item.experiment_id for item in backend.list_experiments()}
    assert "exp_0044" not in {
        page.table.item(index, 1).text() for index in range(page.table.rowCount())
    }
    window.hide()


def test_released_or_running_experiment_cannot_be_deleted(shell, monkeypatch):
    backend = shell.ctx.backend
    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    _click_experiment(page, "exp_0042")
    delete = page.action_map["delete"]
    assert not delete.isEnabled()
    assert "リリース済みモデル" in delete.toolTip()

    backend.get_experiment("exp_0045").status = "running"
    page.refresh()
    _click_experiment(page, "exp_0045")
    assert not page.action_map["delete"].isEnabled()
    assert "学習中" in page.action_map["delete"].toolTip()
    assert "exp_0045" in {item.experiment_id for item in backend.list_experiments()}


def test_run_tab_button_retries_stopped_experiment_with_same_settings(shell, qapp):
    from foam_cell_analysis.gui.navigation import ModeId

    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    window = shell.manager.window(ModeId.TRAINING)
    window.show()
    _click_experiment(page, "exp_0044")
    page.details.setCurrentWidget(page.run_page)
    qapp.processEvents()
    assert page.retry_button.text() == "同じ設定でやり直す"
    assert page.retry_button.isEnabled()

    QTest.mouseClick(page.retry_button, Qt.MouseButton.LeftButton)

    runner = shell.ctx.training_runner
    assert runner.is_busy
    assert runner.experiment_id == "exp_0044"
    runner.request_stop()
    window.hide()
