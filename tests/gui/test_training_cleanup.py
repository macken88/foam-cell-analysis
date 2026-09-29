"""実験一覧の「成果物を整理…」（比較・評価設計 19.5）。"""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from foam_cell_analysis.gui.labels import format_bytes
from foam_cell_analysis.gui.modes.training.cleanup_dialog import ArtifactCleanupDialog
from foam_cell_analysis.gui.navigation import ModeId, PageId


def _row_of(page, experiment_id):
    return next(
        row
        for row in range(page.table.rowCount())
        if page.table.item(row, 1).text() == experiment_id
    )


def _click_checkbox(page, row):
    rect = page.table.visualItemRect(page.table.item(row, 0))
    position = rect.topLeft() + QPoint(12, rect.height() // 2)
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=position)


def _click_check(check):
    """チェック欄の印の位置をクリックする（欄の中央は文字の外になることがある）。"""
    QTest.mouseClick(check, Qt.MouseButton.LeftButton, pos=QPoint(8, check.height() // 2))


def _click_menu_item(window, top_label, item_label):
    """メニューバーのメニューを開き、項目をマウスでクリックする。"""
    menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text().startswith(top_label)
    )
    action = next(item for item in menu.actions() if item.text() == item_label)
    assert action.isEnabled(), action.toolTip()
    menu.popup(window.mapToGlobal(QPoint(0, 0)))
    QTest.qWaitForWindowExposed(menu)
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())


def test_cleanup_from_menu_updates_freed_size_and_marks_pruned_checkpoints(shell, monkeypatch):
    backend = shell.ctx.backend
    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    window = shell.manager.window(ModeId.TRAINING)
    window.resize(1400, 900)
    window.show()
    QTest.qWaitForWindowExposed(window)
    targets = ["exp_0042", "exp_0043"]
    for experiment_id in targets:
        _click_checkbox(page, _row_of(page, experiment_id))
    messages = []
    shell.ctx.status.message.connect(messages.append)
    questions = []

    def answer_yes(_parent, _title, text, *_args):
        questions.append(text)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", answer_yes)
    seen = {}

    def operate(dialog):
        seen["dialog"] = dialog
        assert dialog.experiment_ids == targets
        periodic = dialog.checks["fold_periodic"]
        selected = dialog.checks["fold_selected"]
        assert periodic.isChecked() and not selected.isChecked()
        expected = backend.estimate_freed_bytes(targets, dialog.selected_categories())
        assert dialog.freed_label.text() == f"空く容量: {format_bytes(expected)}"
        # 最終学習モデルは候補 RC-003 が使っているので、その分は理由付きで残る
        assert "RC-003" in dialog.notes["final"].text()
        dialog.show()
        QTest.qWaitForWindowExposed(dialog)
        _click_check(selected)
        assert selected.isChecked()
        larger = backend.estimate_freed_bytes(targets, ["fold_periodic", "fold_selected"])
        assert larger > expected
        assert dialog.freed_label.text() == f"空く容量: {format_bytes(larger)}"
        _click_check(selected)
        QTest.mouseClick(dialog.delete_button, Qt.MouseButton.LeftButton)
        seen["freed"] = expected
        return dialog.result()

    monkeypatch.setattr(ArtifactCleanupDialog, "exec", operate)
    _click_menu_item(window, "学習", "成果物を整理…")

    assert "dialog" in seen
    assert questions == [f"{format_bytes(seen['freed'])} を削除します。取り消せません。"]
    assert any("成果物を整理しました" in message for message in messages)
    experiment = backend.get_experiment("exp_0042")
    assert backend.pruned_paths("exp_0042", experiment.runs[-1].attempt)
    page.table.setCurrentCell(_row_of(page, "exp_0042"), 1)
    names = [
        page.checkpoint_table.item(row, 1).text() for row in range(page.checkpoint_table.rowCount())
    ]
    assert any(name.startswith("epoch_") and name.endswith("（削除済み）") for name in names)
    assert "selected.pt" in names
    assert "final.pt" in names


def test_cleanup_is_in_context_menu(shell):
    shell.navigate(PageId.EXPERIMENTS)
    page = shell.page(PageId.EXPERIMENTS)
    page.table.setCurrentCell(_row_of(page, "exp_0042"), 1)
    page.context_menu.aboutToShow.emit()
    labels = [action.text() for action in page.context_menu.actions()]
    assert "成果物を整理…" in labels
