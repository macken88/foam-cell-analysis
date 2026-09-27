"""共有メニュー項目のページ状態とタブ可否の回帰確認。"""

from PySide6.QtCore import QEvent

from foam_cell_analysis.gui.navigation import ModeId, PageId


def _menu_action(window, top_label, item_label):
    root = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text().startswith(top_label)
    )

    def find(menu):
        for action in menu.actions():
            if action.text().split("\t", 1)[0] == item_label:
                return action
            if action.menu():
                result = find(action.menu())
                if result:
                    return result
        return None

    return find(root)


def test_wrong_training_tab_keeps_queue_actions_disabled_after_refreshes(shell, qapp, monkeypatch):
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    window = shell.manager.window(ModeId.TRAINING)
    shell.ctx.backend.add_training_queue_item(
        shell.ctx.backend.default_experiment_config("mask_rcnn")
    )
    queue.refresh()
    run_action = _menu_action(window, "学習", "キューを実行")
    assert run_action.isEnabled()

    shell.navigate(PageId.EXPERIMENTS)
    assert not run_action.isEnabled()
    assert "学習キュータブ" in run_action.toolTip()
    starts = []
    monkeypatch.setattr(shell.ctx.queue_controller, "start", lambda: starts.append(True))
    run_action.trigger()
    assert starts == []

    shell.ctx.queue_controller.changed.emit()
    qapp.processEvents()
    queue.refresh()
    assert not run_action.isEnabled()
    assert "学習キュータブ" in run_action.toolTip()
    shell.ctx.queue_controller.progressed.emit()
    shell.ctx.jobs.jobs_changed.emit(shell.ctx.jobs.running_count)
    qapp.sendEvent(window, QEvent(QEvent.Type.WindowDeactivate))
    qapp.sendEvent(window, QEvent(QEvent.Type.WindowActivate))
    qapp.processEvents()

    assert not run_action.isEnabled()
    assert "学習キュータブ" in run_action.toolTip()


def test_inactive_comparison_page_cannot_enable_candidate_actions(shell, qapp):
    shell.navigate(PageId.CANDIDATES)
    page = shell.page(PageId.CANDIDATES)
    action = page.candidate_actions["evaluate"]
    shell.navigate(PageId.MASK_COMPARISON)
    assert not action.isEnabled()

    page.table.selectRow(0)
    qapp.processEvents()

    assert not action.isEnabled()
    assert "リリース候補タブ" in action.toolTip()


def test_returning_to_data_page_restores_menu_tooltip(shell, qapp):
    shell.navigate(PageId.DATA_PREPARATION)
    data = shell.page(PageId.DATA_PREPARATION)
    window = shell.manager.window(ModeId.DATA_PREPARATION)
    import_action = _menu_action(window, "ファイル", "画像を取り込む…")
    original_tooltip = import_action.toolTip()
    assert original_tooltip != ""

    shell.navigate(PageId.DATASET_HISTORY)
    assert not import_action.isEnabled()
    assert "作業中データタブ" in import_action.toolTip()
    shell.navigate(PageId.DATA_PREPARATION)
    qapp.processEvents()

    assert import_action.isEnabled()
    assert import_action.toolTip() == original_tooltip
    data.refresh_menu_actions()
    assert import_action.toolTip() == original_tooltip
