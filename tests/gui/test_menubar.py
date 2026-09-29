"""メニューバー、タブの主操作、右クリックの利用者経路。"""

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMenu

from foam_cell_analysis.gui.navigation import ModeId, PageId


def _menu(window, title):
    action = next(
        action for action in window.menuBar().actions() if action.text().startswith(title)
    )
    return action.menu()


def _find_action(menu, title):
    for action in menu.actions():
        if action.text().split("\t", 1)[0] == title:
            return action
        if action.menu():
            nested = _find_action(action.menu(), title)
            if nested:
                return nested
    return None


def test_mode_menu_order_and_mode_specific_menus(shell):
    assert [a.text().split("(", 1)[0] for a in shell.home.menuBar().actions()] == [
        "ファイル",
        "ツール",
        "ヘルプ",
    ]

    cases = (
        (PageId.DATA_PREPARATION, ["ファイル", "編集", "表示", "データセット", "ツール", "ヘルプ"]),
        (PageId.TRAINING, ["ファイル", "編集", "表示", "学習", "ツール", "ヘルプ"]),
        (PageId.CANDIDATES, ["ファイル", "表示", "候補", "リリース", "ツール", "ヘルプ"]),
        (PageId.INFERENCE, ["ヘルプ"]),
    )
    for page_id, expected in cases:
        shell.navigate(page_id)
        window = shell.manager.window(
            ModeId.DATA_PREPARATION
            if page_id == PageId.DATA_PREPARATION
            else ModeId.TRAINING
            if page_id == PageId.TRAINING
            else ModeId.COMPARISON
            if page_id == PageId.CANDIDATES
            else ModeId.INFERENCE
        )
        actual = [action.text().split("(", 1)[0] for action in window.menuBar().actions()]
        assert actual == expected
        assert all("(&" in action.text() for action in window.menuBar().actions())


def _assert_clean_menu(menu):
    actions = menu.actions()
    assert not actions or not actions[0].isSeparator()
    assert not actions or not actions[-1].isSeparator()
    assert all(
        not (left.isSeparator() and right.isSeparator())
        for left, right in zip(actions, actions[1:], strict=False)
    )
    for action in actions:
        assert action.statusTip() == ""
        if action.menu():
            _assert_clean_menu(action.menu())


def test_menu_action_uses_same_operation_and_shortcut_display_tracks_changes(shell, qtbot):
    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    window = shell.manager.window(ModeId.DATA_PREPARATION)
    item = page.items[0]
    previous = item.quality
    new_value = "良" if previous != "良" else "可"

    edit_menu = _menu(window, "編集")
    usage_action = _find_action(edit_menu, "用途を変更")
    train_action = _find_action(usage_action.menu(), "学習")
    train_action.trigger()
    assert page.items[0].usage == "train"

    page._change([item.item_id], quality=new_value)
    undo_index = page.undo_stack.index()
    undo_action = _find_action(edit_menu, "元に戻す")
    assert undo_action is not page._shortcut_actions["undo"]
    assert undo_action.shortcut().isEmpty()
    undo_action.trigger()
    assert page.undo_stack.index() == undo_index - 1
    assert page.items[0].quality == previous

    page._change([item.item_id], quality=new_value)
    keyboard_undo_index = page.undo_stack.index()
    window.show()
    window.activateWindow()
    window.raise_()
    page.table.setFocus()
    qtbot.waitUntil(window.isActiveWindow)
    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert page.undo_stack.index() == keyboard_undo_index - 1
    assert page.items[0].quality == previous

    data_menu = _menu(window, "データセット")
    auto = _find_action(data_menu, "自動振り分け…")
    assert auto.text().endswith("Ctrl+D")
    assert auto.shortcut().isEmpty()
    shell.ctx.shortcuts.assign("auto_triage", "Ctrl+Alt+D")
    shell.ctx.shortcuts.changed.emit()
    assert auto.text().endswith("Ctrl+Alt+D")
    assert auto.shortcut().isEmpty()
    assert page._shortcut_actions["auto_triage"].shortcut().toString() == "Ctrl+Alt+D"


def test_tab_tools_inactive_action_reason_and_candidate_release_reason(shell):
    shell.navigate(PageId.DATA_PREPARATION)
    data_window = shell.manager.window(ModeId.DATA_PREPARATION)
    data_page = shell.page(PageId.DATA_PREPARATION)
    assert data_window.tab_tools_layout.count() == 1
    assert data_window.tab_tools_layout.itemAt(0).widget() is data_page.tab_tools
    shell.navigate(PageId.DATASET_HISTORY)
    assert data_window.tab_tools_layout.count() == 0

    shell.navigate(PageId.EXPERIMENTS)
    training_window = shell.manager.window(ModeId.TRAINING)
    copy_action = _find_action(_menu(training_window, "学習"), "設定を複製して新規実験")
    shell.navigate(PageId.TRAINING_QUEUE)
    assert not copy_action.isEnabled()
    assert "実験一覧タブ" in copy_action.toolTip()

    shell.navigate(PageId.CANDIDATES)
    comparison_window = shell.manager.window(ModeId.COMPARISON)
    release = _find_action(_menu(comparison_window, "候補"), "選択候補をリリース…")
    candidates = shell.page(PageId.CANDIDATES)
    assert not release.isEnabled()
    assert candidates.release_reason.text() == "評価済みの候補を 1 つ選ぶとリリースできます"
    assert release.toolTip() == candidates.release_reason.text()


def test_toolbar_menus_follow_current_data_and_share_home_tool_names(shell, monkeypatch):
    shell.navigate(PageId.DATA_PREPARATION)
    data_window = shell.manager.window(ModeId.DATA_PREPARATION)
    data_menu = _menu(data_window, "編集")
    class_menu = _find_action(data_menu, "画像分類を変更").menu()
    monkeypatch.setattr(shell.ctx.backend, "classifications", ["分類A", "分類C"])
    shell.page(PageId.DATA_PREPARATION).refresh()
    class_labels = [action.text() for action in class_menu.actions() if not action.isSeparator()]
    assert class_labels[:2] == ["分類A\t1", "分類C\t2"]
    assert "分類 3\t3" not in class_labels
    monkeypatch.setattr(shell.ctx.backend, "classifications", ["分類A", "分類B", "分類C"])
    shell.page(PageId.DATA_PREPARATION).refresh()
    assert [action.text().split("\t", 1)[0] for action in class_menu.actions()[:3]] == [
        "分類A",
        "分類B",
        "分類C",
    ]
    shell.ctx.shortcuts.assign("class_1", "9")
    shell.ctx.shortcuts.changed.emit()
    assert class_menu.actions()[0].text() == "分類A\t9"

    data_tools = _menu(data_window, "ツール")
    assert [action.text().split("\t", 1)[0] for action in data_tools.actions()] == [
        "キー割り当て…",
        "原画像と切り替える表示",
    ]
    assert not any(
        action.text() == "キー割り当て…" for action in data_tools.actions()[1].menu().actions()
    )

    home_tools = shell.home.menuBar().actions()[1].menu()
    assert home_tools.actions()[0].text() == data_tools.actions()[0].text()
    assert home_tools.actions()[1].text() == data_tools.actions()[1].text()


def test_training_menu_groups_column_names_and_dynamic_experiment_filter(shell):
    shell.navigate(PageId.TRAINING_QUEUE)
    window = shell.manager.window(ModeId.TRAINING)
    edit = _menu(window, "編集")
    learning = _menu(window, "学習")
    edit_labels = [
        action.text().split("\t", 1)[0] for action in edit.actions() if not action.isSeparator()
    ]
    learning_labels = [
        action.text().split("\t", 1)[0] for action in learning.actions() if not action.isSeparator()
    ]
    assert "複製" in edit_labels and "削除" in edit_labels
    assert "設定を開いて編集…" in edit_labels
    assert "複製" not in learning_labels
    assert "▶ キューをすべて実行" in learning_labels
    assert "設定を検証" in learning_labels

    queue_columns = _find_action(_menu(window, "表示"), "表示する列（学習キュー）")
    shell.navigate(PageId.EXPERIMENTS)
    experiment_columns = _find_action(_menu(window, "表示"), "表示する列（実験一覧）")
    assert experiment_columns.isEnabled()
    assert not queue_columns.isEnabled()
    assert "学習キュータブ" in queue_columns.toolTip()
    page = shell.page(PageId.EXPERIMENTS)
    current = shell.ctx.backend.list_experiments()
    extra = __import__("copy").copy(current[0])
    extra.experiment_id = "exp_other_study"
    extra.study_id = "foam_other"
    shell.ctx.backend.list_experiments = lambda: [*current, extra]
    page.refresh()
    study_menu = _find_action(_menu(window, "表示"), "実験の絞り込み").menu()
    study_submenu = _find_action(study_menu, "実験群").menu()
    study_items = [action for action in study_submenu.actions() if not action.isSeparator()]
    assert [action.text() for action in study_items] == ["すべて", "foam_other", "foam_study"]
    study_items[1].trigger()
    assert page.study_filter.currentText() == "foam_other"
    page.study_filter.setCurrentText("foam_study")
    study_items = [
        action
        for action in _find_action(study_menu, "実験群").menu().actions()
        if not action.isSeparator()
    ]
    assert study_items[2].isChecked()


def test_right_click_selects_unselected_row_and_preserves_multiselection(shell, monkeypatch):
    shell.navigate(PageId.TRAINING_QUEUE)
    queue = shell.page(PageId.TRAINING_QUEUE)
    for _ in range(2):
        queue.ctx.backend.add_training_queue_item(
            queue.ctx.backend.default_experiment_config("mask_rcnn")
        )
    queue.refresh()
    monkeypatch.setattr(QMenu, "exec", lambda self, *_args: None)
    queue.table.show()
    shell.manager.window(ModeId.TRAINING).show()

    queue.table.selectRow(0)
    row_one = queue.table.visualRect(queue.model.index(1, 0)).center()
    QTest.mouseClick(queue.table.viewport(), Qt.MouseButton.RightButton, pos=row_one)
    assert [row.row() for row in queue.table.selectionModel().selectedRows()] == [1]

    selection = queue.table.selectionModel()
    selection.select(
        queue.model.index(0, 0),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )
    row_one = queue.table.visualRect(queue.model.index(1, 0)).center()
    QTest.mouseClick(queue.table.viewport(), Qt.MouseButton.RightButton, pos=row_one)
    assert {row.row() for row in selection.selectedRows()} == {0, 1}
