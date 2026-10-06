"""メニューバー、タブの主操作、右クリックの利用者経路。"""

from collections import Counter

from PySide6.QtCore import QEvent, QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog, QMenu

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


def _click_mode_tab(window, title):
    index = next(i for i in range(window.tab_bar.count()) if window.tab_bar.tabText(i) == title)
    QTest.mouseClick(
        window.tab_bar,
        Qt.MouseButton.LeftButton,
        pos=window.tab_bar.tabRect(index).center(),
    )


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
    auto = _find_action(data_menu, "自動振り分け")
    assert auto.text().endswith("Ctrl+D")
    assert auto.shortcut().isEmpty()
    shell.ctx.shortcuts.assign("auto_triage", "Ctrl+Alt+D")
    shell.ctx.shortcuts.changed.emit()
    assert auto.text().endswith("Ctrl+Alt+D")
    assert auto.shortcut().isEmpty()
    assert page._shortcut_actions["auto_triage"].shortcut().toString() == "Ctrl+Alt+D"


def test_tab_tools_and_refresh_routes_follow_real_tab_clicks(shell):
    shell.navigate(PageId.DATA_PREPARATION)
    data_window = shell.manager.window(ModeId.DATA_PREPARATION)
    data_page = shell.page(PageId.DATA_PREPARATION)
    assert data_window.tab_tools_layout.count() == 1
    assert data_window.tab_tools_layout.itemAt(0).widget() is data_page.tab_tools
    _click_mode_tab(data_window, "データセット版履歴")
    assert data_window.tab_tools_layout.count() == 0
    _click_mode_tab(data_window, "作業中データ")
    assert data_window.tab_tools_layout.itemAt(0).widget() is data_page.tab_tools

    shell.navigate(PageId.EXPERIMENTS)
    training_window = shell.manager.window(ModeId.TRAINING)
    experiment_page = shell.page(PageId.EXPERIMENTS)
    experiment_index = experiment_page.table.model().index(0, 0)
    QTest.mouseClick(
        experiment_page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=experiment_page.table.visualRect(experiment_index).center(),
    )
    copy_action = _find_action(_menu(training_window, "学習"), "設定を引き継いで新規作成")
    assert copy_action.isEnabled()
    _click_mode_tab(training_window, "学習キュー")
    assert not copy_action.isEnabled()
    assert "実験一覧タブ" in copy_action.toolTip()
    _click_mode_tab(training_window, "実験一覧")
    assert copy_action.isEnabled()

    shell.navigate(PageId.CANDIDATES)
    comparison_window = shell.manager.window(ModeId.COMPARISON)
    release = _find_action(_menu(comparison_window, "候補"), "選択候補を採用")
    candidates = shell.page(PageId.CANDIDATES)
    # シード済みで評価完了している候補を、この画面テスト用に採用前状態へ戻す。
    shell.ctx.backend.get_candidate("RC-001").status = "candidate"
    candidates.refresh()
    candidates.table.clearSelection()
    candidate_row = next(
        row
        for row in range(candidates.table.rowCount())
        if candidates.table.item(row, 1).text() == "RC-001"
    )
    candidate_index = candidates.table.model().index(candidate_row, 0)
    QTest.mouseClick(
        candidates.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=candidates.table.visualRect(candidate_index).center(),
    )
    assert release.isEnabled()
    _click_mode_tab(comparison_window, "リリース済みモデル・振り分け")
    assert not release.isEnabled()
    assert "リリース候補タブ" in release.toolTip()
    released = shell.page(PageId.RELEASED_MODELS)
    detail = released.release_actions["detail"]
    released.model_table.clearSelection()
    assert not detail.isEnabled()
    assert detail.toolTip() == "モデルを選択してください"
    _click_mode_tab(comparison_window, "リリース候補")
    assert release.isEnabled()
    _click_mode_tab(comparison_window, "リリース済みモデル・振り分け")
    assert detail.isEnabled() == bool(released.model_table.selectionModel().selectedRows())


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
        "キー割り当て",
        "原画像と切り替える表示",
    ]
    assert not any(
        action.text() == "キー割り当て" for action in data_tools.actions()[1].menu().actions()
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
    assert "設定を開いて編集" in edit_labels
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


def _menus(window):
    pending = [action.menu() for action in window.menuBar().actions() if action.menu()]
    while pending:
        menu = pending.pop()
        yield menu
        pending.extend(action.menu() for action in menu.actions() if action.menu())


def test_all_mode_menus_have_unique_sibling_item_names(shell, qapp):
    mode_for_page = {
        PageId.DATA_PREPARATION: ModeId.DATA_PREPARATION,
        PageId.DATASET_HISTORY: ModeId.DATA_PREPARATION,
        PageId.TRAINING: ModeId.TRAINING,
        PageId.TRAINING_QUEUE: ModeId.TRAINING,
        PageId.EXPERIMENTS: ModeId.TRAINING,
        PageId.CANDIDATES: ModeId.COMPARISON,
        PageId.MASK_COMPARISON: ModeId.COMPARISON,
        PageId.RELEASED_MODELS: ModeId.COMPARISON,
        PageId.INFERENCE: ModeId.INFERENCE,
    }
    violations = []
    for page_id, mode_id in mode_for_page.items():
        shell.navigate(page_id)
        qapp.processEvents()
        window = shell.manager.window(mode_id)
        for menu in _menus(window):
            names = [
                action.text().split("\t", 1)[0]
                for action in menu.actions()
                if not action.isSeparator()
            ]
            duplicates = [name for name, count in Counter(names).items() if count > 1]
            if duplicates:
                violations.append((page_id, menu.title(), duplicates))
    assert violations == []


def test_changed_filter_menu_has_one_action_and_lower_entry_is_safe(shell, qapp):
    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    window = shell.manager.window(ModeId.DATA_PREPARATION)
    filter_menu_action = next(
        action
        for menu in _menus(window)
        for action in menu.actions()
        if action.menu() and action.text().split("\t", 1)[0] == "絞り込み"
    )
    changed = [
        action
        for action in filter_menu_action.menu().actions()
        if action.text().split("\t", 1)[0] == "変更あり"
    ]
    assert len(changed) == 1
    before = page.model.changed_only
    changed[0].trigger()
    qapp.processEvents()
    assert page.model.changed_only is not before


def test_archive_menu_action_records_archive_for_latest_versions(shell, monkeypatch):
    shell.page(PageId.DATA_PREPARATION)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: "C:/archive")
    window = shell.manager.window(ModeId.DATA_PREPARATION)
    file_menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text().startswith("ファイル")
    )
    action = next(action for action in file_menu.actions() if action.text() == "アーカイブを作成")
    assert action.isEnabled()
    action.trigger()
    latest = {}
    for version in shell.ctx.backend.list_dataset_versions():
        latest[version.purpose] = version
    assert all(version.archive_status == "COMPLETED" for version in latest.values())


def test_mask_import_menu_action_adds_revision_to_selected_item(shell):
    page = shell.page(PageId.DATA_PREPARATION)
    item = next(item for item in page.items if item.item_id in page._selected_ids)
    previous_revisions = len(item.mask_revisions)
    window = shell.manager.window(ModeId.DATA_PREPARATION)
    file_menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text().startswith("ファイル")
    )
    action = next(action for action in file_menu.actions() if action.text() == "正解ラベル版を追加")
    assert action.isEnabled()
    action.trigger()
    assert len(item.mask_revisions) == previous_revisions + 1


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
    run_action = _menu_action(window, "学習", "▶ キューをすべて実行")
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
    import_action = _menu_action(window, "ファイル", "画像を取り込む")
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
