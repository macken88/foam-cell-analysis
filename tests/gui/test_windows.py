"""ホームとモードウィンドウの遷移・状態表示。"""

from unittest.mock import patch

from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from foam_cell_analysis.gui.home_summary import build_home_summary
from foam_cell_analysis.gui.navigation import ModeId, PageId
from foam_cell_analysis.gui.window_manager import (
    MODE_PAGES,
    PAGE_TO_MODE_TAB,
    PAGE_TYPES,
)


def test_every_page_can_be_opened(shell, qapp):
    for page_id in PageId:
        shell.ctx.navigator.navigate(page_id, probe=True)
        qapp.processEvents()
        mode = PAGE_TO_MODE_TAB[page_id][0]
        window = shell.manager.window(mode)
        assert window.stack.currentWidget() is shell.page(page_id)
        assert shell.manager.current_page_id(mode) == page_id


def test_shutdown_confirmation_explains_that_training_will_stop(shell, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from foam_cell_analysis.gui import training_runner as training_module

    real_fake_job = training_module.FakeTrainingJob

    def slow_fake_job(*args, **kwargs):
        kwargs["interval_ms"] = 5000
        return real_fake_job(*args, **kwargs)

    monkeypatch.setenv("FOAM_MOCK_SPEED", "1")
    monkeypatch.setattr(training_module, "FakeTrainingJob", slow_fake_job)
    prompts = []

    def decline(_parent, _title, message, *_args, **_kwargs):
        prompts.append(message)
        return (
            QMessageBox.StandardButton.Yes
            if message == "この設定で学習を開始しますか？"
            else QMessageBox.StandardButton.No
        )

    monkeypatch.setattr(QMessageBox, "question", decline)
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    QTest.mouseClick(page.start_button, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert shell.ctx.training_runner.is_busy

    _click_file_menu_exit(shell, qapp)
    assert prompts == ["この設定で学習を開始しますか？", "学習を中断して終了しますか？"]
    assert shell.home.isVisible()
    shell.ctx.training_runner.job.kill()


def _click_file_menu_exit(shell, qapp):
    menu_bar = shell.home.menuBar()
    file_action = next(
        action for action in menu_bar.actions() if action.text().startswith("ファイル")
    )
    QTest.mouseClick(
        menu_bar,
        Qt.MouseButton.LeftButton,
        pos=menu_bar.actionGeometry(file_action).center(),
    )
    qapp.processEvents()
    file_menu = file_action.menu()
    exit_action = next(action for action in file_menu.actions() if action.text() == "終了")
    QTest.mouseClick(
        file_menu,
        Qt.MouseButton.LeftButton,
        pos=file_menu.actionGeometry(exit_action).center(),
    )
    qapp.processEvents()


def _start_training_with_waiting_evaluation(shell, monkeypatch):
    from foam_cell_analysis.gui import training_runner as training_module

    real_fake_job = training_module.FakeTrainingJob
    monkeypatch.setattr(
        training_module,
        "FakeTrainingJob",
        lambda *args, **kwargs: real_fake_job(*args, interval_ms=10_000, **kwargs),
    )
    monkeypatch.setenv("FOAM_MOCK_SPEED", "1")
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    shell.navigate(PageId.TRAINING)
    QTest.mouseClick(shell.page(PageId.TRAINING).start_button, Qt.MouseButton.LeftButton)
    assert shell.ctx.training_runner.is_busy
    prepared = []
    original_prepare = shell.ctx.backend.prepare_evaluation_run

    def record_prepare(*args, **kwargs):
        prepared.append(args)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(shell.ctx.backend, "prepare_evaluation_run", record_prepare)
    shell.ctx.evaluation_runner.start(["RC-WAITING"])
    assert shell.ctx.evaluation_runner.waiting_for_compute
    return prepared


def test_menu_exit_yes_blocks_before_stop_and_skips_waiting_evaluation_prepare(
    shell, qapp, monkeypatch
):
    prepared = _start_training_with_waiting_evaluation(shell, monkeypatch)
    stop_checks = []
    original_stop = shell.ctx.training_runner.request_stop

    def check_block_then_stop(*args, **kwargs):
        stop_checks.append(shell.ctx.compute.is_blocked)
        return original_stop(*args, **kwargs)

    monkeypatch.setattr(shell.ctx.training_runner, "request_stop", check_block_then_stop)
    quit_calls = []
    monkeypatch.setattr(qapp, "quit", lambda: quit_calls.append(True))
    _click_file_menu_exit(shell, qapp)

    assert stop_checks == [True]
    assert prepared == []
    assert shell.ctx.compute.is_blocked
    assert not shell.ctx.training_runner.is_busy
    assert not shell.ctx.evaluation_runner.is_busy
    assert quit_calls == [True]


def test_menu_exit_continues_after_training_stop_save_failure(shell, qapp, monkeypatch, caplog):
    _start_training_with_waiting_evaluation(shell, monkeypatch)
    original_shutdown = shell.ctx.evaluation_runner.shutdown
    shutdown_calls = []

    monkeypatch.setattr(
        shell.ctx.training_runner,
        "request_stop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("stop save failed")),
    )
    monkeypatch.setattr(
        shell.ctx.evaluation_runner,
        "shutdown",
        lambda *args, **kwargs: (
            shutdown_calls.append(shell.ctx.compute.is_blocked)
            or original_shutdown(*args, **kwargs)
        ),
    )
    settings_calls = []
    monkeypatch.setattr(
        shell.manager,
        "save_all_windows",
        lambda: settings_calls.append(shell.ctx.compute.is_blocked),
    )
    quit_calls = []
    monkeypatch.setattr(qapp, "quit", lambda: quit_calls.append(True))
    _click_file_menu_exit(shell, qapp)

    assert shell.ctx.compute.is_blocked
    assert shutdown_calls == [True]
    assert settings_calls == [True]
    assert quit_calls == [True]
    assert "学習停止要求を保存できません" in caplog.text
    monkeypatch.undo()
    shell.ctx.training_runner.request_stop("user_stop", timeout_ms=1000)


def test_menu_exit_no_restores_queue_without_blocking_compute(shell, qapp, monkeypatch):
    from foam_cell_analysis.gui import training_runner as training_module

    real_fake_job = training_module.FakeTrainingJob
    monkeypatch.setattr(
        training_module,
        "FakeTrainingJob",
        lambda *args, **kwargs: real_fake_job(*args, interval_ms=10_000, **kwargs),
    )
    monkeypatch.setenv("FOAM_MOCK_SPEED", "1")
    config = shell.ctx.backend.default_experiment_config("mask_rcnn")
    config["training"]["epochs"] = 4
    shell.ctx.backend.add_training_queue_item(config)
    shell.navigate(PageId.TRAINING_QUEUE)
    QTest.mouseClick(shell.page(PageId.TRAINING_QUEUE).run_button, Qt.MouseButton.LeftButton)
    assert shell.ctx.queue_controller.executing
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )

    _click_file_menu_exit(shell, qapp)

    assert shell.home.isVisible()
    assert shell.ctx.queue_controller.executing
    assert not shell.ctx.compute.is_blocked
    monkeypatch.undo()
    shell.ctx.queue_controller.stop_now()


def test_reopening_mode_uses_single_window(shell):
    shell.navigate(PageId.TRAINING)
    window = shell.manager.window(ModeId.TRAINING)
    shell.navigate(PageId.EXPERIMENTS)
    assert shell.manager.window(ModeId.TRAINING) is window
    assert window.tabs.currentIndex() == 2


def test_closing_each_mode_restores_home_and_allows_reopen(shell, qapp):
    for mode, pages in MODE_PAGES.items():
        shell.navigate(pages[0])
        window = shell.manager.window(mode)
        if mode == ModeId.DATA_PREPARATION:
            shell.home.showMinimized()
        else:
            shell.home.hide()
        window.close()
        qapp.processEvents()
        assert shell.home.isVisible()
        assert not shell.home.isMinimized()
        shell.navigate(pages[0])
        qapp.processEvents()
        assert window.isVisible()


def test_mode_close_does_not_restore_home_during_application_shutdown(shell, qapp):
    shell.navigate(PageId.TRAINING)
    window = shell.manager.window(ModeId.TRAINING)
    shell.home.hide()
    shell.manager._shutdown_requested = True
    window.close()
    qapp.processEvents()
    assert not shell.home.isVisible()


def test_navigation_passes_params_once(shell):
    calls = []
    page = shell.page(PageId.MASK_COMPARISON)
    page.on_enter = calls.append
    params = {"candidate_ids": ["RC-001", "RC-002"]}
    shell.ctx.navigator.navigate(PageId.MASK_COMPARISON, **params)
    assert shell.current_page() is page
    assert calls == [params]


def test_clicking_each_mode_tab_creates_page_and_enters_once(shell, qapp, monkeypatch):
    shell.manager.page(PageId.DATA_PREPARATION)
    calls = {page_id: 0 for page_id in PageId}
    for page_id, page_type in PAGE_TYPES.items():
        original = page_type.on_enter

        def counted(page, params, *, _original=original, _page_id=page_id):
            calls[_page_id] += 1
            _original(page, params)

        monkeypatch.setattr(page_type, "on_enter", counted)

    for mode, pages in MODE_PAGES.items():
        shell.manager.navigate(pages[0], {})
        window = shell.manager.window(mode)
        window.showNormal()
        qapp.processEvents()
        for index, page_id in enumerate(pages):
            if index != window.tab_bar.currentIndex():
                QTest.mouseClick(
                    window.tab_bar,
                    Qt.MouseButton.LeftButton,
                    pos=window.tab_bar.tabRect(index).center(),
                )
                qapp.processEvents()
            assert window.stack.currentWidget() is shell.manager.page(page_id)
            assert shell.manager.current_page_id(mode) == page_id

    assert calls == {page_id: 1 for page_id in PageId}


def test_clicking_empty_mask_comparison_tab_shows_candidate_guidance(shell, qapp):
    shell.navigate(PageId.CANDIDATES)
    window = shell.manager.window(ModeId.COMPARISON)
    QTest.mouseClick(
        window.tab_bar,
        Qt.MouseButton.LeftButton,
        pos=window.tab_bar.tabRect(1).center(),
    )
    qapp.processEvents()
    page = shell.page(PageId.MASK_COMPARISON)
    assert page.placeholder.isVisible()
    assert page.placeholder.text() == "候補一覧で比較する候補を選んでください"


def test_cross_mode_navigation_selects_expected_window_and_tab(shell):
    shell.navigate(PageId.EXPERIMENTS)
    shell.ctx.navigator.navigate(PageId.CANDIDATES)
    comparison = shell.manager.window(ModeId.COMPARISON)
    assert comparison is not None
    assert comparison.tabs.currentIndex() == 0
    assert shell.manager.current_page_id(ModeId.COMPARISON) == PageId.CANDIDATES
    shell.ctx.navigator.navigate(PageId.RELEASED_MODELS)
    assert comparison.tabs.currentIndex() == 2


def test_inference_routes_to_comparison_window(shell):
    shell.navigate(PageId.INFERENCE)
    shell.ctx.navigator.navigate(PageId.RELEASED_MODELS)
    assert shell.manager.window(ModeId.COMPARISON).tabs.currentIndex() == 2


def test_ctrl_h_brings_home_forward(shell, qtbot):
    shell.navigate(PageId.TRAINING)
    window = shell.manager.window(ModeId.TRAINING)
    window.activateWindow()
    qtbot.keyClick(window, Qt.Key.Key_H, modifier=Qt.KeyboardModifier.ControlModifier)
    assert shell.home.isVisible()
    assert all(not hasattr(shell.manager.window(mode), "home_button") for mode in MODE_PAGES)


def test_home_summary_values_match_backend(shell):
    summary = build_home_summary(shell.ctx.backend, shell.ctx.jobs)
    train = shell.ctx.backend.summarize_working_changes("train")
    val = shell.ctx.backend.summarize_working_changes("val")
    expected = sum(
        int(data[key]) for data in (train, val) for key in ("added", "removed", "changed")
    )
    assert summary.unconfirmed_changes == expected
    assert summary.candidate_count == len(shell.ctx.backend.list_candidates())
    assert summary.released_count == len(shell.ctx.backend.list_released_models())
    assert summary.routing == shell.ctx.backend.get_routing()


def test_window_activation_refreshes_the_active_page(shell, qapp):
    shell.navigate(PageId.CANDIDATES)
    page = shell.page(PageId.CANDIDATES)
    window = shell.manager.window(ModeId.COMPARISON)
    for item in shell.ctx.backend.get_working_items():
        if item.usage in {"train", "val"}:
            item.classification = item.classification or "分類A"
            item.quality = item.quality or "良"
            if not item.mask_revisions:
                item.mask_revisions = ["rev_001"]
                item.selected_mask_revision = "rev_001"
    train_item = next(
        item for item in shell.ctx.backend.get_working_items() if item.usage == "train"
    )
    shell.ctx.backend.update_item("all", train_item.item_id, usage="val")
    shell.ctx.backend.finalize_working_dataset("ウィンドウ再表示の確認")
    assert page.validation.findText("val_v004") < 0
    QTest.qWait(550)
    qapp.sendEvent(window, QEvent(QEvent.Type.WindowDeactivate))
    qapp.sendEvent(window, QEvent(QEvent.Type.WindowActivate))
    qapp.processEvents()
    assert page.validation.findText("val_v004") >= 0


def test_reopening_current_mode_from_home_keeps_active_tab(shell, qapp):
    shell.navigate(PageId.EXPERIMENTS)
    window = shell.manager.window(ModeId.TRAINING)
    QTest.mouseClick(shell.home.pipeline._stages[1], Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert window.tabs.currentIndex() == 2
    assert shell.manager.current_page_id(ModeId.TRAINING) == PageId.EXPERIMENTS


def test_home_refreshes_progress_from_started_training(shell, qapp, monkeypatch):
    from foam_cell_analysis.gui import training_runner as training_module

    real_fake_job = training_module.FakeTrainingJob

    def quick_job(*args, **kwargs):
        kwargs["interval_ms"] = 20
        return real_fake_job(*args, **kwargs)

    monkeypatch.setattr(training_module, "FakeTrainingJob", quick_job)
    shell.navigate(PageId.TRAINING)
    page = shell.page(PageId.TRAINING)
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
        QTest.mouseClick(page.start_button, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert shell.ctx.jobs.running_count == 1
    QTest.qWait(60)
    qapp.processEvents()
    assert shell.home._summary.running_experiment
    assert shell.home._summary.running_epoch >= 1
    assert shell.home.pipeline._values[1].text() == str(shell.home._summary.running_epoch)
    shell.ctx.training_runner.job.cancel()
