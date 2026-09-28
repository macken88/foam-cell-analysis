"""ホームとモードウィンドウの遷移・状態表示。"""

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from foam_cell_analysis.gui.home_summary import build_home_summary
from foam_cell_analysis.gui.jobs import FakeJob
from foam_cell_analysis.gui.navigation import ModeId, PageId
from foam_cell_analysis.gui.window_manager import (
    MODE_PAGES,
    PAGE_TO_MODE_TAB,
    PAGE_TYPES,
    WindowManager,
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

    file_menu = next(
        action.menu()
        for action in shell.home.menuBar().actions()
        if action.text().startswith("ファイル")
    )
    file_menu.popup(shell.home.mapToGlobal(shell.home.menuBar().rect().topLeft()))
    qapp.processEvents()
    exit_action = next(action for action in file_menu.actions() if action.text() == "終了")
    QTest.mouseClick(
        file_menu,
        Qt.MouseButton.LeftButton,
        pos=file_menu.actionGeometry(exit_action).center(),
    )
    qapp.processEvents()
    assert prompts == ["この設定で学習を開始しますか？", "学習を中断して終了しますか？"]
    assert shell.home.isVisible()
    shell.ctx.training_runner.job.kill()


def test_reopening_mode_uses_single_window(shell):
    shell.navigate(PageId.TRAINING)
    window = shell.manager.window(ModeId.TRAINING)
    shell.navigate(PageId.EXPERIMENTS)
    assert shell.manager.window(ModeId.TRAINING) is window
    assert window.tabs.currentIndex() == 2


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


def test_job_count_and_autosave_time_are_shown(shell):
    shell.navigate(PageId.TRAINING)
    window = shell.manager.window(ModeId.TRAINING)
    assert window.job_count.text() == "実行中ジョブ 0"
    job = shell.ctx.jobs.start(FakeJob("確認", total_steps=100, key="preview:exp_0046"))
    assert window.job_count.text() == "実行中ジョブ 1"
    assert shell.ctx.jobs.find("preview:exp_0046") is job
    saved_at = datetime.now().astimezone()
    shell.ctx.status.notify_saved(saved_at)
    assert window.autosave_text.text() == f"自動保存 {saved_at.strftime('%H:%M:%S')}"
    job.cancel()
    assert window.job_count.text() == "実行中ジョブ 0"


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


def test_mode_window_geometry_is_saved_and_restored(shell, qapp):
    shell.navigate(PageId.TRAINING)
    first = shell.manager.window(ModeId.TRAINING)
    first.showNormal()
    first.setGeometry(50, 60, 1024, 768)
    qapp.processEvents()
    first.close()
    saved = shell.manager.settings.value("windows/training/geometry")
    assert saved
    second_manager = WindowManager(shell.ctx, shell.manager.settings)
    second_manager.navigate(PageId.TRAINING)
    second = second_manager.window(ModeId.TRAINING)
    qapp.processEvents()
    assert second.geometry().size() == first.geometry().size()
