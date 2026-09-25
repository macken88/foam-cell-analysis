"""ホームとモードウィンドウの遷移・状態表示。"""

from datetime import datetime

from PySide6.QtCore import Qt

from foam_cell_analysis.gui.home_summary import build_home_summary
from foam_cell_analysis.gui.jobs import FakeJob
from foam_cell_analysis.gui.navigation import ModeId, PageId
from foam_cell_analysis.gui.window_manager import PAGE_TO_MODE_TAB, WindowManager


def test_every_page_can_be_opened(shell, qapp):
    for page_id in PageId:
        shell.ctx.navigator.navigate(page_id, probe=True)
        qapp.processEvents()
        mode = PAGE_TO_MODE_TAB[page_id][0]
        window = shell.manager.window(mode)
        assert window.stack.currentWidget() is shell.page(page_id)
        assert shell.manager.current_page_id(mode) == page_id


def test_reopening_mode_uses_single_window(shell):
    shell.navigate(PageId.TRAINING)
    window = shell.manager.window(ModeId.TRAINING)
    shell.navigate(PageId.EXPERIMENTS)
    assert shell.manager.window(ModeId.TRAINING) is window
    assert window.tabs.currentIndex() == 1


def test_navigation_passes_params_once(shell):
    calls = []
    page = shell.page(PageId.MASK_COMPARISON)
    page.on_enter = calls.append
    params = {"candidate_ids": ["RC-001", "RC-002"]}
    shell.ctx.navigator.navigate(PageId.MASK_COMPARISON, **params)
    assert shell.current_page() is page
    assert calls == [params]


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
    job = shell.ctx.jobs.start(FakeJob("確認", total_steps=100, key="training:exp_0046"))
    assert window.job_count.text() == "実行中ジョブ 1"
    assert shell.ctx.jobs.find("training:exp_0046") is job
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
