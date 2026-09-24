"""メインウィンドウの遷移と状態表示。"""

from foam_cell_analysis.gui.jobs import FakeJob
from foam_cell_analysis.gui.navigation import PageId


def test_every_page_can_be_opened(main_window, qapp):
    for page_id in PageId:
        main_window.ctx.navigator.navigate(page_id, probe=True)
        qapp.processEvents()
        assert main_window.stack.currentWidget() is main_window.pages[page_id]


def test_sidebar_tracks_navigation(main_window):
    assert main_window.sidebar.objectName() == "mainSidebar"
    assert main_window.sidebar.item(0).font().bold()
    main_window.ctx.navigator.navigate(PageId.CANDIDATES)
    assert main_window.sidebar.currentItem().text() == "モデル比較・リリース"
    main_window.ctx.navigator.navigate(PageId.MASK_COMPARISON)
    assert main_window.sidebar.currentItem().text() == "モデル比較・リリース"
    assert main_window.stack.currentWidget() is main_window.pages[PageId.MASK_COMPARISON]


def test_navigation_passes_params_once(main_window):
    calls = []
    main_window.pages[PageId.MASK_COMPARISON].on_enter = calls.append
    params = {"candidate_ids": ["RC-001", "RC-002"]}
    main_window.ctx.navigator.navigate(PageId.MASK_COMPARISON, **params)
    assert main_window.stack.currentWidget() is main_window.pages[PageId.MASK_COMPARISON]
    assert calls == [params]


def test_job_count_is_shown(main_window):
    assert main_window.job_count.text() == "実行中ジョブ: 0"
    job = main_window.ctx.jobs.start(FakeJob("確認", total_steps=100, key="training:exp_0046"))
    assert main_window.job_count.text() == "実行中ジョブ: 1"
    assert main_window.ctx.jobs.find("training:exp_0046") is job
    job.cancel()
    assert main_window.job_count.text() == "実行中ジョブ: 0"
    assert main_window.ctx.jobs.find("training:exp_0046") is None


def test_status_bus_updates_message_and_saved_time(main_window):
    from datetime import datetime

    main_window.ctx.status.show_message("保存しました")
    assert main_window.status_text.text() == "保存しました"
    saved_at = datetime.now().astimezone()
    main_window.ctx.status.notify_saved(saved_at)
    assert main_window.autosave_text.text().endswith(saved_at.strftime("%H:%M:%S"))


def test_completed_job_cancel_does_not_emit_finished_again(qtbot):
    job = FakeJob("完了", total_steps=1, interval_ms=5_000)
    finished = []
    job.finished.connect(lambda ok, message: finished.append((ok, message)))
    job._advance()
    job.cancel()
    assert len(finished) == 1
    assert finished[0][0]
