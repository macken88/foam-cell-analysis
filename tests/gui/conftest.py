"""GUI テストの共通環境。"""

import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FOAM_MOCK_SPEED", "1000")

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QSettings
from PySide6.QtWidgets import QApplication

from foam_cell_analysis.gui.context import AppContext, StatusBus
from foam_cell_analysis.gui.home_window import HomeWindow
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.navigation import Navigator
from foam_cell_analysis.gui.settings import SETTINGS_FILE_ENV
from foam_cell_analysis.gui.shortcuts import ShortcutMap
from foam_cell_analysis.gui.window_manager import WindowManager
from foam_cell_analysis.services.mock.backend import MockBackend

_unfinished_job_contexts = []


def _worker_is_running(job):
    """破棄済み WorkerJob を参照せず、稼働状態を返す。"""
    return shiboken6.isValid(job) and job.is_running()


@pytest.fixture
def qapp():
    """共有 QApplication を返す。"""
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_app_settings(tmp_path, monkeypatch):
    """テストごとに空の設定ファイルを使い、利用者の設定を読み書きしない。

    キー割り当て（%APPDATA%\foam-cell-analysis\\keymap.json）も一時フォルダへ向ける。
    """
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(tmp_path / "settings.ini"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    # 共有のサムネイル表示サイズも毎回作り直す（前のテストの段階を持ち越さない）
    from foam_cell_analysis.gui.modes.data_preparation import finalize_thumbnails

    monkeypatch.setattr(finalize_thumbnails, "_size_preference", None)


@pytest.fixture(autouse=True)
def fail_on_qt_slot_exceptions():
    """Qt スロットから伝播した Python 例外をテスト失敗にする。"""
    exceptions = []
    previous_hook = sys.excepthook

    def record_exception(exc_type, value, tb):
        exceptions.append("".join(traceback.format_exception(exc_type, value, tb)))

    sys.excepthook = record_exception
    yield
    sys.excepthook = previous_hook
    if exceptions:
        pytest.fail("Qt スロット内で例外が発生しました:\n" + "\n".join(exceptions))


@pytest.fixture
def mock_backend():
    """新しい MockBackend を返す。"""
    return MockBackend()


@pytest.fixture
def shell(qapp, mock_backend, tmp_path):
    """ホームとモードウィンドウを束ねたテスト用シェルを返す。"""
    context = AppContext(
        mock_backend,
        Navigator(),
        JobManager(),
        StatusBus(),
        ShortcutMap(tmp_path / "keymap.json"),
    )
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "tests", "foam")
    manager = WindowManager(context, settings)
    home = HomeWindow(context, manager)

    class Shell:
        def __init__(self):
            self.ctx = context
            self.manager = manager
            self.home = home

        def page(self, page_id):
            return manager.page(page_id)

        def navigate(self, page_id, **params):
            context.navigator.navigate(page_id, **params)

        def current_page(self):
            return manager.page(manager._last_page) if manager._last_page else None

        @property
        def job_count(self):
            return home.job_count

        @property
        def status_text(self):
            return home.status_text

        @property
        def autosave_text(self):
            mode = next(iter(manager._windows.values()), None)
            return mode.autosave_text if mode else None

    result = Shell()
    home.show()
    yield result
    unsafe_to_destroy = False
    workers = []
    try:
        from foam_cell_analysis.gui.modes.comparison.dialogs import WorkerJob

        context.queue_controller.stop_now()
        training_stopped = context.training_runner.request_stop("test_teardown", timeout_ms=3000)
        evaluation_stopped = context.evaluation_runner.request_stop(
            "test_teardown", timeout_ms=3000
        )
        jobs = tuple(context.jobs.jobs())
        workers = [job for job in jobs if isinstance(job, WorkerJob)]
        for job in jobs:
            job.cancel()
        worker_stop_results = [job._thread.wait(3000) for job in workers]
        qapp.processEvents()
        workers_stopped = all(
            stopped or not _worker_is_running(job)
            for job, stopped in zip(workers, worker_stop_results, strict=True)
        )

        unsafe_to_destroy = (
            context.training_runner.is_busy
            or context.evaluation_runner.is_busy
            or context.compute.active is not None
            or any(_worker_is_running(job) for job in workers)
        )
        failures = []
        if not training_stopped or context.training_runner.is_busy:
            failures.append("学習ジョブが終了しませんでした")
        if not evaluation_stopped or context.evaluation_runner.is_busy:
            failures.append("評価ジョブが終了しませんでした")
        if not workers_stopped or any(_worker_is_running(job) for job in workers):
            failures.append("ワーカースレッドが終了しませんでした")
        if context.compute.active is not None:
            failures.append("計算処理の占有が返却されませんでした")
        if context.compute.waiting:
            failures.append("計算処理の待機要求が残っています")
        if context.jobs.running_count:
            failures.append("実行中ジョブが残っています")
        if failures:
            unsafe_to_destroy = True
            raise RuntimeError("GUI テスト後片付けに失敗しました: " + "、".join(failures))
    finally:
        unsafe_to_destroy = unsafe_to_destroy or (
            context.training_runner.is_busy
            or context.evaluation_runner.is_busy
            or context.compute.active is not None
            or bool(context.compute.waiting)
            or bool(context.jobs.running_count)
            or any(_worker_is_running(job) for job in workers)
        )
        home.hide()
        for window in manager._windows.values():
            window.close()
            window.deleteLater()
        home.deleteLater()
        if unsafe_to_destroy:
            _unfinished_job_contexts.append(context)
        else:
            for obj in (
                manager,
                context.training_runner,
                context.evaluation_runner,
                context.jobs,
                context.compute,
                context.navigator,
                context.status,
                context.shortcuts,
                context.display,
            ):
                obj.deleteLater()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
