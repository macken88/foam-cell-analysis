"""GUI テストの共通環境。"""

import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FOAM_MOCK_SPEED", "1000")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from foam_cell_analysis.gui.context import AppContext, StatusBus
from foam_cell_analysis.gui.home_window import HomeWindow
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.navigation import Navigator
from foam_cell_analysis.gui.settings import SETTINGS_FILE_ENV
from foam_cell_analysis.gui.shortcuts import ShortcutMap
from foam_cell_analysis.gui.window_manager import WindowManager
from foam_cell_analysis.services.mock.backend import MockBackend


@pytest.fixture
def qapp():
    """共有 QApplication を返す。"""
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_app_settings(tmp_path, monkeypatch):
    """テストごとに空の設定ファイルを使い、利用者の設定を読み書きしない。"""
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(tmp_path / "settings.ini"))


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
    home.hide()
    for window in manager._windows.values():
        window.close()
