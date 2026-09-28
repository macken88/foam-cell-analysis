"""Qt アプリケーションの構築。"""

import sys

from PySide6.QtCore import QLibraryInfo, QTranslator
from PySide6.QtWidgets import QApplication

from .gui.context import AppContext, StatusBus
from .gui.home_window import HomeWindow
from .gui.jobs import JobManager
from .gui.navigation import Navigator
from .gui.theme import apply_theme
from .gui.window_manager import WindowManager
from .services.mock.backend import MockBackend


def install_translations(app: QApplication) -> None:
    """Qt 標準の日本語翻訳があればアプリへ登録する。"""
    translator = QTranslator(app)
    translation_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load("qtbase_ja", translation_path):
        app.installTranslator(translator)


def apply_style(app: QApplication) -> None:
    """アプリ共通の日本語 UI スタイルを設定する。"""
    apply_theme(app)


def main() -> int:
    """アプリを起動して終了コードを返す。"""
    app = QApplication.instance() or QApplication(sys.argv)
    backend = MockBackend()
    jobs = JobManager()
    navigator = Navigator()
    context = AppContext(backend=backend, navigator=navigator, jobs=jobs, status=StatusBus())
    install_translations(app)
    apply_style(app)
    manager = WindowManager(context)
    window = HomeWindow(context, manager)
    window.show()
    return app.exec()
