"""Qt アプリケーションの構築。"""

import sys

from PySide6.QtCore import QLibraryInfo, QTranslator
from PySide6.QtWidgets import QApplication

from .gui.context import AppContext, StatusBus
from .gui.jobs import JobManager
from .gui.main_window import MainWindow
from .gui.navigation import Navigator
from .services.mock.backend import MockBackend


def install_translations(app: QApplication) -> None:
    """Qt 標準の日本語翻訳があればアプリへ登録する。"""
    translator = QTranslator(app)
    translation_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load("qtbase_ja", translation_path):
        app.installTranslator(translator)


def apply_style(app: QApplication) -> None:
    """アプリ共通の日本語 UI スタイルを設定する。"""
    app.setStyleSheet(
        'QWidget { font-family: "Yu Gothic UI", "Meiryo UI"; font-size: 10pt; }'
        "QLabel#pageHeading { font-size: 18pt; font-weight: 600; }"
        "QGroupBox { margin-top: 10px; padding-top: 8px; }"
        "QLineEdit:read-only, QSpinBox:read-only, QDoubleSpinBox:read-only "
        "{ background-color: #f0f0f0; }"
        "QPushButton { min-height: 28px; padding: 0 12px; }"
        'QPushButton[primary="true"] { background-color: #2867a5; color: white; border: 0; }'
        'QPushButton[primary="true"]:hover { background-color: #205889; }'
        'QPushButton[primary="true"]:pressed { background-color: #17466f; }'
        'QPushButton[primary="true"]:disabled { background-color: #aebdca; color: #f4f6f8; }'
        "QListWidget#mainSidebar { border: 0; }"
        "QListWidget#mainSidebar::item { padding: 6px 12px; }"
        "QListWidget#mainSidebar::item:selected { background-color: #dce8f3; color: #254f77; }"
        "QTableView { gridline-color: #d9dde2; }"
        "QTableView QHeaderView::section { background-color: #f1f3f5; font-weight: normal; }"
    )


def main() -> int:
    """アプリを起動して終了コードを返す。"""
    app = QApplication.instance() or QApplication(sys.argv)
    backend = MockBackend()
    jobs = JobManager()
    navigator = Navigator()
    context = AppContext(backend=backend, navigator=navigator, jobs=jobs, status=StatusBus())
    install_translations(app)
    apply_style(app)
    window = MainWindow(context)
    window.show()
    return app.exec()
