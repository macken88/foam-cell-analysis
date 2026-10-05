"""Qt アプリケーションの構築。"""

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtCore import QLibraryInfo, QLockFile, QTranslator
from PySide6.QtWidgets import QApplication, QMessageBox

from .gui.context import AppContext, StatusBus
from .gui.home_window import HomeWindow
from .gui.jobs import JobManager
from .gui.navigation import Navigator
from .gui.theme import apply_theme
from .gui.window_manager import WindowManager
from .services.hybrid_backend import HybridBackend
from .services.mock.backend import MockBackend

RECOVERY_BLOCK_MESSAGE = (
    "前回の学習・評価のプロセスが残っているため、新しい学習・評価を開始できません。"
    "タスクマネージャーで終了してから、アプリを再起動してください。"
)


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
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--backend", choices=("mock", "hybrid"))
    args, qt_args = parser.parse_known_args(sys.argv[1:])
    backend_name = args.backend or os.environ.get("FOAM_BACKEND", "hybrid")
    if backend_name not in {"mock", "hybrid"}:
        parser.error("FOAM_BACKEND は mock または hybrid を指定してください")
    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    workspace = Path(__file__).resolve().parents[2] / "workspace"
    lock = None
    blockers: list[str] = []
    if backend_name == "hybrid":
        workspace.mkdir(parents=True, exist_ok=True)
        lock = QLockFile(str(workspace / ".app.lock"))
        if not lock.tryLock(0):
            QMessageBox.critical(
                None, "起動できません", "別のアプリがこの workspace を使用中です。"
            )
            return 2
        try:
            backend = HybridBackend(workspace)
            # 学習（5.4）・比較とリリース（13.3）・評価（7.6）の起動時の復旧
            backend.recover()
        except Exception as error:
            QMessageBox.critical(
                None, "復旧できません", f"学習・評価の状態の復旧に失敗しました: {error}"
            )
            lock.unlock()
            return 2
        blockers = list(getattr(backend, "recovery_blockers", []) or [])
        report = backend.recovery_report()
        has_issues = any(report.get(kind) for kind in ("training", "comparison"))
        if blockers:
            # 評価・学習プロセスを終了できなかった。利用者に終了してから再起動するよう案内する
            QMessageBox.warning(None, "終了できない処理があります", "\n\n".join(blockers))
        if has_issues:
            QMessageBox.warning(
                None,
                "一部の記録を復旧できません",
                "問題がある記録は保護して一覧に残しました。対象の編集や削除はできません。",
            )
    else:
        backend = MockBackend()
    jobs = JobManager()
    navigator = Navigator()
    context = AppContext(backend=backend, navigator=navigator, jobs=jobs, status=StatusBus())
    context.workspace_lock = lock
    if blockers:
        # 前回のプロセスが残っている間は、新しい学習・評価を開始しない（比較・推論設計 15.1）
        context.compute.block(RECOVERY_BLOCK_MESSAGE)
    install_translations(app)
    apply_style(app)
    manager = WindowManager(context)
    window = HomeWindow(context, manager)
    window.show()
    return app.exec()
