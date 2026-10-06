"""Capture native Windows screenshots of the in-memory mock application."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QSettings, Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QFileDialog

from foam_cell_analysis.app import install_translations
from foam_cell_analysis.gui.context import AppContext, StatusBus
from foam_cell_analysis.gui.home_window import ClickablePanel, HomeWindow
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.mode_window import ModeWindow
from foam_cell_analysis.gui.navigation import ModeId, Navigator, PageId
from foam_cell_analysis.gui.shortcuts import ShortcutMap
from foam_cell_analysis.gui.theme import apply_theme
from foam_cell_analysis.gui.window_manager import WindowManager
from foam_cell_analysis.services.mock.backend import MockBackend


def main() -> None:
    """Navigate real widgets and save screenshots without touching user data."""
    temp_root = Path(tempfile.mkdtemp(prefix="foam-manual-native-"))
    short_temp_root = Path("C:/Temp")
    created_temp_root = not short_temp_root.exists()
    short_temp_root.mkdir(exist_ok=True)
    output_folder = Path(tempfile.mkdtemp(prefix="fma-", dir=short_temp_root))
    os.environ["QT_QPA_PLATFORM"] = "windows"
    os.environ["FOAM_SETTINGS_FILE"] = str(temp_root / "settings.ini")
    os.environ["APPDATA"] = str(temp_root / "appdata")

    app = QApplication([])
    apply_theme(app)
    install_translations(app)
    backend = MockBackend()
    context = AppContext(
        backend,
        Navigator(),
        JobManager(),
        StatusBus(),
        ShortcutMap(temp_root / "keymap.json"),
    )

    original_init = ModeWindow.__init__

    def hidden_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)

    ModeWindow.__init__ = hidden_init
    manager = WindowManager(
        context,
        QSettings(str(temp_root / "windows.ini"), QSettings.Format.IniFormat),
    )
    home = HomeWindow(context, manager)
    home.resize(1320, 860)
    home.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    home.show()
    app.processEvents()

    output_root = Path(__file__).resolve().parents[1] / "assets" / "images"
    _save(home, output_root / "common" / "home.png")
    for mode, indexes, folder in (
        (ModeId.DATA_PREPARATION, (0, 1), "data-preparation"),
        (ModeId.TRAINING, (0, 1, 2), "training"),
        (ModeId.COMPARISON, (0, 2), "comparison-release"),
    ):
        panel = next(item for item in home.findChildren(ClickablePanel) if item.target == mode)
        QTest.mouseClick(panel, Qt.MouseButton.LeftButton)
        app.processEvents()
        window = manager.window(mode)
        window.resize(1320, 860)
        for index in indexes:
            if mode == ModeId.TRAINING and index == 1:
                training_page = manager.page(PageId.TRAINING)
                for _ in range(2):
                    QTest.mouseClick(training_page.queue_button, Qt.MouseButton.LeftButton)
                    app.processEvents()
                if len(context.backend.list_training_queue()) < 2:
                    raise RuntimeError("Could not add demo settings to the training queue")
            if mode == ModeId.TRAINING and index == 2:
                experiment_page = manager.page(PageId.EXPERIMENTS)
                row = next(
                    row
                    for row in range(experiment_page.table.rowCount())
                    if experiment_page.table.item(row, 1).text() == "exp_0042"
                )
                rect = experiment_page.table.visualItemRect(experiment_page.table.item(row, 1))
                QTest.mouseClick(
                    experiment_page.table.viewport(),
                    Qt.MouseButton.LeftButton,
                    pos=rect.center(),
                )
                app.processEvents()
            if mode == ModeId.COMPARISON and index == 2:
                window.resize(1320, 1040)
            QTest.mouseClick(
                window.tab_bar,
                Qt.MouseButton.LeftButton,
                pos=window.tab_bar.tabRect(index).center(),
            )
            app.processEvents()
            _save(window, output_root / folder / f"{mode.value}-{index}.png")
        home.show()
        home.raise_()
        app.processEvents()

    _capture_comparison(app, context, manager, output_root)
    _capture_comparison_dialogs(app, context, manager, output_root, output_folder)
    _capture_inference(app, context, manager, output_root, temp_root, output_folder)
    _capture_protected_experiment(app, context, manager, output_root)

    app.processEvents()
    for window in app.topLevelWidgets():
        window.hide()
        window.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.quit()
    output_folder.rmdir()
    if created_temp_root:
        try:
            short_temp_root.rmdir()
        except OSError:
            pass
    shutil.rmtree(temp_root)
    print("Native screen captures: OK")


def _save(widget, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not widget.grab().save(str(path)):
        raise RuntimeError(f"Could not save screenshot: {path}")
    print(path)


def _capture_inference(app, context, manager, output_root, temp_root, output_folder) -> None:
    context.navigator.navigate(PageId.INFERENCE)
    app.processEvents()
    page = manager.page(PageId.INFERENCE)
    window = manager.window(ModeId.INFERENCE)
    window.resize(1320, 860)
    sample = temp_root / "sample.png"
    sample_image = QImage(64, 64, QImage.Format.Format_RGB32)
    sample_image.fill(Qt.GlobalColor.lightGray)
    if not sample_image.save(str(sample)):
        raise RuntimeError(f"Could not create demo image: {sample}")
    QFileDialog.getOpenFileNames = lambda *_args: ([str(sample)], "")
    QTest.mouseClick(page.add_files_button, Qt.MouseButton.LeftButton)
    app.processEvents()

    state = context.backend.get_routing_state()
    original_routing = state.assignments.copy()
    changed = dict(original_routing)
    changed["分類A"] = None
    context.backend.apply_routing(changed, expected_revision=state.revision)
    page.refresh_routing()
    _save(window, output_root / "inference" / "inference-unassigned.png")

    state = context.backend.get_routing_state()
    changed = dict(context.backend.get_routing())
    changed["分類A"] = original_routing.get("分類A")
    context.backend.apply_routing(changed, expected_revision=state.revision)
    page.refresh_routing()
    QFileDialog.getExistingDirectory = lambda *_args: str(output_folder)
    QTest.mouseClick(page.browse_output_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(page.run_button, Qt.MouseButton.LeftButton)
    deadline = 10_000
    while context.jobs.running_count and deadline > 0:
        QTest.qWait(50)
        app.processEvents()
        deadline -= 50
    if context.jobs.running_count:
        raise TimeoutError("Mock inference did not finish")
    rect = page.table.visualItemRect(page.table.item(0, 0))
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=rect.center(),
    )
    if page.display_toggle.is_alternate:
        QTest.mouseClick(page.display_toggle.raw_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(page.display_toggle.alternate_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    _save(window, output_root / "inference" / "inference-result.png")


def _capture_comparison(app, context, manager, output_root) -> None:
    context.navigator.navigate(PageId.CANDIDATES)
    app.processEvents()
    window = manager.window(ModeId.COMPARISON)
    window.resize(1320, 860)
    page = manager.page(PageId.CANDIDATES)
    for candidate_id in ("RC-001", "RC-002"):
        row = next(
            index
            for index in range(page.table.rowCount())
            if page.table.item(index, 1).text() == candidate_id
        )
        rect = page.table.visualItemRect(page.table.item(row, 0))
        QTest.mouseClick(
            page.table.viewport(),
            Qt.MouseButton.LeftButton,
            pos=rect.topLeft() + QPoint(12, rect.height() // 2),
        )
    if not page.buttons["compare"].isEnabled():
        raise RuntimeError("Seeded evaluated candidates are not eligible for comparison")
    QTest.mouseClick(page.buttons["compare"], Qt.MouseButton.LeftButton)
    app.processEvents()
    _save(window, output_root / "comparison-release" / "comparison-1.png")


def _capture_comparison_dialogs(app, context, manager, output_root, output_folder) -> None:
    context.navigator.navigate(PageId.CANDIDATES)
    app.processEvents()
    page = manager.page(PageId.CANDIDATES)
    window = manager.window(ModeId.COMPARISON)
    window.resize(1320, 860)
    row = next(
        index
        for index in range(page.table.rowCount())
        if page.table.item(index, 1).text() == "RC-001"
    )
    rect = page.table.visualItemRect(page.table.item(row, 0))
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=rect.topLeft() + QPoint(12, rect.height() // 2),
    )
    app.processEvents()
    if not page.buttons["detail"].isEnabled():
        raise RuntimeError("Seeded candidate evaluation detail is not available")
    _capture_modal(
        app,
        page.buttons["detail"],
        output_root / "comparison-release" / "evaluation-detail.png",
        close_kind="evaluation",
    )
    if not page.buttons["export"].isEnabled():
        raise RuntimeError("Seeded candidate export is not available")
    _capture_modal(
        app,
        page.buttons["export"],
        output_root / "comparison-release" / "export-dialog.png",
        close_kind="export",
        output_folder=output_folder,
    )


def _capture_modal(app, trigger, path, *, close_kind, output_folder=None) -> None:
    captured = []

    def capture_and_close() -> None:
        dialog = app.activeModalWidget()
        if dialog is None:
            raise RuntimeError(f"Expected modal dialog for screenshot: {path}")
        dialog.resize(1200, 800)
        if close_kind == "export":
            original_dialog = QFileDialog.getExistingDirectory
            QFileDialog.getExistingDirectory = lambda *_args: str(output_folder)
            QTest.mouseClick(dialog.browse, Qt.MouseButton.LeftButton)
            QFileDialog.getExistingDirectory = original_dialog
            close_button = dialog.cancel_button
        else:
            close_button = dialog.controls.button(QDialogButtonBox.StandardButton.Close)
        app.processEvents()
        _save(dialog, path)
        captured.append(True)
        QTest.mouseClick(close_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(300, capture_and_close)
    QTest.mouseClick(trigger, Qt.MouseButton.LeftButton)
    app.processEvents()
    if not captured:
        raise RuntimeError(f"Modal screenshot was not captured: {path}")


def _capture_protected_experiment(app, context, manager, output_root) -> None:
    context.navigator.navigate(PageId.EXPERIMENTS)
    app.processEvents()
    page = manager.page(PageId.EXPERIMENTS)
    window = manager.window(ModeId.TRAINING)
    window.resize(1320, 860)
    experiments = context.backend.list_experiments()
    if not experiments:
        raise RuntimeError("Mock backend has no experiment for recovery screenshot")
    experiment = experiments[0]
    experiment.recovery_state = "unrecoverable"
    experiment.recovery_reason = "記録を確認できないため、一部の操作を制限しています。"
    page.refresh()
    context.status.show_message("実験一覧を更新しました")
    rect = page.table.visualItemRect(page.table.item(0, 0))
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=rect.center(),
    )
    app.processEvents()
    _save(window, output_root / "recovery" / "protected-experiment.png")


if __name__ == "__main__":
    main()
