"""本番推論画面のテスト。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FOAM_MOCK_SPEED", "1000")

import pytest
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from foam_cell_analysis.gui.context import AppContext, StatusBus
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.modes.inference.page import InferencePage
from foam_cell_analysis.gui.navigation import Navigator
from foam_cell_analysis.services.mock.backend import MockBackend


@pytest.fixture
def qapp():
    """共有 QApplication を返す。"""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mock_backend():
    """新しい MockBackend を返す。"""
    return MockBackend()


def make_page(qapp, mock_backend):
    """テスト用推論画面を作る。"""
    context = AppContext(mock_backend, Navigator(), JobManager(), StatusBus())
    page = InferencePage(context)
    page.show()
    qapp.processEvents()
    return page


def test_classification_change_updates_model_immediately(qapp, mock_backend):
    page = make_page(qapp, mock_backend)
    page.add_images(["sample.png"])
    assert page.inputs[0].model_id == mock_backend.get_routing()["分類A"]

    page.table.cellWidget(0, 1).setCurrentText("分類B")

    assert page.inputs[0].model_id == mock_backend.get_routing()["分類B"]
    assert page.table.item(0, 2).text() == mock_backend.get_routing()["分類B"]
    page.close()


def test_unassigned_classification_disables_run(qapp, mock_backend):
    page = make_page(qapp, mock_backend)
    page.add_images(["sample.png"])
    mock_backend.apply_routing({"分類A": None})

    page.refresh_routing()

    assert not page.run_button.isEnabled()
    assert page.route_button.isVisible()
    assert page.error_banner.isVisible()
    assert "分類A" in page.error_text.text()
    assert page.run_button.text() == "推論を実行（0 枚）"
    page.close()


def test_run_completes_all_rows(qapp, mock_backend):
    page = make_page(qapp, mock_backend)
    page.output_path = "mock-output"
    page.add_images(["first.png", "second.png"])
    loop = QEventLoop()
    page.ctx.jobs.jobs_changed.connect(lambda count: loop.quit() if count == 0 else None)
    QTimer.singleShot(3000, loop.quit)

    page.run_inference()
    loop.exec()

    assert [entry.status for entry in page.inputs] == ["完了", "完了"]
    assert all(entry.labels is not None for entry in page.inputs)
    page.close()


def test_cancel_marks_running_row_interrupted_and_keeps_pending_rows(qapp, mock_backend):
    page = make_page(qapp, mock_backend)
    page.output_path = "mock-output"
    page.add_images(["first.png", "second.png"])
    page.run_inference()
    job = page.ctx.jobs.jobs()[0]
    job._advance()

    job.cancel()

    assert [entry.status for entry in page.inputs] == ["中断", "待機"]
    page.close()


def test_result_preview_mode_shortcut_changes_display_and_status(qapp, mock_backend):
    page = make_page(qapp, mock_backend)
    messages = []
    page.ctx.status.message.connect(messages.append)
    page.image_view.setFocus()
    QTest.keyClick(page.image_view, Qt.Key.Key_M)
    assert not page.display_toggle.is_alternate
    QTest.keyClick(page.image_view, Qt.Key.Key_M)
    assert page.display_toggle.is_alternate
    page.close()
