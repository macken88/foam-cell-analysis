"""GUI テストの共通環境。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FOAM_MOCK_SPEED", "1000")

import pytest
from PySide6.QtWidgets import QApplication

from foam_cell_analysis.gui.context import AppContext, StatusBus
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.main_window import MainWindow
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


@pytest.fixture
def main_window(qapp, mock_backend):
    """表示可能な MainWindow を返す。"""
    context = AppContext(mock_backend, Navigator(), JobManager(), StatusBus())
    window = MainWindow(context)
    yield window
    window.close()
