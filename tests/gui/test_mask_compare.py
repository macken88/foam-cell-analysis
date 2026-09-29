"""マスク比較（比較・評価設計 11 章）: 評価の固定、確定版の原画像、未評価の拒否。"""

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from foam_cell_analysis.gui.context import AppContext
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.modes.comparison.mask_compare import MaskComparisonPage
from foam_cell_analysis.gui.navigation import Navigator
from foam_cell_analysis.services.mock.backend import MockBackend


def _page(qtbot, backend):
    page = MaskComparisonPage(AppContext(backend, Navigator(), JobManager()))
    qtbot.addWidget(page)
    page.resize(1200, 700)
    page.show()
    QTest.qWaitForWindowExposed(page)
    return page


def test_opens_with_fixed_evaluations_and_reads_finalized_images(qtbot, monkeypatch):
    backend = MockBackend()
    image_calls, prediction_calls = [], []
    original_image = backend.get_dataset_item_image
    original_prediction = backend.get_candidate_prediction

    def dataset_image(version, item_id, channel):
        image_calls.append((version, item_id))
        return original_image(version, item_id, channel)

    def prediction(candidate_id, evaluation_id, item_id):
        prediction_calls.append((candidate_id, evaluation_id, item_id))
        return original_prediction(candidate_id, evaluation_id, item_id)

    def working_image(*_args):
        raise AssertionError("作業中データの画像を読んではいけない")

    monkeypatch.setattr(backend, "get_dataset_item_image", dataset_image)
    monkeypatch.setattr(backend, "get_candidate_prediction", prediction)
    monkeypatch.setattr(backend, "get_item_image", working_image)
    expected = {
        candidate_id: backend.get_candidate_evaluation(candidate_id, "val_v003").evaluation_id
        for candidate_id in ("RC-001", "RC-002")
    }
    page = _page(qtbot, backend)
    page.on_enter({"validation_version": "val_v003", "candidate_ids": ["RC-001", "RC-002"]})

    assert page.placeholder.isHidden()
    assert page.evaluation_ids == expected
    assert len(page.views) == 3
    assert all(not view.scene().items() == [] for view in page.views)
    first = page.items[page.index].item_id
    QTest.mouseClick(page.next, Qt.MouseButton.LeftButton)
    second = page.items[page.index].item_id
    assert second != first
    assert ("val_v003", second) in image_calls
    assert ("RC-001", expected["RC-001"], second) in prediction_calls
    assert ("RC-002", expected["RC-002"], second) in prediction_calls
    assert all(view.scene().items() for view in page.views)


def test_refuses_candidate_not_evaluated_on_version(qtbot):
    backend = MockBackend()
    page = _page(qtbot, backend)
    page.on_enter({"validation_version": "val_v003", "candidate_ids": ["RC-001", "RC-003"]})

    assert page.placeholder.isVisible()
    assert "RC-003 はこの検証用データセットで未評価です" in page.placeholder.text()
    assert page.views == []
    assert page.items == []
    assert not page.next.isEnabled()
