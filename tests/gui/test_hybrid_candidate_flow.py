"""hybrid の比較候補（ComparisonService の実データ）を比較画面に表示する。"""

from foam_cell_analysis.gui.navigation import PageId
from foam_cell_analysis.services.hybrid_backend import HybridBackend
from tests.gui.test_stage_f_acceptance import _make_shell
from tests.services.test_comparison_service import FakeTraining, _dataset, _write_evaluation


def test_snapshot_candidate_oof_and_evaluation_render_without_experiment_lookup(qapp, tmp_path):
    _dataset(tmp_path, "train_v000", "train", ["分類A", "分類B"], "val_v000")
    _dataset(tmp_path, "val_v000", "val", ["分類A", "分類C"])
    fake = FakeTraining(tmp_path)
    fake.add_attempt("exp_snapshot_001", 2)
    backend = HybridBackend(tmp_path, process_alive=lambda _record: False)
    backend.training.create_candidate_snapshot = fake.create_candidate_snapshot
    config = backend.create_inference_config(
        "mask_rcnn", backend.default_inference_params("mask_rcnn")
    )
    candidate = backend.add_candidate("exp_snapshot_001", 2, config.config_id, "固定試行")
    context, manager, home = _make_shell(qapp, tmp_path, backend)
    context.navigator.navigate(PageId.CANDIDATES)
    page = manager.page(PageId.CANDIDATES)
    assert page.table.rowCount() == 1
    assert page.table.item(0, 8).text() != "—"
    assert page.table.item(0, 6).text() == "val_v000"

    _write_evaluation(backend.comparison, candidate.candidate_id, "eval_001")
    page.refresh()
    assert page.table.item(0, 7).text() != "未評価"
    assert page.table.item(0, 8).text() != "—"
    backend.release_candidate(candidate.candidate_id, "eval_001", "ready")
    assert backend.get_candidate(candidate.candidate_id).status == "released"
    home.hide()
    for window in manager._windows.values():
        window.close()
