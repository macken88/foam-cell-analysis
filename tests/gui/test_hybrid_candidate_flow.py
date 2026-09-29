"""スナップショット由来候補を比較画面に表示する。"""

from foam_cell_analysis.gui.navigation import PageId
from foam_cell_analysis.services.hybrid_backend import HybridBackend
from foam_cell_analysis.services.models import CandidateSnapshot
from tests.gui.test_stage_f_acceptance import _make_shell


def test_snapshot_candidate_oof_and_evaluation_render_without_experiment_lookup(qapp, tmp_path):
    backend = HybridBackend(tmp_path)
    snapshot = CandidateSnapshot(
        experiment_id="exp_snapshot_001",
        attempt=2,
        selected_epoch=3,
        run_id="exp_snapshot_001/attempt_002",
        checkpoint_path="checkpoints/final.pt",
        oof_evaluation={"ap": 0.64, "per_class": {"A": [0.64, 2]}, "n_images": 2},
        experiment_config={"model": {"type": "cellpose", "scale_range": 0.2}},
    )
    candidate = backend.mock.add_candidate_from_snapshot(snapshot, "infer_v006", "固定試行")
    context, manager, home = _make_shell(qapp, tmp_path, backend)
    context.navigator.navigate(PageId.CANDIDATES)
    page = manager.page(PageId.CANDIDATES)
    assert page.table.rowCount() == 1
    assert page.table.item(0, 2).text() == "Cellpose"
    assert page.table.item(0, 7).text() != "—"

    backend.mock.start_evaluation([candidate.candidate_id], "val_v003")
    backend.mock.evaluate_candidate(candidate.candidate_id, "val_v003")
    page.refresh()
    assert page.table.item(0, 6).text() != "未評価"
    assert page.table.item(0, 7).text() != "—"
    backend.mock.release_candidate(candidate.candidate_id, "ready", "val_v003")
    home.hide()
    for window in manager._windows.values():
        window.close()
