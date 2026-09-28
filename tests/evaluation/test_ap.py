import numpy as np
import pytest

from foam_cell_analysis.evaluation.ap import THRESHOLDS, aggregate, match_counts


def test_aggregate_image_mean_pool_reference_and_empty_classes():
    perfect = {"item_id": "a", "tp": np.ones(10), "fp": np.zeros(10), "fn": np.zeros(10)}
    missed = {"item_id": "b", "tp": np.zeros(10), "fp": np.zeros(10), "fn": np.ones(10)}
    result = aggregate([perfect, missed], {"a": "A", "b": "A"}, ["A", "B"])
    assert result["ap"] == 0.5
    assert result["per_class"]["A"] == {"ap": 0.5, "n_images": 2}
    assert result["per_class"]["B"] == {"ap": None, "n_images": 0}
    assert result["pooled_ap_reference"] == 0.5
    assert aggregate([])["ap"] is None
    empty = {"item_id": "empty", "tp": np.zeros(10), "fp": np.zeros(10), "fn": np.zeros(10)}
    assert aggregate([empty])["ap"] == 1.0


def test_image_mean_is_distinct_from_pooled_reference_and_empty_classification():
    accurate = {"item_id": "a", "tp": np.ones(10), "fp": np.zeros(10), "fn": np.zeros(10)}
    noisy = {
        "item_id": "b",
        "tp": np.zeros(10),
        "fp": np.full(10, 4),
        "fn": np.ones(10),
    }
    result = aggregate([accurate, noisy], {"a": "A", "b": "B"}, ["A", "B", "C"])
    assert result["ap"] == 0.5
    assert result["pooled_ap_reference"] == pytest.approx(1 / 6)
    assert result["per_class"]["C"] == {"ap": None, "n_images": 0}


@pytest.mark.ml
def test_match_counts_handles_noncontiguous_labels_and_empty_rules():
    pytest.importorskip("cellpose")
    # 背景画素のない画像も外周 0 付けで扱える。
    true = np.full((2, 2), 4, dtype=np.uint16)
    tp, fp, fn = match_counts(true, true.copy())
    assert tp.tolist() == [1] * len(THRESHOLDS)
    assert not fp.any() and not fn.any()
    tp, fp, fn = match_counts(np.zeros((2, 2), dtype=np.uint16), np.zeros((2, 2), dtype=np.uint16))
    assert not tp.any() and not fp.any() and not fn.any()
    tp, fp, fn = match_counts(np.zeros((2, 2), dtype=np.uint16), np.ones((2, 2), dtype=np.uint16))
    assert not tp.any() and fp.tolist() == [1] * len(THRESHOLDS) and not fn.any()
