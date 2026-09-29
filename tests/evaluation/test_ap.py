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


def test_unequal_fold_sizes_use_image_mean_not_fold_mean():
    rows = [
        {"item_id": "a", "tp": np.ones(10), "fp": np.zeros(10), "fn": np.zeros(10)},
        {"item_id": "b", "tp": np.ones(10), "fp": np.zeros(10), "fn": np.zeros(10)},
        {"item_id": "c", "tp": np.zeros(10), "fp": np.zeros(10), "fn": np.ones(10)},
    ]
    result = aggregate(rows, {"a": "A", "b": "A", "c": "B"}, ["A", "B"])
    fold_mean = (1.0 + 0.0) / 2
    assert result["ap"] == pytest.approx(2 / 3)
    assert result["ap"] != fold_mean


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


def test_instance_table_overlap_noncontiguous_labels_and_empty():
    from foam_cell_analysis.evaluation.ap import instance_table

    truth = np.zeros((4, 4), dtype=np.int32)
    truth[0:2, 0:2] = 10  # 面積 4
    truth[3, 3] = 30  # 面積 1（予測なし）
    pred = np.zeros((4, 4), dtype=np.int32)
    pred[0:2, 0:1] = 7  # 面積 2、10 と交差 2 / 和 4 = 0.5
    pred[2, 0] = 8  # 面積 1、重なりなし
    rows = {(r["side"], r["label"]): r for r in instance_table(truth, pred)}
    assert len(rows) == 4
    assert rows[("true", 10)] == {
        "side": "true",
        "label": 10,
        "area": 4,
        "best_iou": 0.5,
        "best_label": 7,
    }
    assert rows[("true", 30)]["best_iou"] == 0.0 and rows[("true", 30)]["best_label"] is None
    assert rows[("pred", 7)]["best_label"] == 10 and rows[("pred", 7)]["area"] == 2
    assert rows[("pred", 8)]["best_label"] is None
    assert instance_table(np.zeros((3, 3)), np.zeros((3, 3))) == []
    only_pred = instance_table(np.zeros((2, 2), int), np.array([[0, 4], [4, 0]]))
    assert only_pred == [
        {"side": "pred", "label": 4, "area": 2, "best_iou": 0.0, "best_label": None}
    ]
