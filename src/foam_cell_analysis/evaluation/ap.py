"""Cellpose 方式の AP マッチングと numpy による集計。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
METRIC = {
    "id": "cellpose_ap_iou50_95_image_mean_v1",
    "thresholds": list(THRESHOLDS),
    "aggregation": "image_mean",
    "empty_rule": "both_empty_is_1",
}


def _compact(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("ラベル画像は 2 次元である必要があります")
    result = np.zeros(values.shape, dtype=np.int32)
    for number, label in enumerate(np.unique(values[values != 0]), start=1):
        result[values == label] = number
    return result


def match_counts(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """画像ごとの TP/FP/FN を IoU 0.50〜0.95 で数える。"""
    true_labels = _compact(truth)
    pred_labels = _compact(prediction)
    if true_labels.shape != pred_labels.shape:
        raise ValueError("正解と予測の画像サイズが一致しません")
    n_true = int(true_labels.max())
    n_pred = int(pred_labels.max())
    zeros = np.zeros(len(THRESHOLDS), dtype=np.int64)
    if n_true == 0 and n_pred == 0:
        return np.zeros_like(zeros), np.zeros_like(zeros), np.zeros_like(zeros)
    if n_true == 0:
        return zeros.copy(), np.full_like(zeros, n_pred), zeros.copy()
    if n_pred == 0:
        return zeros.copy(), zeros.copy(), np.full_like(zeros, n_true)

    from cellpose.metrics import average_precision

    true_padded = np.pad(true_labels, 1, mode="constant")
    pred_padded = np.pad(pred_labels, 1, mode="constant")
    result = average_precision(true_padded[None, ...], pred_padded[None, ...], list(THRESHOLDS))
    true_positive = np.asarray(result[1]).reshape(-1).astype(np.int64)
    false_positive = np.asarray(result[2]).reshape(-1).astype(np.int64)
    false_negative = np.asarray(result[3]).reshape(-1).astype(np.int64)
    return true_positive, false_positive, false_negative


def aggregate(
    image_counts: Sequence[Mapping[str, Any]],
    classifications: Mapping[str, str | None] | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """画像平均 AP、分類別 AP、参考プール値を集計する。"""
    classifications = classifications or {}
    per_image: dict[str, list[float]] = {}
    pooled_tp = np.zeros(len(THRESHOLDS), dtype=np.float64)
    pooled_fp = np.zeros(len(THRESHOLDS), dtype=np.float64)
    pooled_fn = np.zeros(len(THRESHOLDS), dtype=np.float64)
    for index, row in enumerate(image_counts):
        item_id = str(row.get("item_id", index))
        tp = np.asarray(row["tp"], dtype=np.float64)
        fp = np.asarray(row["fp"], dtype=np.float64)
        fn = np.asarray(row["fn"], dtype=np.float64)
        if not (tp.shape == fp.shape == fn.shape == (len(THRESHOLDS),)):
            raise ValueError("各画像の TP/FP/FN は 10 個必要です")
        denom = tp + fp + fn
        ap = np.divide(tp, denom, out=np.ones_like(tp), where=denom != 0).mean().item()
        per_image[item_id] = [float(ap), classifications.get(item_id)]
        pooled_tp += tp
        pooled_fp += fp
        pooled_fn += fn
    values = [row[0] for row in per_image.values()]
    per_class: dict[str, dict[str, float | int | None]] = {}
    names = sorted(set(class_names or ()) | {row[1] or "未分類" for row in per_image.values()})
    for name in names:
        scores = [score for score, label in per_image.values() if (label or "未分類") == name]
        per_class[name] = {
            "ap": float(np.mean(scores)) if scores else None,
            "n_images": len(scores),
        }
    denom = pooled_tp + pooled_fp + pooled_fn
    pooled = np.divide(pooled_tp, denom, out=np.ones_like(denom), where=denom != 0).mean().item()
    return {
        "ap": float(np.mean(values)) if values else None,
        "per_class": per_class,
        "n_images": len(values),
        "per_image": {key: value[0] for key, value in per_image.items()},
        "pooled_ap_reference": float(pooled) if values else None,
    }
