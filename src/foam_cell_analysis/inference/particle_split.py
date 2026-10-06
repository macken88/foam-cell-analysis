"""粒子分離（後処理）。Qt に依存しない固定アルゴリズム。

設計書: docs/design/comparison_inference_backend_design.html 10 章。
異なるラベルが 8 近傍で接する画素を、元のラベル画像で一斉に判定して両側とも背景化する。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from foam_cell_analysis.utils.backend_contracts import PARTICLE_SPLIT_ID

_EIGHT = np.ones((3, 3), dtype=bool)


def _boundary(values: np.ndarray) -> np.ndarray:
    """異なるラベルと 8 近傍で接する前景画素を True にする（元画像だけで判定）。"""
    boundary = np.zeros(values.shape, dtype=bool)
    height, width = values.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            a = values[y0:y1, x0:x1]
            b = values[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            # 各方向で自画素側だけを立てる。逆方向は別の (dy, dx) が処理する。
            boundary[y0:y1, x0:x1] |= (a > 0) & (b > 0) & (a != b)
    return boundary


def _as_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("ラベル画像は2次元である必要があります")
    return values


def split_labels(labels: np.ndarray) -> np.ndarray:
    """境界画素を 0 にした分離後ラベル画像 S を返す（入力は変更しない）。"""
    values = _as_labels(labels)
    return np.where(_boundary(values), 0, values).astype(values.dtype, copy=False)


def binary_mask(labels: np.ndarray) -> np.ndarray:
    """分離後の前景を 255、背景を 0 とする uint8 画像を返す。"""
    return ((split_labels(labels) > 0) * 255).astype(np.uint8)


def _component_counts(values: np.ndarray) -> dict[int, int]:
    """ラベルごとの 8 連結成分数を返す。"""
    counts: dict[int, int] = {}
    foreground = np.unique(values[values > 0])
    if foreground.size == 0:
        return counts
    # find_objects は 1..n の連番を要求するため、ラベルを詰め直す。
    compact = np.searchsorted(foreground, values) + 1
    compact[values <= 0] = 0
    regions = ndimage.find_objects(compact)
    for number, (label, region) in enumerate(zip(foreground, regions, strict=True), start=1):
        _, count = ndimage.label(compact[region] == number, structure=_EIGHT)
        counts[int(label)] = int(count)
    return counts


def split_report(labels: np.ndarray) -> dict:
    """分離前後の検査結果を返す（設計書 10.1）。"""
    values = _as_labels(labels)
    after = split_labels(values)
    foreground = int((values > 0).sum())
    removed = foreground - int((after > 0).sum())
    before_counts = _component_counts(values)
    after_counts = _component_counts(after)
    vanished = sorted(label for label in before_counts if label not in after_counts)
    split = sorted(
        label
        for label, count in after_counts.items()
        if before_counts.get(label) == 1 and count >= 2
    )
    return {
        "id": PARTICLE_SPLIT_ID,
        "remaining_contacts": int(_boundary(after).sum()),
        "vanished_labels": vanished,
        "split_labels": split,
        "removed_pixels": removed,
        "removed_fraction": removed / foreground if foreground else 0.0,
    }
