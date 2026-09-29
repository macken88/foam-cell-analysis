"""学習・検証に共通する画像正規化。"""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_image(image: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    """percentile 正規化し、定数画像をゼロ画像として返す。"""
    options = config or {}
    low = float(options.get("low_percentile", 1.0))
    high = float(options.get("high_percentile", 99.0))
    if not 0 <= low <= high <= 100:
        raise ValueError("percentile の範囲が不正です")
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("画像は 2 次元である必要があります")
    lo, hi = np.percentile(values, (low, high))
    if hi - lo < 1e-6:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def preprocess(image: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    """設計書の共通前処理 API。"""
    return normalize_image(image, config)
