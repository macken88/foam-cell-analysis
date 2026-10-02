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


def to_model_channels(normalized: np.ndarray, model_type: str) -> np.ndarray:
    """正規化済み 2 次元画像をモデル固有の入力形式へ変換する（拡張なし）。

    Mask R-CNN は同じ画像を 3 チャンネルへ複製し、Cellpose は
    ``[画像, 0, 0]`` を最後の軸へ並べる。fake 系は 2 次元のまま返す。
    """
    image = np.asarray(normalized, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("正規化済み画像は 2 次元である必要があります")
    if model_type == "mask_rcnn":
        return np.ascontiguousarray(np.repeat(image[:, :, None], 3, axis=2))
    if model_type == "cellpose":
        zeros = np.zeros_like(image)
        return np.stack((image, zeros, zeros), axis=-1).astype(np.float32, copy=False)
    if model_type in {"fake", "fake_numpy"}:
        return image
    raise ValueError(f"未対応の model_type です: {model_type}")


def prepare_model_input(
    image: np.ndarray, normalization: dict[str, Any] | None, model_type: str
) -> np.ndarray:
    """生画像を正規化し、モデル固有の入力形式へ変換する。

    学習時の検証（adapter.prepare_sample(training=False)）と推論で共有する。
    """
    return to_model_channels(normalize_image(image, normalization), model_type)
