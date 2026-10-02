"""numpy 画像を Qt 画像形式に変換する。"""

from enum import StrEnum

import numpy as np
from PySide6.QtGui import QImage, QPixmap

from foam_cell_analysis.inference.particle_split import binary_mask


class DisplayMode(StrEnum):
    """画像表示形式。"""

    IMAGE = "image"
    OVERLAY = "overlay"
    INSTANCE_LABEL = "instance_label"
    BINARY = "binary"


def to_binary_separated(labels: np.ndarray) -> np.ndarray:
    """粒子分離後の前景を 1、背景を 0 とする uint8 画像を返す（定義は inference 側）。"""
    return (binary_mask(labels) > 0).astype(np.uint8)


def _label_color(value: int) -> tuple[int, int, int]:
    """ラベル値から再現可能な明るい RGB 色を作る。"""
    red = (value * 67 + 53) % 206 + 50
    green = (value * 131 + 97) % 206 + 50
    blue = (value * 197 + 29) % 206 + 50
    return red, green, blue


# 表示用の階調変換で伸ばす範囲（パーセンタイル）。モデル入力の正規化とは別物
DISPLAY_PERCENTILES = (0.5, 99.5)


def to_display_uint8(image: np.ndarray) -> np.ndarray:
    """画面表示用に 0〜255 の uint8 へ変換する。

    uint8 はそのまま返す。uint16 や浮動小数などは 0.5〜99.5 パーセンタイルを
    0〜255 に伸ばす（表示だけに使い、モデル入力の正規化には使わない）。
    """
    source = np.asarray(image)
    if source.dtype == np.uint8:
        return source
    if source.dtype == np.bool_:
        return source.astype(np.uint8) * 255
    values = source.astype(np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(source.shape, dtype=np.uint8)
    low, high = np.percentile(values[finite], DISPLAY_PERCENTILES)
    if high <= low:
        # 一様な画像は伸ばせないので、値があれば中間の明るさ、なければ黒で表示する
        level = 0 if low <= 0 else 128
        result = np.full(source.shape, level, dtype=np.uint8)
        result[~finite] = 0
        return result
    scaled = (np.where(finite, values, low) - low) * (255.0 / (high - low))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def render(
    image: np.ndarray,
    labels: np.ndarray | None,
    mode: DisplayMode,
) -> np.ndarray:
    """画像とラベルを指定表示形式の RGB 配列へ変換する。"""
    source = np.asarray(image)
    if source.ndim == 2:
        rgb = np.repeat(source[:, :, None], 3, axis=2)
    elif source.ndim == 3 and source.shape[2] >= 3:
        rgb = source[:, :, :3].copy()
    else:
        raise ValueError("画像は HxW または HxWx3 の配列が必要です")
    rgb = to_display_uint8(rgb)
    if mode == DisplayMode.IMAGE or labels is None:
        return rgb

    instances = np.asarray(labels)
    if instances.shape != rgb.shape[:2]:
        raise ValueError("画像とラベルのサイズが一致しません")
    if mode == DisplayMode.BINARY:
        binary = to_binary_separated(instances)
        return np.repeat((binary * 255)[:, :, None], 3, axis=2)

    colored = np.zeros_like(rgb)
    for label in np.unique(instances):
        if label > 0:
            colored[instances == label] = _label_color(int(label))
    if mode == DisplayMode.INSTANCE_LABEL:
        return colored

    result = rgb.copy()
    mask = instances > 0
    result[mask] = (0.6 * rgb[mask] + 0.4 * colored[mask]).astype(np.uint8)
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= instances[1:, :] != instances[:-1, :]
    boundary[:-1, :] |= instances[:-1, :] != instances[1:, :]
    boundary[:, 1:] |= instances[:, 1:] != instances[:, :-1]
    boundary[:, :-1] |= instances[:, :-1] != instances[:, 1:]
    boundary &= mask
    result[boundary] = (0.25 * colored[boundary]).astype(np.uint8)
    return result


def array_to_qimage(array: np.ndarray) -> QImage:
    """uint8 グレースケールまたは RGB 配列を QImage にする。"""
    data = np.ascontiguousarray(array, dtype=np.uint8)
    if data.ndim == 2:
        return QImage(
            data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_Grayscale8
        ).copy()
    if data.ndim == 3 and data.shape[2] == 3:
        return QImage(
            data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_RGB888
        ).copy()
    raise ValueError("画像配列は HxW または HxWx3 である必要があります")


def array_to_pixmap(array: np.ndarray) -> QPixmap:
    """numpy 配列から QPixmap を作る。"""
    return QPixmap.fromImage(array_to_qimage(array))
