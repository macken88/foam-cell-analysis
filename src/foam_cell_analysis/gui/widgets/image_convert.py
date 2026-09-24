"""numpy 画像を Qt 画像形式に変換する。"""

from enum import StrEnum

import numpy as np
from PySide6.QtGui import QImage, QPixmap


class DisplayMode(StrEnum):
    """画像表示形式。"""

    IMAGE = "image"
    OVERLAY = "overlay"
    INSTANCE_LABEL = "instance_label"
    BINARY = "binary"


def to_binary_separated(labels: np.ndarray) -> np.ndarray:
    """異なるラベルが8近傍で接する両側ピクセルを背景化する。"""
    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("ラベル画像は2次元である必要があります")
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
            touching = (a > 0) & (b > 0) & (a != b)
            boundary[y0:y1, x0:x1] |= touching
            boundary[y0 + dy : y1 + dy, x0 + dx : x1 + dx] |= touching
    return ((values > 0) & ~boundary).astype(np.uint8)


def _label_color(value: int) -> tuple[int, int, int]:
    """ラベル値から再現可能な明るい RGB 色を作る。"""
    red = (value * 67 + 53) % 206 + 50
    green = (value * 131 + 97) % 206 + 50
    blue = (value * 197 + 29) % 206 + 50
    return red, green, blue


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
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
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
