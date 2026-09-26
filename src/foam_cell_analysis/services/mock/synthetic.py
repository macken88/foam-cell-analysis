"""再現可能な気泡画像と予測マスクの生成。"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def _make_sample_cached(
    seed: int, size: int, channels: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    labels = np.zeros((size, size), dtype=np.int32)
    n = int(rng.integers(20, 35))
    centers: list[tuple[int, int, int]] = []
    for index in range(1, n + 1):
        radius = int(rng.integers(max(5, size // 45), max(7, size // 20)))
        cy, cx = int(rng.integers(radius, size - radius)), int(rng.integers(radius, size - radius))
        centers.append((cy, cx, radius))
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        labels[disk & (labels == 0)] = index
    contact_radius = max(5, size // 24)
    contact_y = size // 2
    first_x = size // 2 - contact_radius // 2
    second_x = first_x + 2 * contact_radius - 1
    first_disk = (yy - contact_y) ** 2 + (xx - first_x) ** 2 <= contact_radius**2
    second_disk = (yy - contact_y) ** 2 + (xx - second_x) ** 2 <= contact_radius**2
    labels[first_disk] = n + 1
    labels[second_disk] = n + 2
    centers.extend([(contact_y, first_x, contact_radius), (contact_y, second_x, contact_radius)])
    images: dict[str, np.ndarray] = {}
    for channel in channels:
        image = rng.normal(40, 7, (size, size))
        for cy, cx, radius in centers:
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            image[(dist < radius) & (dist > radius * 0.78)] += 145
            image[dist <= radius * 0.78] += 75
        images[channel] = np.clip(image, 0, 255).astype(np.uint8)
    return images, labels


def make_sample(
    seed: int, size: int = 512, channels: tuple[str, ...] = ("A",)
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """背景ノイズと気泡を含む画像・インスタンスラベルを生成する。"""
    images, labels = _make_sample_cached(seed, size, tuple(channels))
    return {channel: image.copy() for channel, image in images.items()}, labels.copy()


def predict_like(labels: np.ndarray, seed: int) -> np.ndarray:
    """正解マスクを少し変化させた予測マスクを返す。"""
    rng = np.random.default_rng(seed)
    result = labels.copy()
    ids = np.unique(result)
    ids = ids[ids > 0]
    if len(ids):
        for value in rng.choice(ids, size=max(1, len(ids) // 12), replace=False):
            result[result == value] = 0
    return result


def make_thumbnail(seed: int, size: tuple[int, int]) -> np.ndarray:
    """小さい解像度で合成画像とラベルのオーバーレイを作る。"""
    width, height = size
    rng = np.random.default_rng(seed)
    gray = np.clip(rng.normal(42, 7, (height, width)), 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    for index in range(1, 7):
        radius = int(
            rng.integers(max(3, min(width, height) // 22), max(5, min(width, height) // 10))
        )
        cy = int(rng.integers(radius, max(radius + 1, height - radius)))
        cx = int(rng.integers(radius, max(radius + 1, width - radius)))
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        color = np.array(
            (
                (index * 67 + 53) % 206 + 50,
                (index * 131 + 97) % 206 + 50,
                (index * 197 + 29) % 206 + 50,
            ),
            dtype=np.uint16,
        )
        region = rgb[y0:y1, x0:x1]
        region[disk] = ((region[disk].astype(np.uint16) * 3 + color) // 4).astype(np.uint8)
        rim = disk & ((yy - cy) ** 2 + (xx - cx) ** 2 <= (radius * 0.86) ** 2)
        region[rim] = color.astype(np.uint8)
    return np.ascontiguousarray(rgb)
