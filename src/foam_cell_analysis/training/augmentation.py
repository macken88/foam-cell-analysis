"""transforms.v2 の範囲・順序を使う 2D 画像／ラベル拡張。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _setting_map(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["key"]): dict(item) for item in profile.get("transforms", [])}


def _warp(
    image: np.ndarray,
    mask: np.ndarray,
    theta: np.ndarray,
    *,
    displacement: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    height, width = image.shape
    image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None]
    label_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float64))[None, None]
    transform = torch.as_tensor(theta, dtype=torch.float32)[None]
    grid = functional.affine_grid(transform, image_tensor.shape, align_corners=False)
    if displacement is not None:
        grid = grid + torch.from_numpy(displacement.astype(np.float32))[None]
    warped_image = functional.grid_sample(
        image_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, 0]
    warped_mask = functional.grid_sample(
        label_tensor,
        grid.to(dtype=label_tensor.dtype),
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]
    return warped_image.numpy().reshape(height, width), warped_mask.numpy().reshape(height, width)


def _clip_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def apply_profile(
    image: np.ndarray,
    mask: np.ndarray,
    profile: Mapping[str, Any],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """確率・順序に従って拡張し、ラベルの整数値を保つ。"""
    random = rng or np.random.default_rng()
    output = np.asarray(image, dtype=np.float32).copy()
    original_mask = np.asarray(mask)
    if output.ndim != 2 or original_mask.ndim != 2 or output.shape != original_mask.shape:
        raise ValueError("画像とラベルは同じ 2 次元サイズである必要があります")
    labels = original_mask.copy()
    settings = _setting_map(profile)
    order = profile.get("pipeline_order", profile.get("order", list(settings)))
    for key in order:
        setting = settings.get(str(key))
        if not setting or not setting.get("enabled", True):
            continue
        probability = float(setting.get("probability", 1.0))
        if random.random() >= probability:
            continue
        low = setting.get("range_min")
        high = setting.get("range_max")
        value = (
            float(random.uniform(float(low), float(high)))
            if low is not None and high is not None
            else None
        )
        height, width = output.shape
        theta = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        if key in {"horizontal_flip", "vertical_flip"}:
            if key == "horizontal_flip":
                output, labels = output[:, ::-1].copy(), labels[:, ::-1].copy()
            else:
                output, labels = output[::-1, :].copy(), labels[::-1, :].copy()
        elif key == "rotation":
            angle = np.deg2rad(float(value))
            theta[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            output, labels = _warp(output, labels, theta)
        elif key == "scale":
            theta[0, 0] = theta[1, 1] = 1.0 / float(value)
            output, labels = _warp(output, labels, theta)
        elif key == "translation":
            theta[0, 2] = -float(random.uniform(float(low), float(high))) * 2
            theta[1, 2] = -float(random.uniform(float(low), float(high))) * 2
            output, labels = _warp(output, labels, theta)
        elif key == "crop":
            fraction = float(value)
            crop_h, crop_w = max(1, round(height * fraction)), max(1, round(width * fraction))
            top = int(random.integers(0, height - crop_h + 1))
            left = int(random.integers(0, width - crop_w + 1))
            import torch
            import torch.nn.functional as functional

            image_tensor = torch.from_numpy(output[top : top + crop_h, left : left + crop_w])[
                None, None
            ]
            label_tensor = torch.from_numpy(
                labels[top : top + crop_h, left : left + crop_w].astype(np.float64)
            )[None, None]
            output = functional.interpolate(
                image_tensor, size=(height, width), mode="bilinear", align_corners=False
            )[0, 0].numpy()
            labels = functional.interpolate(label_tensor, size=(height, width), mode="nearest")[
                0, 0
            ].numpy()
        elif key == "elastic":
            import torch
            from torchvision.transforms import v2
            from torchvision.transforms.v2 import functional as functional
            from torchvision.tv_tensors import Image as TVImage
            from torchvision.tv_tensors import Mask as TVMask

            transform = v2.ElasticTransform(alpha=50.0 * float(value), sigma=5)
            generator = torch.Generator()
            generator.manual_seed(int(random.integers(0, 2**63 - 1)))
            dx = torch.rand((1, 1, height, width), generator=generator) * 2 - 1
            dy = torch.rand((1, 1, height, width), generator=generator) * 2 - 1
            if transform.sigma[0] > 0:
                kernel_size = int(8 * transform.sigma[0] + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                dx = functional.gaussian_blur(
                    dx, [kernel_size, kernel_size], sigma=list(transform.sigma)
                )
            if transform.sigma[1] > 0:
                kernel_size = int(8 * transform.sigma[1] + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                dy = functional.gaussian_blur(
                    dy, [kernel_size, kernel_size], sigma=list(transform.sigma)
                )
            dx = dx * transform.alpha[0] / width
            dy = dy * transform.alpha[1] / height
            displacement = torch.cat([dx, dy], dim=1).permute(0, 2, 3, 1)

            instance_ids = np.unique(labels)
            instance_ids = instance_ids[instance_ids != 0]
            if len(instance_ids) >= 2**24:
                raise ValueError("elastic 拡張できるインスタンス数の上限を超えています")
            compact = np.zeros(labels.shape, dtype=np.int32)
            foreground = labels != 0
            if foreground.any():
                compact[foreground] = (
                    np.searchsorted(instance_ids, labels[foreground]).astype(np.int32) + 1
                )
            image_tensor = TVImage(torch.from_numpy(output.copy())[None])
            label_tensor = TVMask(torch.from_numpy(compact)[None])
            params = {"displacement": displacement}
            transformed_image = transform.transform(image_tensor, params)
            transformed_labels = transform.transform(label_tensor, params)
            output = transformed_image[0].numpy()
            compact_transformed = np.rint(transformed_labels[0].numpy()).astype(np.int64)
            labels = np.zeros(original_mask.shape, dtype=original_mask.dtype)
            visible = compact_transformed > 0
            if visible.any():
                labels[visible] = instance_ids[compact_transformed[visible] - 1]
        elif key == "brightness":
            output *= 1.0 + float(value) / 100.0
        elif key == "contrast":
            mean = float(output.mean())
            output = (output - mean) * float(value) + mean
        elif key == "gamma":
            output = np.power(np.clip(output, 0.0, 1.0), float(value))
        elif key == "blur":
            sigma = float(value)
            if sigma >= 0.1:
                radius = int(np.ceil(3 * sigma))
                coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
                kernel = np.exp(-(coordinates**2) / (2 * sigma**2))
                kernel /= kernel.sum()
                import torch
                import torch.nn.functional as functional

                tensor = torch.from_numpy(output)[None, None]
                kernel_x = torch.from_numpy(kernel)[None, None, None, :]
                kernel_y = torch.from_numpy(kernel)[None, None, :, None]
                tensor = functional.conv2d(tensor, kernel_x, padding=(0, radius))
                output = functional.conv2d(tensor, kernel_y, padding=(radius, 0))[0, 0].numpy()
        elif key == "noise":
            output += random.normal(0.0, float(value) / 255.0, size=output.shape).astype(np.float32)
        elif key in {"channel_dropout", "channel_intensity"}:
            pass
        else:
            raise ValueError(f"未対応の拡張です: {key}")
        output = _clip_image(output)
        labels = labels.astype(original_mask.dtype, copy=False)
    return output, labels
