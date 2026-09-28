"""Cellpose 4.2 の SAM 系モデルを学習ループへ接続する。"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from foam_cell_analysis.training.adapters.base import Sample
from foam_cell_analysis.training.schedule import learning_rate

FLOW_CACHE_BYTES = 2 * 1024**3
LIMIT_PIXELS = 16_777_216
PATCH_SIZE = 256
PRETRAINED_MODELS = {"cpsam", "cpsam_v2"}
GEOMETRIC_TRANSFORMS = {
    "horizontal_flip",
    "vertical_flip",
    "rotation",
    "scale",
    "translation",
    "crop",
    "elastic",
}


def _flow_function():
    """Cellpose dynamics の関数を遅延 import して返す。"""
    from cellpose import dynamics

    return dynamics.labels_to_flows


def _segmentation_loss():
    """cellpose 4.2 固有の segmentation loss を返す。"""
    from cellpose import train

    return train._loss_fn_seg


def scaled_shape(
    height: int,
    width: int,
    scale_range: float,
    rng: np.random.Generator,
    *,
    limit_pixels: int = LIMIT_PIXELS,
) -> tuple[int, int]:
    """9.4 の乱数倍率と最大画素数に従う出力サイズを計算する。"""
    if height < 1 or width < 1 or not 0 <= scale_range <= 1:
        raise ValueError("画像サイズまたは scale_range が不正です")
    factor = 1.0 - scale_range / 2.0 + scale_range * float(rng.random())
    scaled_height = max(1, round(height * factor))
    scaled_width = max(1, round(width * factor))
    if scaled_height * scaled_width > limit_pixels:
        ratio = float(np.sqrt(limit_pixels / (height * width)))
        scaled_height = max(1, int(np.floor(height * ratio)))
        scaled_width = max(1, int(np.floor(width * ratio)))
    if scaled_height * scaled_width > limit_pixels:
        raise ValueError(
            f"画像が Cellpose の入力上限を超えています: {scaled_height}x{scaled_width}"
        )
    return scaled_height, scaled_width


def resize_image_and_labels(
    image: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """画像を双線形、ラベルを最近傍で同じサイズへ変更する。"""
    import cv2

    height, width = shape
    resized_image = cv2.resize(
        np.asarray(image, dtype=np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    resized_labels = cv2.resize(
        np.asarray(labels, dtype=np.int32), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized_image.astype(np.float32, copy=False), resized_labels.astype(np.int32, copy=False)


def _active_geometry(profile: dict[str, Any]) -> bool:
    settings = {str(item.get("key")): item for item in profile.get("transforms", [])}
    return any(
        key in settings
        and settings[key].get("enabled", True)
        and float(settings[key].get("probability", 1.0)) > 0
        for key in GEOMETRIC_TRANSFORMS
    )


def _pad_and_crop(
    image: np.ndarray,
    labels: np.ndarray,
    flows: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """画像・ラベル・flow を 256 パッチへゼロ詰めして切り出す。"""
    height, width = image.shape
    pad_height, pad_width = max(0, PATCH_SIZE - height), max(0, PATCH_SIZE - width)
    if pad_height or pad_width:
        image = np.pad(image, ((0, pad_height), (0, pad_width)), mode="constant")
        labels = np.pad(labels, ((0, pad_height), (0, pad_width)), mode="constant")
        flows = np.pad(flows, ((0, 0), (0, pad_height), (0, pad_width)), mode="constant")
    height, width = image.shape
    top = int(rng.integers(0, height - PATCH_SIZE + 1)) if height > PATCH_SIZE else 0
    left = int(rng.integers(0, width - PATCH_SIZE + 1)) if width > PATCH_SIZE else 0
    region = np.s_[top : top + PATCH_SIZE, left : left + PATCH_SIZE]
    return image[region], labels[region], flows[:, region[0], region[1]]


class CellposeAdapter:
    """SAM 系 Cellpose ネットワークの学習・推論・保存を提供する。"""

    def __init__(self, *, flow_cache_bytes: int = FLOW_CACHE_BYTES) -> None:
        self.model = None
        self.optimizer = None
        self.device = None
        self.model_config: dict[str, Any] = {}
        self._batch_size = 1
        self._flow_cache_limit = max(0, int(flow_cache_bytes))
        self._flow_cache_ceiling = self._flow_cache_limit
        self._flow_cache: OrderedDict[tuple[str, tuple[int, int]], np.ndarray] = OrderedDict()
        self._flow_cache_size = 0
        self._shared_array_cache = None

    def initial_weights_file(self, model_config: dict[str, Any]) -> Path:
        """固定された Cellpose 重みディレクトリ内の初期モデルを返す。"""
        name = model_config.get("pretrained_model")
        if name not in PRETRAINED_MODELS:
            raise ValueError("Cellpose の学習版は cpsam または cpsam_v2 です")
        root = Path(
            os.environ.get(
                "CELLPOSE_LOCAL_MODELS_PATH",
                Path(__file__).resolve().parents[4] / "workspace" / "pretrained" / "cellpose",
            )
        )
        return root / str(name)

    def build(self, model_config: dict[str, Any], device: Any) -> None:
        """事前学習済み CellposeModel を fp32 で構築し、SAM 系を検証する。"""
        import cellpose.models as cellpose_models
        import torch

        name = model_config.get("pretrained_model")
        if name not in PRETRAINED_MODELS:
            raise ValueError("Cellpose の学習版は cpsam または cpsam_v2 です")
        device = torch.device(device)
        local_models = self.initial_weights_file(model_config).parent
        cellpose_models.MODEL_DIR = local_models
        cellpose_models.MODEL_LIST_PATH = os.fspath(local_models / "gui_models.txt")
        model = cellpose_models.CellposeModel(
            pretrained_model=name, device=device, use_bfloat16=False
        )
        if model.net.backbone != "sam_vitl":
            raise ValueError(
                f"Cellpose のバックボーンが SAM 系ではありません: {model.net.backbone}"
            )
        model.net.to(device=device, dtype=torch.float32)
        self.model = model
        self.device = device
        self.model_config = dict(model_config)
        self._shared_array_cache = self.model_config.get("_shared_array_cache")
        dataset_cache_bytes = max(0, int(self.model_config.get("_dataset_cache_bytes", 0)))
        self._flow_cache_limit = min(
            self._flow_cache_ceiling, max(0, FLOW_CACHE_BYTES - dataset_cache_bytes)
        )
        self.optimizer = None
        self._flow_cache.clear()
        self._flow_cache_size = 0

    def make_optimizer(self, training: dict[str, Any]) -> Any:
        """Cellpose 4 と同じ AdamW を学習ネットへ作る。"""
        import torch

        if self.model is None:
            raise RuntimeError("Cellpose モデルが構築されていません")
        self._batch_size = int(training.get("batch_size", 1))
        if self._batch_size < 1:
            raise ValueError("Cellpose の batch_size は正の整数が必要です")
        self.optimizer = torch.optim.AdamW(
            self.model.net.parameters(),
            lr=float(training.get("learning_rate", 1e-5)),
            weight_decay=float(training.get("weight_decay", 0.1)),
        )
        return self.optimizer

    def lr(self, epoch: int, total_epochs: int, base_lr: float) -> float:
        return learning_rate("cellpose", epoch, total_epochs, base_lr)

    def training_exclusion_reason(
        self, mask: np.ndarray, model_config: dict[str, Any]
    ) -> str | None:
        instance_count = np.unique(np.asarray(mask)[np.asarray(mask) != 0]).size
        minimum = int(model_config.get("min_train_masks", 5))
        return f"min_train_masks:{minimum}" if instance_count < minimum else None

    def should_exclude_training_item(self, mask: np.ndarray, model_config: dict[str, Any]) -> bool:
        return self.training_exclusion_reason(mask, model_config) is not None

    def epoch_item_ids(
        self, item_ids: list[str], model_config: dict[str, Any], rng: np.random.Generator
    ) -> list[str]:
        """必要数が異なるときだけ fold seed 由来の乱数で復元抽出する。"""
        if not item_ids:
            raise ValueError("Cellpose の学習画像がありません")
        requested = model_config.get("nimg_per_epoch")
        sample_count = len(item_ids) if requested is None else int(requested)
        if sample_count < 1:
            raise ValueError("nimg_per_epoch は正の整数または null が必要です")
        if sample_count == len(item_ids):
            return list(item_ids)
        indices = rng.integers(0, len(item_ids), size=sample_count)
        return [item_ids[int(index)] for index in indices]

    def _flows(self, item_id: str, labels: np.ndarray) -> np.ndarray:
        key = (item_id, labels.shape)
        shared_key = ("flow", key)
        if self._shared_array_cache is not None:
            cached = self._shared_array_cache.get(shared_key)
            if cached is not None:
                return cached
        cached = self._flow_cache.get(key)
        if cached is not None:
            self._flow_cache.move_to_end(key)
            return cached
        flow_rows = _flow_function()(
            [np.asarray(labels, dtype=np.int32).copy()], files=None, device=self.device
        )
        flow = np.asarray(flow_rows[0])
        if flow.ndim != 3 or flow.shape[0] != 4 or flow.shape[1:] != labels.shape:
            raise ValueError("Cellpose labels_to_flows は (4,H,W) を返す必要があります")
        targets = np.ascontiguousarray(flow[1:], dtype=np.float32)
        if self._shared_array_cache is not None:
            self._shared_array_cache.put(shared_key, targets)
        elif targets.nbytes <= self._flow_cache_limit:
            while (
                self._flow_cache and self._flow_cache_size + targets.nbytes > self._flow_cache_limit
            ):
                _, evicted = self._flow_cache.popitem(last=False)
                self._flow_cache_size -= evicted.nbytes
            self._flow_cache[key] = targets
            self._flow_cache_size += targets.nbytes
        return targets

    def prepare_sample(self, sample: Sample, *, training: bool, rng: Any) -> Sample:
        """学習時は拡大縮小・flow・切り出しをし、推論時は 3 チャンネル化だけ行う。"""
        image = np.asarray(sample.image, dtype=np.float32)
        labels = np.asarray(sample.mask)
        if image.ndim != 2 or labels.ndim != 2 or image.shape != labels.shape:
            raise ValueError("Cellpose の画像とラベルは同じ 2 次元サイズが必要です")
        if not training:
            channels = np.stack((image, np.zeros_like(image), np.zeros_like(image)), axis=-1)
            return Sample(sample.item_id, channels.astype(np.float32, copy=False), labels.copy())

        scale_range = float(self.model_config.get("scale_range", 0.5))
        height, width = scaled_shape(*image.shape, scale_range, rng)
        image, labels = resize_image_and_labels(image, labels, (height, width))
        profile = self.model_config.get("augmentation_profile", {})
        cacheable = scale_range == 0 and not _active_geometry(profile)
        if cacheable:
            flows = self._flows(sample.item_id, labels)
        else:
            flow_rows = _flow_function()(
                [labels.astype(np.int32, copy=True)], files=None, device=self.device
            )
            all_channels = np.asarray(flow_rows[0])
            if all_channels.shape != (4, height, width):
                raise ValueError("Cellpose labels_to_flows は (4,H,W) を返す必要があります")
            flows = np.ascontiguousarray(all_channels[1:], dtype=np.float32)
        image, labels, flows = _pad_and_crop(image, labels, flows, rng)
        channels = np.stack((image, np.zeros_like(image), np.zeros_like(image)), axis=-1)
        return Sample(
            sample.item_id,
            channels.astype(np.float32, copy=False),
            np.ascontiguousarray(flows, dtype=np.float32),
        )

    def train_one_epoch(self, samples: Iterable[Sample], lr: float) -> float:
        """指定 batch ごとに forward/backward し、端数 batch も処理する。"""
        import torch

        if self.model is None or self.optimizer is None:
            raise RuntimeError("Cellpose モデルと optimizer が必要です")
        self.model.net.train()
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)
        weighted_loss = 0.0
        sample_count = 0
        batch_images: list[Any] = []
        batch_targets: list[Any] = []
        loss_function = _segmentation_loss()

        def step_batch() -> None:
            nonlocal weighted_loss, sample_count, batch_images, batch_targets
            if not batch_images:
                return
            images = torch.stack(batch_images)
            targets = torch.stack(batch_targets)
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model.net(images)[0]
            loss = loss_function(targets, prediction, self.device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Cellpose loss が有限値ではありません")
            loss.backward()
            self.optimizer.step()
            count = len(batch_images)
            weighted_loss += float(loss.detach().cpu()) * count
            sample_count += count
            batch_images = []
            batch_targets = []

        for sample in samples:
            batch_images.append(
                torch.from_numpy(np.asarray(sample.image).copy())
                .permute(2, 0, 1)
                .to(self.device, dtype=torch.float32)
            )
            batch_targets.append(
                torch.from_numpy(np.asarray(sample.mask).copy()).to(
                    self.device, dtype=torch.float32
                )
            )
            if len(batch_images) == self._batch_size:
                step_batch()
        step_batch()
        return weighted_loss / sample_count if sample_count else 0.0

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """モデル自身の eval を使い、終了後に同じ net を train 状態へ戻す。"""
        if self.model is None:
            raise RuntimeError("Cellpose モデルが構築されていません")
        self.model.net.eval()
        try:
            result = self.model.eval(
                images,
                channel_axis=2,
                normalize=False,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
                min_size=15,
                max_size_fraction=0.4,
                bsize=PATCH_SIZE,
            )
            masks = result[0]
            return [np.asarray(mask) for mask in masks]
        finally:
            self.model.net.train()

    def state_dict(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Cellpose モデルが構築されていません")
        return self.model.net.state_dict()

    def weights_nbytes(self) -> int:
        """fp32 state_dict の保存サイズを見積もる。"""
        if self.model is None:
            raise RuntimeError("Cellpose モデルが構築されていません")
        return sum(value.numel() * value.element_size() for value in self.state_dict().values())


def build_for_inference(weights_path: str | Path, device: Any = "cpu") -> CellposeAdapter:
    """事前学習済みファイルに触れず、保存済み Cellpose state_dict を開く。"""
    import torch
    from cellpose.models import CellposeModel

    device = torch.device(device)
    model = CellposeModel(
        pretrained_model=os.fspath(weights_path), device=device, use_bfloat16=False
    )
    adapter = CellposeAdapter()
    adapter.model = model
    adapter.device = device
    return adapter
