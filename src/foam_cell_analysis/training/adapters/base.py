"""学習ループからモデル実装を分離するアダプタ契約。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class Sample:
    """モデルへ渡す前の単一画像と整数ラベル。"""

    item_id: str
    image: np.ndarray
    mask: np.ndarray


class ModelAdapter(Protocol):
    """モデル構築・学習・推論を提供する契約。"""

    def build(self, model_config: dict[str, Any], device: Any) -> None: ...

    def initial_weights_file(self, model_config: dict[str, Any]) -> Path | None: ...

    def make_optimizer(self, training: dict[str, Any]) -> Any: ...

    def lr(self, epoch: int, total_epochs: int, base_lr: float) -> float: ...

    def should_exclude_training_item(self, mask: np.ndarray, model_config: dict[str, Any]) -> bool:
        """モデル固有の条件で学習画像を除外する。"""
        ...

    def training_exclusion_reason(
        self, mask: np.ndarray, model_config: dict[str, Any]
    ) -> str | None: ...

    def epoch_item_ids(
        self, item_ids: list[str], model_config: dict[str, Any], rng: Any
    ) -> list[str]:
        """1 エポックで使う画像 ID を順序込みで返す。"""
        ...

    def prepare_sample(self, sample: Sample, *, training: bool, rng: Any) -> Sample:
        """必要ならモデル固有の入力変換を加える。"""
        ...

    def train_one_epoch(self, samples: Iterable[Sample], lr: float) -> float: ...

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def weights_nbytes(self) -> int: ...
