"""CPU で短時間に end-to-end を試せる小型 torch アダプタ。"""

from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from foam_cell_analysis.training.adapters.base import Sample
from foam_cell_analysis.training.schedule import learning_rate


class FakeAdapter:
    """前景二値分類と 4 近傍連結成分によるラベル予測。"""

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        self.device = None

    def build(self, model_config: dict[str, Any], device: Any) -> None:
        from torch import nn

        del model_config
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=1),
        ).to(device)
        self.optimizer = None

    def initial_weights_file(self, model_config: dict[str, Any]) -> Path | None:
        del model_config
        return None

    def make_optimizer(self, training: dict[str, Any]) -> Any:
        import torch

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        return self.optimizer

    def lr(self, epoch: int, total_epochs: int, base_lr: float) -> float:
        return learning_rate("fake", epoch, total_epochs, base_lr)

    def should_exclude_training_item(self, mask: np.ndarray, model_config: dict[str, Any]) -> bool:
        del mask, model_config
        return False

    def training_exclusion_reason(
        self, mask: np.ndarray, model_config: dict[str, Any]
    ) -> str | None:
        del mask, model_config
        return None

    def epoch_item_ids(
        self, item_ids: list[str], model_config: dict[str, Any], rng: Any
    ) -> list[str]:
        del model_config, rng
        return list(item_ids)

    def prepare_sample(self, sample: Sample, *, training: bool, rng: Any) -> Sample:
        del training, rng
        return sample

    def train_one_epoch(self, samples: Iterable[Sample], lr: float) -> float:
        import torch
        from torch.nn import functional

        if self.optimizer is None:
            raise RuntimeError("optimizer が作られていません")
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train()
        losses = []
        for sample in samples:
            image = torch.from_numpy(sample.image.astype(np.float32))[None, None].to(self.device)
            target = torch.from_numpy((sample.mask > 0).astype(np.float32))[None, None].to(
                self.device
            )
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(image)
            loss = functional.binary_cross_entropy_with_logits(prediction, target)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    @staticmethod
    def _components(binary: np.ndarray) -> np.ndarray:
        height, width = binary.shape
        labels = np.zeros((height, width), dtype=np.uint32)
        next_label = 0
        for y, x in np.argwhere(binary):
            if labels[y, x]:
                continue
            next_label += 1
            labels[y, x] = next_label
            pending = [(int(y), int(x))]
            while pending:
                row, column = pending.pop()
                for neighbor_y, neighbor_x in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if (
                        0 <= neighbor_y < height
                        and 0 <= neighbor_x < width
                        and binary[neighbor_y, neighbor_x]
                        and labels[neighbor_y, neighbor_x] == 0
                    ):
                        labels[neighbor_y, neighbor_x] = next_label
                        pending.append((neighbor_y, neighbor_x))
        return labels

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        import torch

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for image in images:
                tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None].to(
                    self.device
                )
                foreground = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy() >= 0.5
                predictions.append(self._components(foreground))
        return predictions

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def weights_nbytes(self) -> int:
        import torch

        buffer = BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.tell()
