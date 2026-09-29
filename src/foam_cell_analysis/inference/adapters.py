"""評価プロセスが使う推論専用アダプタ（比較・推論設計 8 章）。

torch / torchvision / cellpose は必要になった時点で import する。
このモジュール自体は torch なしで import できる。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from foam_cell_analysis.training.preprocessing import prepare_model_input


class InferenceAdapter(Protocol):
    """前処理済みの画像 1 枚からラベル画像を返す契約。"""

    def predict(self, image: np.ndarray) -> np.ndarray: ...


def select_device() -> Any:
    """学習と同じ規則で、CUDA が使えれば CUDA、なければ CPU を返す。"""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _SingleImageAdapter:
    """バッチ入力の学習用アダプタを 1 枚ずつの推論契約へ包む。"""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def predict(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(self.adapter.predict([image])[0])


class NumpyFakeInferenceAdapter:
    """torch なしで試験するための決定的な推論器。

    画像が閾値以上の画素を 4 近傍連結成分に分けてラベル化する。
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def predict(self, image: np.ndarray) -> np.ndarray:
        from foam_cell_analysis.training.adapters.fake import FakeAdapter

        values = np.asarray(image, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("fake_numpy の入力は 2 次元画像です")
        return FakeAdapter._components(values >= self.threshold)


def build_inference_adapter(
    *,
    model_type: str,
    model_config: dict[str, Any],
    weights_path: str | Path,
    effective_params: dict[str, Any],
    device: Any,
) -> InferenceAdapter:
    """model_type に応じた推論用アダプタを作る。事前学習済み重みは読まない。"""
    params = dict(effective_params or {})
    if model_type == "mask_rcnn":
        from foam_cell_analysis.training.adapters.mask_rcnn import build_for_inference

        config = dict(model_config)
        config["eval_params"] = dict(config.get("eval_params") or {}) | params
        return _SingleImageAdapter(build_for_inference(config, weights_path, device))
    if model_type == "cellpose":
        from foam_cell_analysis.training.adapters.cellpose import build_for_inference

        return _SingleImageAdapter(build_for_inference(weights_path, device, eval_params=params))
    if model_type == "fake":
        from foam_cell_analysis.training.adapters.fake import build_for_inference

        return _SingleImageAdapter(build_for_inference(model_config, weights_path, device, params))
    if model_type == "fake_numpy":
        return NumpyFakeInferenceAdapter(params.get("threshold", 0.5))
    raise ValueError(f"未対応の model_type です: {model_type}")


def run_inference(
    adapter: InferenceAdapter,
    image_raw: np.ndarray,
    normalization: dict[str, Any] | None,
    model_type: str,
) -> np.ndarray:
    """生画像を正規化し、モデル入力へ変換して推論する。"""
    return adapter.predict(prepare_model_input(image_raw, normalization, model_type))
