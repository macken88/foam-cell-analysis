"""推論用アダプタ・共通入力変換・デバイス選択の試験。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from foam_cell_analysis.inference import adapters
from foam_cell_analysis.inference.adapters import (
    NumpyFakeInferenceAdapter,
    build_inference_adapter,
    run_inference,
)
from foam_cell_analysis.training.preprocessing import (
    normalize_image,
    prepare_model_input,
    to_model_channels,
)


def _image():
    return (np.arange(48, dtype=np.float32).reshape(6, 8) * 10) + 100


def test_module_imports_without_loading_torch():
    code = (
        "import sys; import foam_cell_analysis.inference.adapters; "
        "raise SystemExit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_prepare_model_input_matches_previous_validation_conversion():
    image = _image()
    normalization = {"low_percentile": 2.0, "high_percentile": 98.0}
    normalized = normalize_image(image, normalization)

    mask_rcnn = prepare_model_input(image, normalization, "mask_rcnn")
    assert mask_rcnn.shape == (6, 8, 3) and mask_rcnn.dtype == np.float32
    assert all(np.array_equal(mask_rcnn[..., c], normalized) for c in range(3))

    cellpose = prepare_model_input(image, normalization, "cellpose")
    assert cellpose.shape == (6, 8, 3) and cellpose.dtype == np.float32
    assert np.array_equal(cellpose[..., 0], normalized)
    assert not cellpose[..., 1:].any()

    assert np.array_equal(prepare_model_input(image, normalization, "fake"), normalized)
    with pytest.raises(ValueError):
        to_model_channels(normalized, "unknown")


def test_training_validation_path_uses_same_conversion():
    pytest.importorskip("torch")
    from foam_cell_analysis.training.adapters.base import Sample
    from foam_cell_analysis.training.adapters.cellpose import CellposeAdapter
    from foam_cell_analysis.training.adapters.mask_rcnn import MaskRCNNAdapter

    normalized = normalize_image(_image())
    labels = np.zeros_like(normalized, dtype=np.uint16)
    sample = Sample("a", normalized, labels)
    for model_type, adapter in (("cellpose", CellposeAdapter()), ("mask_rcnn", MaskRCNNAdapter())):
        prepared = adapter.prepare_sample(sample, training=False, rng=None)
        assert np.array_equal(prepared.image, to_model_channels(normalized, model_type))


def _numpy_adapter(threshold):
    return build_inference_adapter(
        model_type="fake_numpy",
        model_config={},
        weights_path="unused",
        effective_params={"threshold": threshold},
        device=None,
    )


def test_numpy_fake_adapter_is_deterministic_and_threshold_driven():
    image = np.zeros((6, 8), dtype=np.float32)
    image[1:3, 1:3] = 0.9
    image[4:6, 5:8] = 0.6

    low = _numpy_adapter(0.5).predict(image)
    assert np.array_equal(low, _numpy_adapter(0.5).predict(image))
    assert low.max() == 2
    assert _numpy_adapter(0.8).predict(image).max() == 1
    assert isinstance(_numpy_adapter(0.5), NumpyFakeInferenceAdapter)


def test_run_inference_normalizes_then_predicts():
    raw = np.zeros((6, 8), dtype=np.float32)
    raw[1:3, 1:3] = 1000.0
    raw[4:6, 5:8] = 1000.0
    labels = run_inference(NumpyFakeInferenceAdapter(0.5), raw, None, "fake_numpy")
    assert labels.shape == raw.shape and labels.max() == 2


def test_build_inference_adapter_dispatch_and_params(monkeypatch):
    pytest.importorskip("torch")
    from foam_cell_analysis.training.adapters import cellpose as cp
    from foam_cell_analysis.training.adapters import mask_rcnn as mr

    seen = {}

    class Stub:
        def predict(self, images):
            return [np.ones(images[0].shape[:2], dtype=np.uint16)]

    def fake_mask(config, weights, device):
        seen["mask"] = (config, weights, device)
        return Stub()

    def fake_cellpose(weights, device, eval_params=None):
        seen["cellpose"] = (weights, device, eval_params)
        return Stub()

    monkeypatch.setattr(mr, "build_for_inference", fake_mask)
    monkeypatch.setattr(cp, "build_for_inference", fake_cellpose)

    mask = build_inference_adapter(
        model_type="mask_rcnn",
        model_config={"eval_params": {"mask_thresh": 0.1}},
        weights_path="w.pt",
        effective_params={"box_score_thresh": 0.3},
        device="cpu",
    )
    assert seen["mask"][0]["eval_params"] == {"mask_thresh": 0.1, "box_score_thresh": 0.3}
    assert mask.predict(np.zeros((4, 4, 3), np.float32)).shape == (4, 4)

    build_inference_adapter(
        model_type="cellpose",
        model_config={},
        weights_path="w.pt",
        effective_params={"flow_threshold": 0.6},
        device="cpu",
    )
    assert seen["cellpose"] == ("w.pt", "cpu", {"flow_threshold": 0.6})

    with pytest.raises(ValueError):
        build_inference_adapter(
            model_type="nope", model_config={}, weights_path="w", effective_params={}, device=None
        )


def test_select_device_prefers_cuda_only_when_available(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert adapters.select_device().type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert adapters.select_device().type == "cuda"


def test_fake_torch_adapter_save_load_predict(tmp_path):
    torch = pytest.importorskip("torch")
    from foam_cell_analysis.training.adapters.fake import FakeAdapter

    source = FakeAdapter()
    source.build({}, "cpu")
    weights = tmp_path / "w.pt"
    torch.save(source.state_dict(), weights)

    adapter = build_inference_adapter(
        model_type="fake",
        model_config={},
        weights_path=weights,
        effective_params={"threshold": 0.5},
        device="cpu",
    )
    image = np.random.default_rng(0).random((6, 8)).astype(np.float32)
    assert np.array_equal(adapter.predict(image), source.predict([image])[0])


@pytest.mark.ml
@pytest.mark.slow
def test_real_mask_rcnn_save_load_predict(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from foam_cell_analysis.training.adapters.mask_rcnn import MaskRCNNAdapter

    config = {
        "type": "mask_rcnn",
        "backbone": "resnet50_fpn_v2",
        "pretrained_weights": "none",
        "trainable_backbone_layers": 0,
        "input": {"min_size": 64, "max_size": 64},
    }
    source = MaskRCNNAdapter()
    source.build(config, "cpu")
    weights = tmp_path / "w.pt"
    torch.save(source.state_dict(), weights)
    adapter = build_inference_adapter(
        model_type="mask_rcnn",
        model_config=config,
        weights_path=weights,
        effective_params={"box_score_thresh": 0.05},
        device="cpu",
    )
    labels = run_inference(adapter, np.random.default_rng(0).random((64, 64)), None, "mask_rcnn")
    assert labels.shape == (64, 64)


@pytest.mark.ml
@pytest.mark.slow
def test_real_cellpose_save_load_predict(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cellpose")
    from foam_cell_analysis.training.adapters.cellpose import CellposeAdapter

    pretrained = Path(__file__).resolve().parents[2] / "workspace" / "pretrained" / "cellpose"
    if not (pretrained / "cpsam").is_file():
        pytest.skip("cpsam の重みがありません")
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", os.fspath(pretrained))
    source = CellposeAdapter()
    source.build({"pretrained_model": "cpsam"}, torch.device("cpu"))
    weights = tmp_path / "w.pt"
    torch.save(source.state_dict(), weights)
    adapter = build_inference_adapter(
        model_type="cellpose",
        model_config={},
        weights_path=weights,
        effective_params={"flow_threshold": 0.4},
        device=torch.device("cpu"),
    )
    labels = run_inference(adapter, np.random.default_rng(0).random((256, 256)), None, "cellpose")
    assert labels.shape == (256, 256)
