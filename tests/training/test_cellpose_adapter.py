from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foam_cell_analysis.training.adapters import cellpose as cellpose_module
from foam_cell_analysis.training.adapters.base import Sample
from foam_cell_analysis.training.adapters.cellpose import (
    CellposeAdapter,
    _pad_and_crop,
    scaled_shape,
)

torch = pytest.importorskip("torch")


def _sample(shape=(12, 16)):
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.prod(shape)
    labels = np.zeros(shape, dtype=np.uint16)
    labels[2:5, 3:7] = 4
    return Sample("item_a", image, labels)


def _fake_flows(labels, *, files, device):
    del files, device
    mask = labels[0]
    return [
        np.stack(
            (
                mask,
                (mask != 0).astype(np.float32),
                np.ones(mask.shape, dtype=np.float32),
                np.full(mask.shape, 2, dtype=np.float32),
            )
        )
    ]


def test_scaled_shape_uses_cellpose_range_and_applies_area_cap():
    rng = np.random.default_rng(9)
    assert scaled_shape(512, 512, 0, rng) == (512, 512)
    height, width = scaled_shape(8192, 4096, 0, np.random.default_rng(1))
    assert height * width <= cellpose_module.LIMIT_PIXELS
    assert height < 8192 and width < 4096
    assert scaled_shape(512, 512, 1, np.random.default_rng(4)) != (512, 512)


def test_scaled_shape_rejects_extreme_aspect_ratio_after_limit_shrink():
    with pytest.raises(ValueError, match="入力上限"):
        scaled_shape(50_000_000, 1, 0, np.random.default_rng(1))


def test_pad_and_crop_keeps_all_targets_aligned_and_zero_padded():
    image = np.ones((64, 128), dtype=np.float32)
    labels = np.ones((64, 128), dtype=np.int32)
    flows = np.ones((3, 64, 128), dtype=np.float32)
    cropped_image, cropped_labels, cropped_flows = _pad_and_crop(
        image, labels, flows, np.random.default_rng(1)
    )

    assert cropped_image.shape == (256, 256)
    assert cropped_labels.shape == (256, 256)
    assert cropped_flows.shape == (3, 256, 256)
    assert np.all(cropped_image[:, 128:] == 0)
    assert np.all(cropped_labels[:, 128:] == 0)
    assert np.all(cropped_flows[:, :, 128:] == 0)
    assert np.all(cropped_image[:64, :128] == 1)


def test_flow_contract_receives_copy_and_uses_only_last_three_planes(monkeypatch):
    original = _sample()
    original_mask = original.mask.copy()
    calls = []

    def flow_function(masks, *, files, device):
        calls.append((masks, files, device))
        return _fake_flows(masks, files=files, device=device)

    monkeypatch.setattr(cellpose_module, "_flow_function", lambda: flow_function)
    adapter = CellposeAdapter()
    adapter.device = "cpu"
    adapter.model_config = {"scale_range": 0, "augmentation_profile": {"transforms": []}}

    prepared = adapter.prepare_sample(original, training=True, rng=np.random.default_rng(2))

    assert calls[0][1:] == (None, "cpu")
    assert calls[0][0][0].dtype == np.int32
    assert not np.shares_memory(calls[0][0][0], original.mask)
    assert np.array_equal(original.mask, original_mask)
    assert prepared.image.shape == (256, 256, 3)
    assert prepared.mask.shape == (3, 256, 256)
    assert np.array_equal(
        prepared.mask[:, :12, :16], _fake_flows([original_mask], files=None, device="cpu")[0][1:]
    )
    assert np.all(prepared.image[:, :, 1:] == 0)


def test_flow_cache_is_used_only_without_geometric_augmentation(monkeypatch):
    calls = 0

    def flow_function(labels, *, files, device):
        nonlocal calls
        calls += 1
        return _fake_flows(labels, files=files, device=device)

    monkeypatch.setattr(cellpose_module, "_flow_function", lambda: flow_function)
    adapter = CellposeAdapter()
    adapter.device = "cpu"
    adapter.model_config = {"scale_range": 0, "augmentation_profile": {"transforms": []}}
    for _ in range(2):
        adapter.prepare_sample(_sample(), training=True, rng=np.random.default_rng(3))
    assert calls == 1

    adapter.model_config["augmentation_profile"] = {
        "transforms": [{"key": "horizontal_flip", "enabled": True, "probability": 1.0}]
    }
    adapter.prepare_sample(_sample(), training=True, rng=np.random.default_rng(3))
    assert calls == 2


def test_epoch_sampling_repeats_and_is_seed_reproducible():
    adapter = CellposeAdapter()
    config = {"nimg_per_epoch": 7}
    first = adapter.epoch_item_ids(["a", "b", "c"], config, np.random.default_rng(42))
    second = adapter.epoch_item_ids(["a", "b", "c"], config, np.random.default_rng(42))
    assert first == second
    assert len(first) == 7
    assert len(set(first)) < len(first)
    assert adapter.epoch_item_ids(
        ["a", "b"], {"nimg_per_epoch": None}, np.random.default_rng(2)
    ) == [
        "a",
        "b",
    ]


def test_min_train_masks_excludes_based_on_original_instances():
    adapter = CellposeAdapter()
    mask = np.zeros((10, 10), dtype=np.uint16)
    mask[1:3, 1:3] = 2
    mask[5:7, 5:7] = 8
    assert adapter.training_exclusion_reason(mask, {"min_train_masks": 3}) == "min_train_masks:3"
    assert adapter.training_exclusion_reason(mask, {"min_train_masks": 2}) is None


def test_validation_input_channel_order_and_predict_eval_contract():
    class Net:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    class Model:
        def __init__(self):
            self.net = Net()
            self.call = None

        def eval(self, images, **kwargs):
            self.call = (images, kwargs)
            return [np.zeros(image.shape[:2], dtype=np.uint16) for image in images], None, None

    adapter = CellposeAdapter()
    adapter.model = Model()
    sample = _sample()
    prepared = adapter.prepare_sample(sample, training=False, rng=None)
    assert prepared.image.shape == (*sample.image.shape, 3)
    assert np.allclose(prepared.image[..., 0], sample.image)
    assert np.all(prepared.image[..., 1:] == 0)

    predictions = adapter.predict([prepared.image])

    passed_images, kwargs = adapter.model.call
    assert passed_images[0].shape == (*sample.image.shape, 3)
    assert kwargs == {
        "channel_axis": 2,
        "normalize": False,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "min_size": 15,
        "max_size_fraction": 0.4,
        "bsize": 256,
    }
    assert predictions[0].shape == sample.image.shape
    assert adapter.model.net.training is True


def test_cellpose_lr_schedule_matches_4_2_warmup_and_decay():
    adapter = CellposeAdapter()
    assert adapter.lr(1, 40, 1e-5) == 0
    assert adapter.lr(10, 40, 1e-5) == pytest.approx(1e-5)
    assert adapter.lr(40, 40, 1e-5) == pytest.approx(1e-5)
    assert adapter.lr(51, 100, 1e-5) == pytest.approx(5e-6)
    assert adapter.lr(301, 400, 1e-5) == pytest.approx(5e-6)


def test_build_for_inference_reads_only_checkpoint_with_empty_local_model_path(
    tmp_path, monkeypatch
):
    models = importlib.import_module("cellpose.models")
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"weights")
    empty_models = tmp_path / "empty-cellpose-models"
    empty_models.mkdir()
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(empty_models))
    calls = []

    class Net:
        backbone = "sam_vitl"

    class FakeCellposeModel:
        def __init__(self, *, pretrained_model, device, use_bfloat16):
            calls.append((pretrained_model, device, use_bfloat16))
            self.net = Net()

    monkeypatch.setattr(models, "CellposeModel", FakeCellposeModel)

    restored = cellpose_module.build_for_inference(checkpoint, "cpu")

    assert calls == [(str(checkpoint), torch.device("cpu"), False)]
    assert restored.model.net.backbone == "sam_vitl"
    assert list(empty_models.iterdir()) == []


def test_initial_weights_file_uses_fixed_local_models_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(tmp_path))
    adapter = CellposeAdapter()
    assert adapter.initial_weights_file({"pretrained_model": "cpsam"}) == tmp_path / "cpsam"
    with pytest.raises(ValueError):
        adapter.initial_weights_file({"pretrained_model": "cyto3"})


def test_cellpose_adapter_runs_fp32_training_step(monkeypatch):
    class TinyNet(torch.nn.Module):
        backbone = "sam_vitl"

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Conv2d(3, 3, kernel_size=1)

        def forward(self, images):
            return (self.layer(images),)

    net = TinyNet()
    adapter = CellposeAdapter()
    adapter.model = SimpleNamespace(net=net)
    adapter.device = torch.device("cpu")
    adapter.optimizer = torch.optim.AdamW(net.parameters(), lr=0.1)
    monkeypatch.setattr(
        cellpose_module,
        "_segmentation_loss",
        lambda: lambda target, output, device: (output - target.to(device)).square().mean(),
    )
    before = net.layer.weight.detach().clone()
    sample = Sample(
        "item",
        np.ones((8, 8, 3), dtype=np.float32),
        np.zeros((3, 8, 8), dtype=np.float32),
    )

    loss = adapter.train_one_epoch([sample], 0.02)

    assert np.isfinite(loss)
    assert adapter.optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
    assert not torch.equal(before, net.layer.weight.detach())


def test_cellpose_training_batches_full_and_remainder(monkeypatch):
    class RecordingNet(torch.nn.Module):
        backbone = "sam_vitl"

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Conv2d(3, 3, kernel_size=1)
            self.batch_sizes = []

        def forward(self, images):
            self.batch_sizes.append(len(images))
            return (self.layer(images),)

    net = RecordingNet()
    adapter = CellposeAdapter()
    adapter.model = SimpleNamespace(net=net)
    adapter.device = torch.device("cpu")
    adapter.make_optimizer({"batch_size": 2, "learning_rate": 0.01, "weight_decay": 0})
    monkeypatch.setattr(
        cellpose_module,
        "_segmentation_loss",
        lambda: lambda target, output, device: (output - target.to(device)).square().mean(),
    )
    samples = [
        Sample(
            str(index),
            np.full((8, 8, 3), index + 1, dtype=np.float32),
            np.zeros((3, 8, 8), dtype=np.float32),
        )
        for index in range(3)
    ]

    loss = adapter.train_one_epoch(samples, 0.01)

    assert np.isfinite(loss)
    assert net.batch_sizes == [2, 1]


def test_dataset_and_flow_arrays_share_one_lru_budget(monkeypatch):
    from foam_cell_analysis.data.dataset_store import ArrayLRU

    cache = ArrayLRU(200)
    image = np.zeros((4, 4), dtype=np.uint8)
    cache.put(("dataset", "image"), image)
    cache.put(("dataset", "mask"), image.copy())
    monkeypatch.setattr(cellpose_module, "_flow_function", lambda: _fake_flows)
    adapter = CellposeAdapter()
    adapter.device = "cpu"
    adapter.model_config = {
        "scale_range": 0,
        "augmentation_profile": {"transforms": []},
        "_shared_array_cache": cache,
    }
    adapter._shared_array_cache = cache

    adapter._flows("item_a", np.ones((4, 4), dtype=np.int32))

    assert cache.size <= cache.capacity_bytes
    assert cache.size == 192
    assert cache.get(("dataset", "image")) is None


@pytest.mark.ml
@pytest.mark.slow
@pytest.mark.parametrize("pretrained_model", ["cpsam", "cpsam_v2"])
def test_cellpose_model_train_save_reload_and_infer_without_pretrained_cache(
    tmp_path, monkeypatch, pretrained_model
):
    import gc

    torch.set_num_threads(1)
    pretrained_dir = Path(__file__).resolve().parents[2] / "workspace" / "pretrained" / "cellpose"
    weight_file = pretrained_dir / pretrained_model
    if not weight_file.is_file():
        pytest.skip(f"{pretrained_model} の重みが workspace/pretrained/cellpose にありません")
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(pretrained_dir))
    config = {
        "type": "cellpose",
        "pretrained_model": pretrained_model,
        "bsize": 256,
        "scale_range": 0,
        "nimg_per_epoch": 2,
        "min_train_masks": 1,
        "augmentation_profile": {"transforms": []},
    }
    adapter = CellposeAdapter()
    adapter.build(config, torch.device("cpu"))
    assert adapter.model.net.backbone == "sam_vitl"
    assert all(parameter.dtype == torch.float32 for parameter in adapter.model.net.parameters())
    optimizer = adapter.make_optimizer(
        {"batch_size": 1, "learning_rate": 1e-5, "weight_decay": 0.1}
    )
    assert isinstance(optimizer, torch.optim.AdamW)

    samples = []
    for item_id, offset in (("sample_a", 0), ("sample_b", 4)):
        image = np.zeros((256, 256), dtype=np.float32)
        image[32 + offset : 80 + offset, 40:88] = 1
        image[128:176, 130 + offset : 178 + offset] = 0.8
        labels = np.zeros((256, 256), dtype=np.uint16)
        labels[32 + offset : 80 + offset, 40:88] = 1
        labels[128:176, 130 + offset : 178 + offset] = 2
        samples.append(
            adapter.prepare_sample(
                Sample(item_id, image, labels), training=True, rng=np.random.default_rng(1)
            )
        )
    first_loss = adapter.train_one_epoch(samples, adapter.lr(1, 2, 1e-5))
    second_loss = adapter.train_one_epoch(samples, adapter.lr(2, 2, 1e-5))
    assert np.isfinite(first_loss) and np.isfinite(second_loss)

    checkpoint = tmp_path / f"{pretrained_model}-final.pt"
    torch.save(adapter.state_dict(), checkpoint)
    assert all(
        not value.is_floating_point() or value.dtype == torch.float32
        for value in adapter.state_dict().values()
    )
    del samples, adapter
    gc.collect()

    local_model_cache = tmp_path / "empty-cellpose-models"
    local_model_cache.mkdir()
    empty_torch_home = tmp_path / "empty-torch-home"
    empty_torch_home.mkdir()
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(local_model_cache))
    monkeypatch.setenv("TORCH_HOME", str(empty_torch_home))
    cellpose_models = importlib.import_module("cellpose.models")
    monkeypatch.setattr(cellpose_models, "MODEL_DIR", local_model_cache)
    monkeypatch.setattr(
        cellpose_models, "MODEL_LIST_PATH", str(local_model_cache / "gui_models.txt")
    )
    restored = cellpose_module.build_for_inference(checkpoint.resolve(), "cpu")
    assert restored.model.net.backbone == "sam_vitl"
    input_image = np.zeros((256, 256), dtype=np.float32)
    input_image[32:80, 40:88] = 1
    image_channels = np.stack(
        (input_image, np.zeros_like(input_image), np.zeros_like(input_image)), axis=-1
    )
    prediction = restored.predict([image_channels])
    assert prediction[0].shape == input_image.shape
    assert list(local_model_cache.iterdir()) == []
    del restored
    gc.collect()


def _recording_model():
    class Net:
        def eval(self):
            pass

        def train(self):
            pass

    class Model:
        def __init__(self):
            self.net = Net()
            self.kwargs = None

        def eval(self, images, **kwargs):
            self.kwargs = kwargs
            return [np.zeros(image.shape[:2], dtype=np.uint16) for image in images], None, None

    return Model()


def test_default_eval_params_equal_previous_hardcoded_values():
    assert cellpose_module.DEFAULT_EVAL_PARAMS == {
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "min_size": 15,
        "max_size_fraction": 0.4,
    }
    assert CellposeAdapter().eval_params == cellpose_module.DEFAULT_EVAL_PARAMS


def test_predict_passes_configured_eval_params():
    adapter = CellposeAdapter()
    adapter.model = _recording_model()
    adapter.eval_params = cellpose_module.DEFAULT_EVAL_PARAMS | {
        "flow_threshold": 0.7,
        "cellprob_threshold": -1.5,
    }
    adapter.predict([np.zeros((8, 8, 3), dtype=np.float32)])

    kwargs = adapter.model.kwargs
    assert kwargs["flow_threshold"] == 0.7
    assert kwargs["cellprob_threshold"] == -1.5
    assert kwargs["min_size"] == 15
    assert kwargs["max_size_fraction"] == 0.4
    assert kwargs["bsize"] == 256


def test_build_for_inference_accepts_eval_params_and_keeps_old_signature(tmp_path, monkeypatch):
    models = pytest.importorskip("cellpose.models")
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"weights")

    class FakeCellposeModel:
        def __init__(self, *, pretrained_model, device, use_bfloat16):
            self.net = SimpleNamespace(backbone="sam_vitl")

    monkeypatch.setattr(models, "CellposeModel", FakeCellposeModel)

    plain = cellpose_module.build_for_inference(checkpoint, "cpu")
    tuned = cellpose_module.build_for_inference(
        checkpoint, "cpu", eval_params={"flow_threshold": 0.9}
    )

    assert plain.eval_params == cellpose_module.DEFAULT_EVAL_PARAMS
    assert tuned.eval_params["flow_threshold"] == 0.9
    assert tuned.eval_params["min_size"] == 15
