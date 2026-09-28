import numpy as np
import pytest

from foam_cell_analysis.training.augmentation import apply_profile
from foam_cell_analysis.training.preprocessing import normalize_image


def _profile(key, low=None, high=None):
    setting = {"key": key, "enabled": True, "probability": 1.0}
    if low is not None:
        setting["range_min"] = low
        setting["range_max"] = high
    return {"order": [key], "transforms": [setting]}


def test_percentile_normalization_clips_and_handles_constant_images():
    image = np.arange(100, dtype=np.uint16).reshape(10, 10)
    result = normalize_image(image, {"low_percentile": 10, "high_percentile": 90})
    assert result.dtype == np.float32
    assert result.min() == 0 and result.max() == 1
    constant = normalize_image(np.full((4, 5), 7, dtype=np.uint16))
    assert constant.dtype == np.float32 and not constant.any()


@pytest.mark.ml
@pytest.mark.parametrize(
    ("key", "low", "high"),
    [
        ("horizontal_flip", None, None),
        ("vertical_flip", None, None),
        ("rotation", 17.0, 17.0),
        ("scale", 1.1, 1.1),
        ("translation", 0.05, 0.05),
        ("crop", 0.7, 0.7),
        ("elastic", 0.2, 0.2),
        ("brightness", 10.0, 10.0),
        ("contrast", 1.1, 1.1),
        ("gamma", 0.8, 0.8),
        ("blur", 0.5, 0.5),
        ("noise", 3.0, 3.0),
        ("channel_dropout", 0.0, 0.0),
        ("channel_intensity", 1.2, 1.2),
    ],
)
def test_profile_transforms_preserve_shape_and_integer_labels(key, low, high):
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:20, 9:21] = 0.99
    labels = np.zeros((32, 32), dtype=np.uint32)
    labels[8:20, 9:21] = 70000
    output, transformed = apply_profile(
        image,
        labels,
        _profile(key, low, high),
        np.random.default_rng(12),
    )
    assert output.shape == image.shape and output.dtype == np.float32
    assert output.min() >= 0 and output.max() <= 1
    assert transformed.shape == labels.shape and transformed.dtype == np.uint32
    assert set(np.unique(transformed)).issubset({0, 70000})


@pytest.mark.ml
def test_profile_pipeline_order_and_image_only_clipping():
    image = np.full((8, 8), 0.99, dtype=np.float32)
    labels = np.full((8, 8), 70001, dtype=np.uint32)
    profile = {
        "order": ["brightness", "horizontal_flip"],
        "transforms": [
            {
                "key": "brightness",
                "enabled": True,
                "probability": 1,
                "range_min": 10,
                "range_max": 10,
            },
            {"key": "horizontal_flip", "enabled": True, "probability": 1},
        ],
    }
    output, transformed = apply_profile(image, labels, profile, np.random.default_rng(1))
    assert np.all(output == 1.0)
    assert np.all(transformed == 70001)


@pytest.mark.ml
def test_elastic_preserves_large_sparse_labels_and_torch_global_rng():
    torch = pytest.importorskip("torch")
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 1
    labels = np.zeros((32, 32), dtype=np.uint32)
    labels[8:24, 8:24] = 16_777_217
    state_before = torch.random.get_rng_state().clone()
    _, transformed = apply_profile(
        image,
        labels,
        _profile("elastic", 0.2, 0.2),
        np.random.default_rng(123),
    )
    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)
    assert transformed.dtype == np.uint32
    assert set(np.unique(transformed)).issubset({0, 16_777_217})
