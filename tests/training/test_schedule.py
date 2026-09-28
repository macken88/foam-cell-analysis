import numpy as np
import pytest

from foam_cell_analysis.training.schedule import learning_rate


def _cellpose_reference(total_epochs, base_lr):
    values = np.linspace(0, base_lr, 10)
    values = np.append(values, base_lr * np.ones(max(0, total_epochs - 10)))
    if total_epochs > 300:
        values = values[:-100]
        for _ in range(10):
            values = np.append(values, values[-1] / 2 * np.ones(10))
    elif total_epochs > 99:
        values = values[:-50]
        for _ in range(10):
            values = np.append(values, values[-1] / 2 * np.ones(5))
    return values


@pytest.mark.parametrize("total_epochs", [1, 10, 40, 99, 100, 150, 300, 301, 400])
def test_cellpose_schedule_matches_train_seg_formula(total_epochs):
    base_lr = 0.012
    expected = _cellpose_reference(total_epochs, base_lr)
    actual = [
        learning_rate("cellpose", epoch, total_epochs, base_lr)
        for epoch in range(1, total_epochs + 1)
    ]
    assert actual == pytest.approx(expected[:total_epochs].tolist())


def test_mask_rcnn_schedule_is_constant_and_epoch_bounds_are_checked():
    assert learning_rate("mask_rcnn", 1, 20, 0.25) == 0.25
    assert learning_rate("fake", 20, 20, 0.25) == 0.25
    with pytest.raises(ValueError):
        learning_rate("cellpose", 0, 20, 0.01)
