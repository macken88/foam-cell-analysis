"""モデル種類ごとの学習率スケジュール。"""

from __future__ import annotations


def learning_rate(model_type: str, epoch: int, total_epochs: int, base_lr: float) -> float:
    """1 始まりの epoch に対応する学習率を返す。"""
    if epoch < 1 or total_epochs < 1 or epoch > total_epochs:
        raise ValueError("epoch は 1 から total_epochs の範囲で指定してください")
    if model_type != "cellpose":
        return float(base_lr)

    # cellpose 4.2 train_seg の np.linspace(0, lr, 10) を 1 始まりに写す。
    if epoch <= 10:
        value = base_lr * (epoch - 1) / 9
    else:
        value = base_lr
    if total_epochs > 300 and epoch > total_epochs - 100:
        steps = (epoch - (total_epochs - 100) - 1) // 10 + 1
        value *= 0.5**steps
    elif total_epochs > 99 and epoch > total_epochs - 50:
        steps = (epoch - (total_epochs - 50) - 1) // 5 + 1
        value *= 0.5**steps
    return float(value)


def lr(epoch: int, total_epochs: int, base_lr: float, model_type: str = "mask_rcnn") -> float:
    """互換性のため短い名前でもスケジュールを呼べる。"""
    return learning_rate(model_type, epoch, total_epochs, base_lr)
