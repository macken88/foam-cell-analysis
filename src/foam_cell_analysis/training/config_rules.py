"""学習設定の数値範囲を検証する標準ライブラリだけの共通ルール。"""

from __future__ import annotations

import math
from typing import Any


def validate_numeric_config(config: dict[str, Any]) -> list[tuple[str, str]]:
    """(設定キー, 利用者向け説明) の形で数値条件違反を返す。"""
    errors: list[tuple[str, str]] = []

    def add(key: str, valid: bool, message: str) -> None:
        if not valid:
            errors.append((key, message))

    model = config.get("model", {})
    training = config.get("training", {})
    checkpoint = config.get("checkpoint", {})
    if not all(isinstance(value, dict) for value in (model, training, checkpoint)):
        return [("config", "学習設定の数値項目を確認してください")]

    def finite(value: Any) -> bool:
        return type(value) is int or (type(value) is float and math.isfinite(value))

    if model.get("type") == "cellpose":
        value = model.get("scale_range")
        add(
            "model.scale_range",
            finite(value) and 0 <= value <= 1,
            "scale_range は 0〜1 で指定してください",
        )
        value = model.get("nimg_per_epoch")
        add(
            "model.nimg_per_epoch",
            value is None or (type(value) is int and value >= 1),
            "nimg_per_epoch は正の整数または自動です",
        )
        value = model.get("min_train_masks")
        add(
            "model.min_train_masks",
            type(value) is int and value >= 0,
            "min_train_masks は 0 以上の整数です",
        )
        value = model.get("bsize")
        add("model.bsize", type(value) is int and value == 256, "Cellpose の bsize は 256 固定です")
        value = training.get("batch_size")
        add(
            "training.batch_size",
            type(value) is int and value >= 1,
            "Cellpose の学習 batch_size は 1 以上です",
        )

    value = training.get("epochs")
    add("training.epochs", type(value) is int and value >= 1, "epochs は 1 以上の整数です")
    value = training.get("learning_rate")
    add(
        "training.learning_rate",
        finite(value) and value > 0,
        "learning_rate は 0 より大きい有限値です",
    )
    if "weight_decay" in training:
        value = training["weight_decay"]
        add(
            "training.weight_decay",
            finite(value) and value >= 0,
            "weight_decay は 0 以上の有限値です",
        )
    early = training.get("early_stopping")
    valid_early = (
        isinstance(early, dict)
        and type(early.get("enabled")) is bool
        and type(early.get("patience")) is int
        and early["patience"] >= 1
    )
    add(
        "training.early_stopping",
        valid_early,
        "early_stopping の有効/無効と patience（1 以上）を確認してください",
    )
    for key in ("validation_interval", "save_every"):
        value = checkpoint.get(key)
        add(f"checkpoint.{key}", type(value) is int and value >= 1, f"{key} は 1 以上の整数です")
    value = checkpoint.get("save_fold_models")
    add(
        "checkpoint.save_fold_models",
        type(value) is bool,
        "save_fold_models は有効/無効で指定してください",
    )
    return errors
