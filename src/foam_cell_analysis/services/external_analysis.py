"""外部解析値の検証と集計（比較・評価設計 9.4）。"""

from __future__ import annotations

import math
from typing import Any

# 外部解析（9.4）: 画像ごとの「検出された全気泡の円相当径の中央値」を手入力する
EXTERNAL_FORMAT = "median_equivalent_diameter_v1"
EXTERNAL_METRIC = "median_equivalent_diameter"
EXTERNAL_UNITS = ("µm", "px")
UNCLASSIFIED = "未分類"


def is_external_analysis(record: Any) -> bool:
    """現行形式（画像ごとの円相当径の中央値）の外部解析の記録か。旧形式の自由項目は False。"""
    return isinstance(record, dict) and record.get("format") == EXTERNAL_FORMAT


def _external_value(item_id: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{item_id} の値が数値ではありません")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{item_id} の値が有限の数値ではありません")
    if number < 0:
        raise ValueError(f"{item_id} の値が負です")
    return number


def summarize_external_values(
    values: dict[str, float], classifications: dict[str, str | None]
) -> dict[str, Any]:
    """画像ごとの値の平均（全体・分類別）と、入力済み枚数／全枚数を求める。

    classifications は評価対象の全画像（item_id → 分類）。空欄の画像は平均に含めない。
    """

    def mean(numbers: list[float]) -> float | None:
        return sum(numbers) / len(numbers) if numbers else None

    groups: dict[str, list[str]] = {}
    for item_id, name in classifications.items():
        groups.setdefault(name or UNCLASSIFIED, []).append(item_id)
    per_class = {}
    for name in sorted(groups):
        entered = [values[item_id] for item_id in groups[name] if item_id in values]
        per_class[name] = {
            "mean": mean(entered),
            "n_images": len(entered),
            "n_total": len(groups[name]),
        }
    return {
        "mean": mean(list(values.values())),
        "n_images": len(values),
        "n_total": len(classifications),
        "per_class": per_class,
    }


def build_external_analysis(
    values: dict[str, Any],
    classifications: dict[str, str | None],
    *,
    unit: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """入力値を検証し、（保存する値, 集計）を返す。不正なら ValueError（何も保存しない）。

    None・空欄は未入力（値の削除）として扱う。評価対象にない画像・負値・NaN・無限大は拒否する。
    """
    if unit not in EXTERNAL_UNITS:
        raise ValueError("単位は µm または px を選んでください")
    if not isinstance(values, dict):
        raise ValueError("外部解析の値の形が不正です")
    unknown = sorted(str(item_id) for item_id in values if item_id not in classifications)
    if unknown:
        raise ValueError("評価の対象にない画像があります: " + "、".join(unknown[:5]))
    cleaned = {
        item_id: _external_value(item_id, value)
        for item_id, value in sorted(values.items())
        if value is not None and value != ""
    }
    return cleaned, summarize_external_values(cleaned, classifications)
