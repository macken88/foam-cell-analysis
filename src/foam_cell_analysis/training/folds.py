"""グループを保った決定的な交差検証 fold 割り当て。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from foam_cell_analysis.services.models import DataItem

FOLD_ALGORITHM = "group_greedy_v1"


def assign_folds(
    items: Sequence[DataItem],
    n_folds: int = 5,
    seed: int = 42,
    *,
    group_by_source_folder: bool = True,
    stratify_by_classification: bool = True,
) -> dict[str, int]:
    """分類件数と全体件数を均し、同じフォルダを同じ fold に割り当てる。"""
    ordered = sorted(items, key=lambda item: item.item_id)
    if len(ordered) < n_folds:
        raise ValueError(f"交差検証には画像が最低 {n_folds} 件必要です")
    groups: dict[str, list[DataItem]] = {}
    for item in ordered:
        key = item.source_folder if group_by_source_folder else item.item_id
        groups.setdefault(key, []).append(item)
    if len(groups) < n_folds:
        raise ValueError(f"交差検証には異なる取込元フォルダが最低 {n_folds} 個必要です")
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    fold_counts = [0] * n_folds
    class_counts: list[dict[str, int]] = [dict() for _ in range(n_folds)]
    target_count = len(ordered) / n_folds
    class_totals: dict[str, int] = {}
    for group in groups.values():
        for item in group:
            label = item.classification or "未分類"
            class_totals[label] = class_totals.get(label, 0) + 1
    assignments: dict[str, int] = {}
    for key in keys:
        group = groups[key]
        classes: dict[str, int] = {}
        for item in group:
            label = item.classification or "未分類"
            classes[label] = classes.get(label, 0) + 1

        def placement_cost(
            fold_index: int,
            group_size: int = len(group),
            group_classes: dict[str, int] = classes,
        ) -> float:
            previous_total = (fold_counts[fold_index] - target_count) ** 2
            next_total = (fold_counts[fold_index] + group_size - target_count) ** 2
            cost = next_total - previous_total
            if stratify_by_classification:
                for label, class_total in class_totals.items():
                    target_class = class_total / n_folds
                    before = (class_counts[fold_index].get(label, 0) - target_class) ** 2
                    after = (
                        class_counts[fold_index].get(label, 0)
                        + group_classes.get(label, 0)
                        - target_class
                    ) ** 2
                    cost += after - before
            return cost

        fold = min(range(n_folds), key=placement_cost)
        for item in group:
            assignments[item.item_id] = fold + 1
            label = item.classification or "未分類"
            class_counts[fold][label] = class_counts[fold].get(label, 0) + 1
            fold_counts[fold] += 1
    return assignments
