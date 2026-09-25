"""ページ識別子と画面遷移管理。"""

from enum import StrEnum
from typing import Any

from PySide6.QtCore import QObject, Signal


class PageId(StrEnum):
    """アプリ内の全ページ。"""

    DATA_PREPARATION = "data_preparation"
    DATASET_HISTORY = "dataset_history"
    TRAINING = "training"
    EXPERIMENTS = "experiments"
    CANDIDATES = "candidates"
    MASK_COMPARISON = "mask_comparison"
    RELEASED_MODELS = "released_models"
    INFERENCE = "inference"


class ModeId(StrEnum):
    """独立ウィンドウを持つアプリのモード。"""

    DATA_PREPARATION = "data_preparation"
    TRAINING = "training"
    COMPARISON = "comparison"
    INFERENCE = "inference"


class Navigator(QObject):
    """ウィンドウ管理側へページ遷移を通知する。"""

    navigation_requested = Signal(object, dict)

    def navigate(self, page_id: PageId, **params: Any) -> None:
        """指定ページへ遷移する。"""
        self.navigation_requested.emit(page_id, params)
