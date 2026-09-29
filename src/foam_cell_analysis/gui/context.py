"""ページ間で共有するアプリケーションコンテキスト。"""

from dataclasses import dataclass, field
from datetime import datetime

from PySide6.QtCore import QObject, QSettings, Signal

from ..services.backend import Backend
from .compute_coordinator import ComputeCoordinator
from .jobs import JobManager
from .navigation import Navigator
from .settings import app_settings
from .shortcuts import ShortcutMap

DEFAULT_CHANNEL = "A"  # GUI は現時点で先頭チャンネルだけを使う。複数チャンネル対応時に拡張する。


class StatusBus(QObject):
    """ページからメインウィンドウへ状態表示を通知する。"""

    message = Signal(str)
    saved = Signal(object)

    def show_message(self, message: str) -> None:
        """ステータスバーへメッセージを通知する。"""
        self.message.emit(message)

    def notify_saved(self, value: datetime) -> None:
        """自動保存時刻を通知する。"""
        self.saved.emit(value)


class DisplayPreference(QObject):
    """原画像と切り替える表示を保存して全画面へ通知する。"""

    changed = Signal(str)
    MODES = {"オーバーレイ", "インスタンスラベル", "二値マスク"}

    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or app_settings()
        value = self.settings.value("display/alternate", "オーバーレイ")
        self._value = value if value in self.MODES else "オーバーレイ"

    @property
    def value(self) -> str:
        """現在の切替表示を返す。"""
        return self._value

    def set_value(self, value: str) -> None:
        """設定を保存し、全画面へ変更を通知する。"""
        if value not in self.MODES or value == self._value:
            return
        self._value = value
        self.settings.setValue("display/alternate", value)
        self.changed.emit(value)


@dataclass
class AppContext:
    """Backend、画面遷移、バックグラウンドジョブを束ねる。"""

    backend: Backend
    navigator: Navigator
    jobs: JobManager
    status: StatusBus = field(default_factory=StatusBus)
    shortcuts: ShortcutMap = field(default_factory=ShortcutMap)
    display: DisplayPreference = field(default_factory=DisplayPreference)
    keymap_window: QObject | None = None
    queue_controller: QObject | None = None
    training_runner: QObject | None = None
    workspace_lock: QObject | None = None
    # 学習と評価の計算処理を 1 件ずつにする排他制御（比較・推論設計 15.1）
    compute: ComputeCoordinator = field(default_factory=ComputeCoordinator)
    # 検証用データセットでの評価を候補ごとに順に実行する（比較・推論設計 15.2）
    evaluation_runner: QObject | None = None

    def __post_init__(self) -> None:
        """すべての画面が共有する学習・評価の runner を準備する。"""
        if self.training_runner is None:
            from .training_runner import TrainingRunner

            self.training_runner = TrainingRunner(self.backend, compute=self.compute)
        if self.evaluation_runner is None:
            from .evaluation_runner import EvaluationRunner

            self.evaluation_runner = EvaluationRunner(self.backend, compute=self.compute)
        # 候補の非採用・リリース・成果物の整理で「評価中・評価待ち」を判定できるようにする
        set_activity = getattr(self.backend, "set_evaluation_activity", None)
        is_active = getattr(self.evaluation_runner, "is_evaluation_active", None)
        if callable(set_activity) and callable(is_active):
            set_activity(is_active)
        if hasattr(self.jobs, "set_training_runner"):
            self.jobs.set_training_runner(self.training_runner)
