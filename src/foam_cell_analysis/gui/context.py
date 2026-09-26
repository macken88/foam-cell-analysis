"""ページ間で共有するアプリケーションコンテキスト。"""

from dataclasses import dataclass, field
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from ..services.backend import Backend
from .jobs import JobManager
from .navigation import Navigator
from .shortcuts import ShortcutMap


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


@dataclass
class AppContext:
    """Backend、画面遷移、バックグラウンドジョブを束ねる。"""

    backend: Backend
    navigator: Navigator
    jobs: JobManager
    status: StatusBus = field(default_factory=StatusBus)
    shortcuts: ShortcutMap = field(default_factory=ShortcutMap)
