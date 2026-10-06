"""ページ共通の見出しと説明。"""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..context import AppContext
from ..theme import set_style


class BasePage(QWidget):
    """全画面が継承する見出し付きページ。"""

    def __init__(
        self,
        ctx: AppContext,
        title: str,
        description: str = "",
        parent=None,
        show_heading: bool = True,
    ) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        self.heading = QLabel(title)
        self.heading.setObjectName("pageHeading")
        self.description = QLabel(description)
        set_style(self.description, role="note")
        self.heading.setVisible(show_heading)
        self.description.setVisible(show_heading)
        layout.addWidget(self.heading)
        layout.addWidget(self.description)
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)

    def on_enter(self, params: dict) -> None:
        """ページが表示されるたびに呼び出す。"""

    def refresh_on_activate(self) -> None:
        """ウィンドウの再表示時に状態を保ちながらページを更新する。"""
        self.on_enter({})

    def refresh_menu_actions(self) -> None:
        """メニュー操作の可否を更新する。ページごとの必要な状態だけを再評価する。"""

    def set_menu_action_enabled(self, action, enabled: bool) -> None:
        """共有メニュー項目の本来の有効状態をモードウィンドウへ伝える。"""
        window = self.window()
        setter = getattr(window, "set_page_action_enabled", None)
        if callable(setter):
            setter(action, enabled)
        else:
            action.setEnabled(enabled)
