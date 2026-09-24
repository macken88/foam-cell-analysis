"""ページ共通の見出しと説明。"""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..context import AppContext


class BasePage(QWidget):
    """全画面が継承する見出し付きページ。"""

    def __init__(self, ctx: AppContext, title: str, description: str = "", parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        self.heading = QLabel(title)
        self.heading.setObjectName("pageHeading")
        self.heading.setStyleSheet("font-size: 22px; font-weight: 600;")
        self.description = QLabel(description)
        self.description.setStyleSheet("color: #666;")
        layout.addWidget(self.heading)
        layout.addWidget(self.description)
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)

    def on_enter(self, params: dict) -> None:
        """ページが表示されるたびに呼び出す。"""
