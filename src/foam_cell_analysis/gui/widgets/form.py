"""フォーム用の共通セクション。"""

from PySide6.QtWidgets import QFormLayout, QGroupBox, QToolButton, QVBoxLayout, QWidget


class FormSection(QGroupBox):
    """ラベルと入力欄を追加できるフォーム枠。"""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(title, parent)
        self.form = QFormLayout(self)

    def add_row(self, label: str, widget: QWidget, tooltip: str = "") -> None:
        """入力欄を追加し、内部キーをツールチップに設定する。"""
        if tooltip:
            widget.setToolTip(tooltip)
        self.form.addRow(label, widget)


class CollapsibleSection(QWidget):
    """ボタンで内容を開閉する設定セクション。"""

    def __init__(self, title: str, content: QWidget | None = None, parent=None) -> None:
        super().__init__(parent)
        self.button = QToolButton()
        self.button.setText(f"{title} ▶")
        self.button.setCheckable(True)
        self.content = content or QWidget()
        self.content.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)
        layout.addWidget(self.content)
        self.button.toggled.connect(self._toggle)
        self.title = title

    def _toggle(self, opened: bool) -> None:
        self.content.setVisible(opened)
        self.button.setText(f"{self.title} {'▼' if opened else '▶'}")
