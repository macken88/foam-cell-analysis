"""フォーム用の共通セクション。"""

from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class FormSection(QGroupBox):
    """ラベルと入力欄を追加できるフォーム枠。"""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(title, parent)
        self._section_title = title
        super().setTitle("")
        self.section_layout = QVBoxLayout(self)
        self.section_layout.setContentsMargins(0, 0, 0, 0)
        self.heading_label = QLabel(title, self)
        self.heading_label.setProperty("role", "sectionHeading")
        self.heading_rule = QFrame(self)
        self.heading_rule.setFrameShape(QFrame.Shape.HLine)
        self.heading_rule.setProperty("role", "sectionHeadingRule")
        self.heading_label.hide()
        self.heading_rule.hide()
        self.section_layout.addWidget(self.heading_label)
        self.section_layout.addWidget(self.heading_rule)
        self.form = QFormLayout()
        self.section_layout.addLayout(self.form)

    def title(self) -> str:
        return self._section_title

    def setTitle(self, title: str) -> None:
        self._section_title = title
        self.heading_label.setText(title)

    def set_prominent_heading(self, enabled: bool = True) -> None:
        self.heading_label.setVisible(enabled)
        self.heading_rule.setVisible(enabled)

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
