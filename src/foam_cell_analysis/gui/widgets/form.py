"""フォーム用の共通セクションと入力部品。"""

import re

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..labels import format_exponent

_PARTIAL_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d*)?([eE][+-]?\d*)?")


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """指数表記（1e-5、1.0e-5）と通常の小数を受け付け、指数表記で表示する数値欄。

    小数点以下の桁数制限を持たない。学習率のように桁の小さい値に使う。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # QDoubleSpinBox は値を decimals 桁に丸めるため、十分大きくして丸めを避ける
        self.setDecimals(60)
        self.setRange(0.0, 1.0e6)

    def textFromValue(self, value: float) -> str:
        return format_exponent(value)

    def valueFromText(self, text: str) -> float:
        try:
            return float(text.strip())
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        stripped = text.strip()
        try:
            value = float(stripped)
        except ValueError:
            state = (
                QValidator.State.Intermediate
                if _PARTIAL_NUMBER.fullmatch(stripped)
                else QValidator.State.Invalid
            )
            return state, text, pos
        if self.minimum() <= value <= self.maximum():
            return QValidator.State.Acceptable, text, pos
        return QValidator.State.Intermediate, text, pos

    def fixup(self, text: str) -> str:
        return self.textFromValue(self.value())

    def stepBy(self, steps: int) -> None:
        """上下キーでは 10 倍・10 分の 1 ずつ変える。"""
        value = self.value() or 1.0e-5
        self.setValue(value * (10.0**steps))


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
