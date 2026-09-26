"""用途・状態タグと操作ヒント用の小さな部品。"""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QWidget,
)

from ..theme import Color, numeric_font, set_style

USAGE_MARKS = {
    "train": ("学習", Color.TRAIN_BG, Color.TRAIN, False),
    "val": ("検証", Color.VAL_BG, Color.VAL_INK, False),
    "unassigned": ("未振り分け", Color.IDLE_BG, Color.SLATE, False),
    "excluded": ("不採用", Color.IDLE_BG, Color.SLATE, True),
}
STATUS_MARKS = {
    "実行中": (Color.GRAPHITE, Color.WHITE, False),
    "評価中": (Color.GRAPHITE, Color.WHITE, False),
    "完了": (Color.OK_BG, Color.OK, False),
    "下書き": (Color.IDLE_BG, Color.SLATE, False),
    "中断": (Color.IDLE_BG, Color.SLATE, False),
    "候補": (Color.IDLE_BG, Color.SLATE, False),
    "失敗": (Color.ERROR_BG, Color.ERROR, False),
    "リリース済み": (Color.OK_BG, Color.OK, False),
    "非採用": (Color.IDLE_BG, Color.SLATE, True),
}


class UsageTag(QLabel):
    """データの用途を色と文字で示す。"""

    def __init__(self, usage: str, parent=None) -> None:
        text = USAGE_MARKS[usage][0]
        super().__init__(text, parent)
        set_style(self, usage=usage)


class StatusTag(QLabel):
    """実験・候補・リリースの状態を示す。"""

    def __init__(self, status: str, parent=None) -> None:
        background, foreground, struck = STATUS_MARKS[status]
        super().__init__(status, parent)
        set_style(self, state=status)


class TagDelegate(QStyledItemDelegate):
    """表示文字列に対応する用途・状態色でタグを描画する。"""

    def __init__(self, mapping: dict[str, tuple[str, str, bool]], parent=None) -> None:
        super().__init__(parent)
        self.mapping = mapping

    def paint(self, painter: QPainter, option, index) -> None:
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        style = self.mapping.get(text)
        if style is None:
            super().paint(painter, option, index)
            return
        background, foreground, struck = style
        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        rect = option.rect.adjusted(5, 3, -5, -3)
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QColor(background))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(QColor(foreground))
        font = option.font
        font.setStrikeOut(struck)
        painter.setFont(font)
        painter.drawText(rect.adjusted(7, 0, -7, 0), Qt.AlignmentFlag.AlignVCenter, text)
        painter.restore()


class LayoutButton(QPushButton):
    """中に置いたレイアウトの大きさに合わせるボタン。

    QPushButton の既定の sizeHint は自身の文字しか見ないため、
    子レイアウトで中身を並べるボタンではこちらを使う。
    """

    def sizeHint(self) -> QSize:  # noqa: N802
        layout = self.layout()
        return layout.sizeHint() if layout is not None else super().sizeHint()

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()


class CountChip(LayoutButton):
    """用途色見本と件数を持つ絞り込みボタン。"""

    def __init__(self, name: str, count: int, usage: str = "plain", parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.name = name
        self.count = count
        self.usage = usage
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(5)
        self.swatch = QLabel("", self)
        self.swatch.setFixedSize(9, 9)
        set_style(self.swatch, role="countSwatch", usage=usage)
        self.name_label = QLabel(name, self)
        self.count_label = QLabel(str(count), self)
        self.count_label.setFont(numeric_font())
        layout.addWidget(self.swatch)
        layout.addWidget(self.name_label)
        layout.addWidget(self.count_label)
        self.swatch.setVisible(
            usage != "error" and usage in {"train", "val", "unassigned", "excluded"}
        )
        set_style(self, role="countChip", usage=usage)


class KeyCap(QLabel):
    """キーボードのキー名を表示する。"""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        set_style(self, role="keycap")


class KeyHintBar(QWidget):
    """キーと操作説明の一覧を横並びで表示する。"""

    def __init__(self, hints: list[tuple[str, str]], parent=None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QHBoxLayout

        layout = QHBoxLayout(self)
        self._layout = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._groups: list[QWidget] = []
        self.set_hints(hints)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_hints(self, hints: list[tuple[str, str]]) -> None:
        """キー割り当て変更後にキー説明を作り直す。"""
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._groups.clear()
        for key, description in hints:
            group = QWidget(self)
            group_layout = QHBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(3)
            keycap = KeyCap(key)
            label = QLabel(description)
            keycap.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            group_layout.addWidget(keycap)
            group_layout.addWidget(label)
            self._groups.append(group)
            self._layout.addWidget(group)
        self._layout.addStretch(1)
        self._update_visible_groups()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_visible_groups()

    def _update_visible_groups(self) -> None:
        """幅に収まる先頭項目と末尾のキー一覧だけを表示する。"""
        if not self._groups:
            return
        last_index = len(self._groups) - 1
        available = self.contentsRect().width()
        spacing = self._layout.spacing()
        widths = [group.sizeHint().width() for group in self._groups]
        visible = {last_index}
        used = widths[last_index]
        for index in range(last_index):
            needed = widths[index] + spacing * (len(visible))
            if used + needed > available:
                break
            visible.add(index)
            used += needed
        for index, group in enumerate(self._groups):
            group.setVisible(index in visible)
