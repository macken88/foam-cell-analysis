"""QPainter で描く軽量折れ線グラフ。"""

from itertools import pairwise

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..theme import SERIES, Color, numeric_font

Series = tuple[str, QColor, list[float], list[float]]


class LineChart(QWidget):
    """軸目盛、系列名、強調点を持つ複数系列グラフ。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.series: list[Series] = []
        self.highlight: tuple[str, float, float] | None = None
        self.best: tuple[float, float] | None = None
        self.setMinimumHeight(200)

    def set_series(self, series: list[Series]) -> None:
        """系列名・色・X値・Y値を設定する。"""
        self.series = series
        self.update()

    def set_highlight(self, name: str, x: float, y: float) -> None:
        """既存の系列強調点を設定する。"""
        self.highlight = (name, x, y)
        self.update()

    def set_best(self, epoch: float | None, value: float | None) -> None:
        """最良エポックの印を表示する。None を渡すと消す。"""
        self.best = (epoch, value) if epoch is not None and value is not None else None
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(52, 12, -92 if len(self.series) > 1 else -14, -40)
        painter.setPen(QPen(QColor(Color.RULE), 1))
        painter.drawRect(rect)
        points = [(x, y) for _, _, xs, ys in self.series for x, y in zip(xs, ys, strict=False)]
        if not points:
            return
        xmin = min(x for x, _ in points)
        xmax = max(x for x, _ in points)
        ymin = min(y for _, y in points)
        ymax = max(y for _, y in points)
        if xmin == xmax:
            xmax = xmin + 1
        if ymin == ymax:
            ymin -= 0.5
            ymax += 0.5

        for fraction in (0.0, 0.5, 1.0):
            y = rect.bottom() - fraction * rect.height()
            value = ymin + fraction * (ymax - ymin)
            painter.setPen(QColor(Color.SLATE))
            painter.setFont(numeric_font())
            painter.drawText(0, int(y + 5), 46, 18, Qt.AlignmentFlag.AlignRight, f"{value:.3g}")
            painter.setPen(QPen(QColor(Color.RULE_SOFT), 1))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        painter.setPen(QColor(Color.SLATE))
        painter.setFont(numeric_font())
        painter.drawText(
            rect.left(),
            rect.bottom() + 28,
            rect.width(),
            18,
            Qt.AlignmentFlag.AlignCenter,
            "エポック",
        )
        painter.drawText(rect.left(), rect.bottom() + 16, f"{xmin:g}")
        painter.drawText(rect.right() - 32, rect.bottom() + 16, f"{xmax:g}")

        def map_point(x: float, y: float) -> QPointF:
            px = rect.left() + (x - xmin) * rect.width() / (xmax - xmin)
            py = rect.bottom() - (y - ymin) * rect.height() / (ymax - ymin)
            return QPointF(px, py)

        end_labels: list[tuple[str, QColor, float]] = []
        for series_index, (name, _color, xs, ys) in enumerate(self.series):
            color = QColor(
                Color.GRAPHITE if len(self.series) == 1 else SERIES[series_index % len(SERIES)]
            )
            coords = [map_point(x, y) for x, y in zip(xs, ys, strict=False)]
            painter.setPen(QPen(color, 2))
            for left, right in pairwise(coords):
                painter.drawLine(left, right)
            if len(self.series) == 1:
                continue
            if coords:
                end_labels.append((name, color, coords[-1].y()))

        if end_labels:
            end_labels.sort(key=lambda entry: entry[2])
            label_positions = [max(entry[2], rect.top() + 8) for entry in end_labels]
            for index in range(1, len(label_positions)):
                label_positions[index] = max(
                    label_positions[index], label_positions[index - 1] + 13
                )
            overflow = label_positions[-1] - (rect.bottom() - 2)
            if overflow > 0:
                label_positions = [position - overflow for position in label_positions]
            for (name, color, _), label_y in zip(end_labels, label_positions, strict=True):
                painter.setPen(color)
                painter.drawText(
                    rect.right() + 6, int(label_y + 4), 82, 16, Qt.AlignmentFlag.AlignLeft, name
                )

        if self.highlight:
            name, x, y = self.highlight
            entry_index = next(
                (index for index, item in enumerate(self.series) if item[0] == name), None
            )
            entry = self.series[entry_index] if entry_index is not None else None
            if entry:
                color = (
                    Color.GRAPHITE if len(self.series) == 1 else SERIES[entry_index % len(SERIES)]
                )
                painter.setBrush(QColor(color))
                painter.drawEllipse(map_point(x, y), 5, 5)

        if self.best:
            epoch, value = self.best
            point = map_point(epoch, value)
            painter.setPen(QPen(QColor(Color.SLATE), 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(point.x(), rect.top()), QPointF(point.x(), rect.bottom()))
            painter.setBrush(QColor(Color.SLIDE))
            painter.setPen(QPen(QColor(Color.GRAPHITE), 2))
            painter.drawEllipse(point, 5, 5)
            painter.setPen(QColor(Color.GRAPHITE))
            painter.setFont(numeric_font())
            painter.drawText(int(point.x() + 7), int(point.y() - 7), f"{value:.3g}")
