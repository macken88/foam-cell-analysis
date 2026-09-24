"""QPainter で描く軽量折れ線グラフ。"""

from itertools import pairwise

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

Series = tuple[str, QColor, list[float], list[float]]


class LineChart(QWidget):
    """凡例、軸目盛、強調点を持つ複数系列グラフ。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.series: list[Series] = []
        self.highlight: tuple[str, float, float] | None = None
        self.setMinimumHeight(200)

    def set_series(self, series: list[Series]) -> None:
        """系列名・色・X値・Y値を設定する。"""
        self.series = series
        self.update()

    def set_highlight(self, name: str, x: float, y: float) -> None:
        """系列上の強調点を設定する。"""
        self.highlight = (name, x, y)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(48, 14, -14, -40)
        painter.setPen(QPen(QColor("#888"), 1))
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
            ymax = ymin + 1

        for fraction in (0.0, 0.5, 1.0):
            y = rect.bottom() - fraction * rect.height()
            value = ymin + fraction * (ymax - ymin)
            painter.setPen(QColor("#555"))
            painter.drawText(0, int(y + 5), 42, 18, Qt.AlignmentFlag.AlignRight, f"{value:.3g}")
            painter.setPen(QPen(QColor("#ddd"), 1))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        painter.setPen(QColor("#555"))
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

        for series_index, (name, color, xs, ys) in enumerate(self.series):
            coords = [map_point(x, y) for x, y in zip(xs, ys, strict=False)]
            painter.setPen(QPen(color, 2))
            for left, right in pairwise(coords):
                painter.drawLine(left, right)
            legend_y = rect.top() + 14 * (series_index + 1)
            painter.drawLine(rect.left() + 4, legend_y, rect.left() + 20, legend_y)
            painter.setPen(color)
            painter.drawText(rect.left() + 26, legend_y + 5, name)
        if self.highlight:
            name, x, y = self.highlight
            series = next((entry for entry in self.series if entry[0] == name), None)
            if series:
                painter.setBrush(series[1])
                painter.drawEllipse(map_point(x, y), 5, 5)
