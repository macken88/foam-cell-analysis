"""QPainter で描く軽量折れ線グラフ。"""

from itertools import pairwise

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from ..theme import Color, body_font, numeric_font

Series = tuple[str, QColor, list[float], list[float]]


class LineChart(QWidget):
    """軸目盛、系列名、強調点を持つ複数系列グラフ。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.series: list[Series] = []
        self.highlight: tuple[str, float, float] | None = None
        self.best: tuple[float, float] | None = None
        self.y_min: float | None = None
        self.y_max: float | None = None
        self.y_axis_note = ""
        self.setMinimumHeight(180)
        self.setMouseTracking(True)

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

    def set_y_range(self, minimum: float | None = None, maximum: float | None = None) -> None:
        """縦軸範囲を設定する。指定しない端はデータから決める。"""
        self.y_min = minimum
        self.y_max = maximum
        self.y_axis_note = f"縦軸は {minimum:g} から" if minimum is not None else ""
        self.update()

    def _plot_rect(self):
        return self.rect().adjusted(52, 24 if self.y_axis_note else 12, -self._label_margin(), -40)

    def _label_margin(self) -> int:
        """右端の系列名が収まる余白を返す（系列が 1 本なら名前を出さない）。"""
        if len(self.series) <= 1:
            return 14
        metrics = QFontMetricsF(body_font())
        widest = max(
            metrics.horizontalAdvance(self._end_label(name)) for name, *_rest in self.series
        )
        return max(108, int(widest) + 16)

    @staticmethod
    def _end_label(name: str) -> str:
        """右端に書く名前。フォールドの細線はまとめて 1 つの名前にする。"""
        return "各フォールド（細線）" if name.startswith("分割 ") else name

    def mouseMoveEvent(self, event) -> None:
        """ポインター位置に最も近いデータ点の値をツールチップで示す。"""
        rect = self._plot_rect()
        points = [
            (name, x, y)
            for name, _color, xs, ys in self.series
            for x, y in zip(xs, ys, strict=False)
        ]
        if not points or not rect.contains(event.position().toPoint()):
            QToolTip.hideText()
            return
        xs = [x for _name, x, _y in points]
        xmin, xmax = min(xs), max(xs)
        if xmin == xmax:
            xmax += 1
        target_x = xmin + (event.position().x() - rect.left()) * (xmax - xmin) / rect.width()
        name, x, y = min(points, key=lambda item: abs(item[1] - target_x))
        QToolTip.showText(event.globalPosition().toPoint(), f"{name}　epoch {x:g}　{y:.3f}", self)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._plot_rect()
        points = [(x, y) for _, _, xs, ys in self.series for x, y in zip(xs, ys, strict=False)]
        if not points:
            return
        xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
        ymin = self.y_min if self.y_min is not None else min(y for _, y in points)
        ymax = self.y_max if self.y_max is not None else max(y for _, y in points)
        if xmin == xmax:
            xmax = xmin + 1
        if ymin == ymax:
            ymin -= 0.5
            ymax += 0.5

        if self.y_axis_note:
            painter.setPen(QColor(Color.SLATE))
            painter.setFont(body_font())
            prefix = "縦軸は "
            suffix = " から"
            prefix_width = QFontMetricsF(painter.font()).horizontalAdvance(prefix)
            number = self.y_axis_note.removeprefix(prefix).removesuffix(suffix)
            painter.drawText(rect.left(), 15, prefix)
            painter.setFont(numeric_font())
            number_width = QFontMetricsF(painter.font()).horizontalAdvance(number)
            painter.drawText(int(rect.left() + prefix_width), 15, number)
            painter.setFont(body_font())
            painter.drawText(int(rect.left() + prefix_width + number_width), 15, suffix)
        for fraction in (0.0, 0.5, 1.0):
            y = rect.bottom() - fraction * rect.height()
            value = ymin + fraction * (ymax - ymin)
            painter.setPen(QColor(Color.SLATE))
            painter.setFont(numeric_font())
            painter.drawText(0, int(y + 5), 46, 18, Qt.AlignmentFlag.AlignRight, f"{value:.3g}")
            painter.setPen(QPen(QColor(Color.RULE_SOFT), 1))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setPen(QColor(Color.SLATE))
        painter.setFont(body_font())
        painter.drawText(
            rect.left(),
            rect.bottom() + 20,
            rect.width(),
            18,
            Qt.AlignmentFlag.AlignCenter,
            "エポック",
        )
        painter.setFont(numeric_font())
        painter.drawText(rect.left(), rect.bottom() + 16, f"{xmin:g}")
        painter.drawText(rect.right() - 32, rect.bottom() + 16, f"{xmax:g}")

        def map_point(x: float, y: float) -> QPointF:
            px = rect.left() + (x - xmin) * rect.width() / (xmax - xmin)
            py = rect.bottom() - (y - ymin) * rect.height() / (ymax - ymin)
            return QPointF(px, py)

        end_labels: list[tuple[str, QColor, float]] = []
        painter.save()
        painter.setClipRect(rect)
        fold_label_added = False
        for name, series_color, xs, ys in self.series:
            color = QColor(series_color)
            if name.startswith("分割 "):
                color = QColor(Color.SLATE).lighter(165)
            coords = [map_point(x, y) for x, y in zip(xs, ys, strict=False)]
            width = 3 if name == "OOF AP" else 1 if name.startswith("分割 ") else 2
            painter.setPen(QPen(color, width))
            for left, right in pairwise(coords):
                painter.drawLine(left, right)
            if len(self.series) > 1 and coords:
                if name.startswith("分割 "):
                    if not fold_label_added:
                        end_labels.append((self._end_label(name), color, coords[-1].y()))
                        fold_label_added = True
                else:
                    end_labels.append((name, color, coords[-1].y()))
        if self.highlight:
            name, x, y = self.highlight
            entry = next((item for item in self.series if item[0] == name), None)
            if entry:
                painter.setBrush(QColor(entry[1]))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(map_point(x, y), 5, 5)
        if self.best:
            epoch, value = self.best
            point = map_point(epoch, value)
            painter.setPen(QPen(QColor(Color.SLATE), 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(point.x(), rect.top()), QPointF(point.x(), rect.bottom()))
            painter.setBrush(QColor(Color.SLIDE))
            painter.setPen(QPen(QColor(Color.GRAPHITE), 2))
            painter.drawEllipse(point, 5, 5)
        painter.restore()

        if end_labels:
            painter.setFont(body_font())
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
                    rect.right() + 6,
                    int(label_y + 4),
                    self.width() - rect.right() - 6,
                    16,
                    Qt.AlignmentFlag.AlignLeft,
                    name,
                )
        if self.best:
            epoch, value = self.best
            point = map_point(epoch, value)
            painter.setPen(QColor(Color.GRAPHITE))
            painter.setFont(body_font())
            label_y = point.y() - 7
            if label_y < rect.top() + 14:
                label_y = point.y() + 20
            prefix = "選択 epoch"
            painter.drawText(int(point.x() + 7), int(label_y), prefix)
            painter.setFont(numeric_font())
            prefix_width = QFontMetricsF(body_font()).horizontalAdvance(prefix)
            painter.drawText(
                int(point.x() + 7 + prefix_width), int(label_y), f" {epoch:g}　{value:.3f}"
            )
