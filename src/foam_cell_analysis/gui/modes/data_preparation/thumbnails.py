"""遅延画像生成とフォルダ見出しを備えたサムネイル一覧。"""

from collections import OrderedDict
from dataclasses import dataclass

from PySide6.QtCore import QAbstractListModel, QRect, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QListView, QStyle, QStyledItemDelegate

from ....services.models import DataItem
from ...theme import Color, body_font
from ...widgets.image_convert import array_to_pixmap

USAGE_BORDER = {
    "train": Color.TRAIN,
    "val": Color.VAL,
    "unassigned": Color.IDLE,
    "excluded": Color.IDLE,
}
USAGE_LABEL = {
    "train": "学習",
    "val": "検証",
    "unassigned": "未振り分け",
    "excluded": "不採用",
}


@dataclass(frozen=True)
class FolderHeader:
    """一覧に挿入するフォルダ見出し。"""

    source_folder: str


class ThumbnailModel(QAbstractListModel):
    """項目データを渡し、要求されたサムネイルだけ生成する。"""

    def __init__(self, backend, parent=None) -> None:
        super().__init__(parent)
        self.backend = backend
        self.items: list[DataItem] = []
        self.rows: list[DataItem | FolderHeader] = []
        self.errors: dict[str, str] = {}
        self.cache: OrderedDict[tuple[str, int], QPixmap] = OrderedDict()
        self.size = 160

    def set_items(self, items: list[DataItem], errors: dict[str, str] | None = None) -> None:
        """フォルダ見出しを挟み、項目をモデルへ設定する。"""
        self.beginResetModel()
        self.items = list(items)
        self.errors = errors or {}
        self.rows = []
        prior_folder = None
        for item in self.items:
            if item.source_folder != prior_folder:
                self.rows.append(FolderHeader(item.source_folder))
                prior_folder = item.source_folder
            self.rows.append(item)
        self.endResetModel()

    def rowCount(self, parent=None) -> int:
        if parent is None:
            return len(self.rows)
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return row
        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(row, FolderHeader):
                return row.source_folder
            return f"{row.item_id}  {row.source_filename}"
        return None

    def flags(self, index):
        if index.isValid() and isinstance(self.rows[index.row()], FolderHeader):
            return Qt.ItemFlag.NoItemFlags
        return super().flags(index)

    def pixmap_for(self, item: DataItem) -> QPixmap:
        """描画された項目だけ生成し、LRU キャッシュから返す。"""
        key = (item.item_id, self.size)
        if key not in self.cache:
            image = self.backend.get_item_image(
                "all", item.item_id, item.channels[0] if item.channels else "A"
            )
            height = round(self.size * 0.75)
            pixmap = array_to_pixmap(image).scaled(
                QSize(self.size, height),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self.cache[key] = pixmap
            if len(self.cache) > 320:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


class ThumbnailDelegate(QStyledItemDelegate):
    """モックに合わせて見出しと用途別サムネイルを描く。"""

    def sizeHint(self, option, index) -> QSize:
        row = index.data(Qt.ItemDataRole.UserRole)
        if isinstance(row, FolderHeader):
            view = option.widget
            width = view.viewport().width() if isinstance(view, QListView) else 320
            return QSize(max(1, width - 12), 28)
        model = index.model()
        size = model.size if isinstance(model, ThumbnailModel) else 160
        return QSize(size + 12, round(size * 0.75) + 12)

    def paint(self, painter: QPainter, option, index) -> None:
        row = index.data(Qt.ItemDataRole.UserRole)
        painter.save()
        if isinstance(row, FolderHeader):
            self._paint_folder_header(painter, option.rect, row)
        elif isinstance(row, DataItem):
            self._paint_thumbnail(painter, option, index, row)
        painter.restore()

    @staticmethod
    def _paint_folder_header(painter: QPainter, rect: QRect, header: FolderHeader) -> None:
        """横幅いっぱいの見出しと下線を描く。"""
        painter.setFont(body_font(9))
        painter.setPen(QColor(Color.SLATE))
        painter.drawText(
            rect.adjusted(2, 0, -2, -2), Qt.AlignmentFlag.AlignVCenter, header.source_folder
        )
        painter.setPen(QPen(QColor(Color.RULE), 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())

    def _paint_thumbnail(self, painter: QPainter, option, index, item: DataItem) -> None:
        """画像、用途枠、バッジ、下帯を一枚のサムネイルへ重ねる。"""
        model = index.model()
        if not isinstance(model, ThumbnailModel):
            return
        rect = option.rect.adjusted(4, 3, -4, -3)
        issue = model.errors.get(item.item_id)
        frame_color = Color.ERROR if issue else USAGE_BORDER[item.usage]
        painter.save()
        if item.usage == "excluded":
            painter.setOpacity(0.5)

        painter.fillRect(rect, QColor(Color.IMAGE_BG))
        image_rect = rect.adjusted(3, 3, -3, -3)
        pixmap = model.pixmap_for(item)
        x = image_rect.x() + (image_rect.width() - pixmap.width()) // 2
        y = image_rect.y() + (image_rect.height() - pixmap.height()) // 2
        painter.drawPixmap(x, y, pixmap)

        frame = QPen(QColor(frame_color), 3)
        painter.setPen(frame)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect.adjusted(1, 1, -1, -1))

        classification = item.classification or "未設定"
        if classification.startswith("分類"):
            classification = classification.removeprefix("分類")
        if issue:
            badge = "⚠ マスクなし" if "マスク" in issue else f"⚠ {issue}"
            badge_color = Color.ERROR
            badge_text = Color.WHITE
        else:
            badge = f"{classification}・{item.quality or '未設定'}"
            badge_color = Color.SLIDE
            badge_text = Color.GRAPHITE
        self._paint_badge(painter, rect, badge, badge_color, badge_text)

        band = QRect(rect.left() + 3, rect.bottom() - 21, rect.width() - 6, 18)
        band_color = QColor(Color.GRAPHITE)
        band_color.setAlphaF(0.72)
        painter.fillRect(band, band_color)
        painter.setFont(body_font(9))
        painter.setPen(QColor(Color.WHITE))
        painter.drawText(
            band.adjusted(5, 0, -5, 0), Qt.AlignmentFlag.AlignVCenter, USAGE_LABEL[item.usage]
        )

        if option.state & QStyle.StateFlag.State_Selected:
            painter.setOpacity(1.0)
            selection = QPen(QColor(Color.GRAPHITE), 2)
            painter.setPen(selection)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(option.rect.adjusted(1, 1, -1, -1))
        painter.restore()

    @staticmethod
    def _paint_badge(
        painter: QPainter, rect: QRect, text: str, background: str, foreground: str
    ) -> None:
        """右上に小さな分類・品質タグを描く。"""
        font = body_font(9)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = min(rect.width() - 10, metrics.horizontalAdvance(text) + 10)
        badge = QRect(rect.right() - width - 3, rect.top() + 3, width, metrics.height() + 2)
        color = QColor(background)
        if background == Color.SLIDE:
            color.setAlphaF(0.92)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(badge, 2, 2)
        painter.setPen(QColor(foreground))
        painter.drawText(badge.adjusted(5, 0, -5, 0), Qt.AlignmentFlag.AlignVCenter, text)


class ThumbnailListView(QListView):
    """Ctrl+ホイールでサムネイルを拡大・縮小する一覧。"""

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            model = self.model()
            if isinstance(model, ThumbnailModel):
                step = 16 if event.angleDelta().y() > 0 else -16
                model.size = max(96, min(320, model.size + step))
                model.cache.clear()
                model.layoutChanged.emit()
                self.doItemsLayout()
                self.viewport().update()
                event.accept()
                return
        super().wheelEvent(event)


def configure_thumbnail_view(view: QListView) -> None:
    """幅に応じて列を増やし、項目と見出しの寸法はデリゲートへ委ねる。"""
    view.setViewMode(QListView.ViewMode.IconMode)
    view.setFlow(QListView.Flow.LeftToRight)
    view.setWrapping(True)
    view.setResizeMode(QListView.ResizeMode.Adjust)
    # 見出しだけ横幅いっぱいにするため、行ごとの sizeHint を使う。
    view.setUniformItemSizes(False)
    view.setGridSize(QSize())
    view.setMovement(QListView.Movement.Static)
    view.setSpacing(6)
