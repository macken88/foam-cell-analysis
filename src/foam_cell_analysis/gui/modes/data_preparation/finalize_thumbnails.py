"""確定ダイアログ用の非同期サムネイル一覧。"""

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from math import ceil
from queue import Empty, Queue
from threading import Lock

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QRect,
    QSize,
    Qt,
    QTimer,
)
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
)

from ....services.models import DataItem
from ...theme import Color, body_font
from ...widgets.image_convert import array_to_pixmap

THUMBNAIL_SIZE = (160, 120)
GRID_SIZE = (184, 150)
_thumbnail_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="foam-thumbnail")
_CACHE_LIMIT = 240
_thumbnail_cache: OrderedDict[tuple[str, int, int], object] = OrderedDict()
_cache_lock = Lock()


def _cache_get(key):
    with _cache_lock:
        image = _thumbnail_cache.get(key)
        if image is not None:
            _thumbnail_cache.move_to_end(key)
        return image


def _cache_put(key, image) -> None:
    with _cache_lock:
        _thumbnail_cache[key] = image
        _thumbnail_cache.move_to_end(key)
        while len(_thumbnail_cache) > _CACHE_LIMIT:
            _thumbnail_cache.popitem(last=False)


class FinalizeThumbnailModel(QAbstractListModel):
    """可視範囲に近い画像だけを要求する一覧モデル。"""

    def __init__(self, backend, parent=None) -> None:
        super().__init__(parent)
        self.backend = backend
        self.items: list[DataItem] = []
        self.errors: dict[str, str] = {}
        self.filter_name = "changed"
        self.view = None
        self._pending: set[str] = set()
        self._tasks: dict[str, Future] = {}
        self._results: Queue = Queue()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(30)
        self._poll_timer.timeout.connect(self._poll_results)
        self._wanted: set[str] = set()

    def set_items(self, items: list[DataItem], errors: dict[str, str]) -> None:
        """絞り込みを適用した項目一覧へ切り替える。"""
        self.beginResetModel()
        self.items = [item for item in items if self._matches(item, errors)]
        self.errors = errors
        self._pending.intersection_update(item.item_id for item in self.items)
        self._wanted.clear()
        self.endResetModel()
        QTimer.singleShot(0, self.request_visible)

    def _matches(self, item: DataItem, errors: dict[str, str]) -> bool:
        if self.filter_name == "all":
            return True
        if self.filter_name == "changed":
            return item.change in {"added", "changed"}
        if self.filter_name == "errors":
            return item.item_id in errors
        return True

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.items)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.items):
            return None
        item = self.items[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return item
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{item.item_id}　{item.source_filename}"
        return None

    def attach_view(self, view) -> None:
        """表示範囲の監視先を登録する。"""
        self.view = view
        view.verticalScrollBar().valueChanged.connect(self.request_visible)
        QTimer.singleShot(0, self.request_visible)

    def request_visible(self, *_args) -> None:
        """表示中と数行先のサムネイルだけを生成要求する。"""
        if self.view is None or not self.items or not self.view.isVisible():
            self._wanted.clear()
            return
        columns = max(1, self.view.viewport().width() // GRID_SIZE[0])
        first_line = self.view.verticalScrollBar().value() // GRID_SIZE[1]
        top = first_line * columns
        visible_lines = ceil(self.view.viewport().height() / GRID_SIZE[1])
        bottom = min(len(self.items) - 1, top + visible_lines * columns - 1)
        start = max(0, top - 2)
        end = min(len(self.items), bottom + 6)
        self._wanted = {item.item_id for item in self.items[start:end]}
        for item_id, future in tuple(self._tasks.items()):
            if item_id not in self._wanted and future.cancel():
                self._tasks.pop(item_id, None)
                self._pending.discard(item_id)
        for item in self.items[start:end]:
            key = (item.item_id, *THUMBNAIL_SIZE)
            if _cache_get(key) is None and item.item_id not in self._pending:
                self._pending.add(item.item_id)
                future = _thumbnail_executor.submit(
                    self.backend.get_item_thumbnail, item.item_id, THUMBNAIL_SIZE
                )
                future.add_done_callback(
                    lambda result, key=item.item_id: self._queue_result(key, result)
                )
                self._tasks[item.item_id] = future
        if self._pending and not self._poll_timer.isActive():
            self._poll_timer.start()

    def _queue_result(self, item_id: str, future: Future) -> None:
        """バックグラウンド結果をGUIスレッドへ渡す。"""
        try:
            image = future.result()
        except Exception:
            image = None
        self._results.put((item_id, future, image))

    def _poll_results(self) -> None:
        """ワーカ結果をGUIスレッドで受け取る。"""
        while True:
            try:
                item_id, future, image = self._results.get_nowait()
            except Empty:
                break
            if self._tasks.get(item_id) is not future:
                continue
            self._finished(item_id, image)
        if not self._pending:
            self._poll_timer.stop()

    def _finished(self, item_id: str, image) -> None:
        self._pending.discard(item_id)
        self._tasks.pop(item_id, None)
        if item_id not in self._wanted:
            return
        if image is None:
            return
        _cache_put((item_id, *THUMBNAIL_SIZE), image)
        row = next((row for row, item in enumerate(self.items) if item.item_id == item_id), -1)
        if row >= 0:
            self.dataChanged.emit(
                self.index(row, 0), self.index(row, 0), [Qt.ItemDataRole.DecorationRole]
            )

    def image_for(self, item: DataItem):
        """キャッシュ済みの小画像を返す。"""
        return _cache_get((item.item_id, *THUMBNAIL_SIZE))


class FinalizeThumbnailView(QListView):
    """固定サイズのタイル表示を持つビュー。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setWrapping(True)
        self.setGridSize(QSize(*GRID_SIZE))
        self.setUniformItemSizes(True)
        self.setSelectionMode(QListView.SelectionMode.SingleSelection)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        model = self.model()
        if isinstance(model, FinalizeThumbnailModel):
            QTimer.singleShot(0, model.request_visible)


class FinalizeThumbnailDelegate(QStyledItemDelegate):
    """用途、分類、品質、変更状態をサムネイルに重ねる。"""

    def sizeHint(self, option, index) -> QSize:
        return QSize(*GRID_SIZE)

    def paint(self, painter: QPainter, option, index) -> None:
        item = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(item, DataItem):
            return
        model = index.model()
        rect = option.rect.adjusted(4, 4, -4, -4)
        painter.save()
        painter.fillRect(rect, QColor(Color.IMAGE_BG))
        image = model.image_for(item)
        image_rect = rect.adjusted(3, 3, -3, -23)
        if image is None:
            painter.fillRect(image_rect, QColor(Color.SLIDE))
        else:
            qimage = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
            painter.drawImage(image_rect, qimage)
        edge = (
            Color.ERROR
            if item.item_id in model.errors
            else (Color.TRAIN if item.usage == "train" else Color.VAL)
        )
        painter.setPen(QPen(QColor(edge), 3))
        painter.drawRect(rect.adjusted(1, 1, -1, -1))
        issue = item.item_id in model.errors
        badge = (
            "⚠ 要確認"
            if issue
            else f"{item.classification or '未設定'}・{item.quality or '未設定'}"
        )
        painter.setFont(body_font(8))
        badge_background = Color.ERROR if issue else Color.SLIDE
        badge_foreground = Color.WHITE if issue else Color.GRAPHITE
        painter.setPen(QColor(badge_foreground))
        metrics = painter.fontMetrics()
        badge_width = min(rect.width() - 10, metrics.horizontalAdvance(badge) + 10)
        badge_rect = QRect(
            rect.right() - badge_width - 3, rect.top() + 3, badge_width, metrics.height() + 2
        )
        painter.fillRect(badge_rect, QColor(badge_background))
        painter.drawText(badge_rect.adjusted(5, 0, -5, 0), Qt.AlignmentFlag.AlignVCenter, badge)
        band = rect.adjusted(2, rect.height() - 22, -2, -2)
        painter.fillRect(band, QColor(Color.GRAPHITE))
        painter.setPen(QColor(Color.WHITE))
        purpose = "学習" if item.usage == "train" else "検証"
        change = {"added": "追加", "changed": "変更"}.get(item.change, "変更なし")
        painter.drawText(band, Qt.AlignmentFlag.AlignCenter, f"{purpose} ・ {change}")
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(QPen(QColor(Color.GRAPHITE), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(option.rect.adjusted(1, 1, -1, -1))
        painter.restore()


class ThumbnailPreviewDialog(QDialog):
    """選択画像の簡易拡大プレビュー。"""

    def __init__(self, parent, backend, item: DataItem) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{item.item_id} のプレビュー")
        self.setMinimumSize(560, 440)
        layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image = backend.get_item_image(
            "all", item.item_id, item.channels[0] if item.channels else "A"
        )
        self.image_label.setPixmap(array_to_pixmap(image))
        layout.addWidget(self.image_label, 1)
        purpose = "学習" if item.usage == "train" else "検証"
        layout.addWidget(QLabel(f"{item.item_id}　{item.source_filename}　用途: {purpose}"))
