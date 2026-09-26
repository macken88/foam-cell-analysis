"""確定ダイアログ用の非同期サムネイル一覧。"""

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from math import ceil
from queue import Empty, Queue
from threading import Lock

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QRect,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ....services.models import DataItem
from ...context import DEFAULT_CHANNEL
from ...settings import app_settings
from ...theme import Color, body_font
from ...widgets.image_convert import array_to_pixmap

THUMBNAIL_SIZE = (160, 120)
GRID_SIZE = (184, 150)
# 表示サイズの段階（名前, サムネイル画像の大きさ）。先頭が既定。
SIZE_LEVELS = (("小", (160, 120)), ("中", (240, 180)), ("大", (320, 240)))
_SIZE_SETTING_KEY = "thumbnails/sizeLevel"


def grid_size_for(thumbnail_size: tuple[int, int]) -> tuple[int, int]:
    """サムネイル画像の大きさから、枠・バッジ・帯を含むタイルの大きさを返す。"""
    return (thumbnail_size[0] + 24, thumbnail_size[1] + 30)


class ThumbnailSizePreference(QObject):
    """サムネイルの表示サイズ（アプリ全体で共通、設定に保存）。"""

    changed = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        level = int(app_settings().value(_SIZE_SETTING_KEY, 0))
        self.level = level if 0 <= level < len(SIZE_LEVELS) else 0

    @property
    def thumbnail_size(self) -> tuple[int, int]:
        """現在の段階のサムネイル画像の大きさ。"""
        return SIZE_LEVELS[self.level][1]

    def set_level(self, level: int) -> None:
        """段階を変え、保存して通知する。"""
        if level == self.level or not 0 <= level < len(SIZE_LEVELS):
            return
        self.level = level
        app_settings().setValue(_SIZE_SETTING_KEY, level)
        self.changed.emit(level)


_size_preference: ThumbnailSizePreference | None = None


def thumbnail_size_preference() -> ThumbnailSizePreference:
    """共有の表示サイズ設定を返す（初回に作る）。"""
    global _size_preference
    if _size_preference is None:
        _size_preference = ThumbnailSizePreference()
    return _size_preference


class ThumbnailSizeSelector(QWidget):
    """「表示サイズ [小 ▾]」の組。共有設定を変更し、変更に追従する。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.preference = thumbnail_size_preference()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(QLabel("表示サイズ"))
        self.combo = QComboBox()
        self.combo.addItems([name for name, _size in SIZE_LEVELS])
        self.combo.setCurrentIndex(self.preference.level)
        self.combo.currentIndexChanged.connect(self.preference.set_level)
        self.preference.changed.connect(self._preference_changed)
        layout.addWidget(self.combo)

    def _preference_changed(self, level: int) -> None:
        if self.combo.currentIndex() != level:
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(level)
            self.combo.blockSignals(False)


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
        self.classification_filter = "すべて"
        self.quality_filter = "すべて"
        self.view = None
        self._pending: set[str] = set()
        self._tasks: dict[str, Future] = {}
        self._results: Queue = Queue()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(30)
        self._poll_timer.timeout.connect(self._poll_results)
        self._wanted: set[str] = set()
        self.thumbnail_size = thumbnail_size_preference().thumbnail_size

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
        if (
            self.classification_filter != "すべて"
            and item.classification != self.classification_filter
        ):
            return False
        if self.quality_filter != "すべて" and item.quality != self.quality_filter:
            return False
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
        grid = grid_size_for(self.thumbnail_size)
        columns = max(1, self.view.viewport().width() // grid[0])
        first_line = self.view.verticalScrollBar().value() // grid[1]
        top = first_line * columns
        visible_lines = ceil(self.view.viewport().height() / grid[1])
        bottom = min(len(self.items) - 1, top + visible_lines * columns - 1)
        start = max(0, top - 2)
        end = min(len(self.items), bottom + 6)
        self._wanted = {item.item_id for item in self.items[start:end]}
        for item_id, future in tuple(self._tasks.items()):
            if item_id not in self._wanted and future.cancel():
                self._tasks.pop(item_id, None)
                self._pending.discard(item_id)
        for item in self.items[start:end]:
            size = self.thumbnail_size
            key = (item.item_id, *size)
            if _cache_get(key) is None and item.item_id not in self._pending:
                self._pending.add(item.item_id)
                future = _thumbnail_executor.submit(
                    self.backend.get_item_thumbnail, item.item_id, size
                )
                future.add_done_callback(
                    lambda result, key=item.item_id, size=size: self._queue_result(
                        key, result, size
                    )
                )
                self._tasks[item.item_id] = future
        if self._pending and not self._poll_timer.isActive():
            self._poll_timer.start()

    def _queue_result(self, item_id: str, future: Future, size: tuple[int, int]) -> None:
        """バックグラウンド結果をGUIスレッドへ渡す。"""
        try:
            image = future.result()
        except Exception:
            image = None
        self._results.put((item_id, future, image, size))

    def _poll_results(self) -> None:
        """ワーカ結果をGUIスレッドで受け取る。"""
        while True:
            try:
                item_id, future, image, size = self._results.get_nowait()
            except Empty:
                break
            if self._tasks.get(item_id) is not future:
                continue
            self._finished(item_id, image, size)
        if not self._pending:
            self._poll_timer.stop()

    def _finished(self, item_id: str, image, size: tuple[int, int]) -> None:
        self._pending.discard(item_id)
        self._tasks.pop(item_id, None)
        if item_id not in self._wanted:
            return
        if image is None:
            return
        # 要求したときの大きさで保存する（途中で表示サイズが変わっても混ざらない）
        _cache_put((item_id, *size), image)
        if size != self.thumbnail_size:
            return
        row = next((row for row, item in enumerate(self.items) if item.item_id == item_id), -1)
        if row >= 0:
            self.dataChanged.emit(
                self.index(row, 0), self.index(row, 0), [Qt.ItemDataRole.DecorationRole]
            )

    def image_for(self, item: DataItem):
        """キャッシュ済みの小画像を返す。"""
        return _cache_get((item.item_id, *self.thumbnail_size))

    def set_thumbnail_size(self, size: tuple[int, int]) -> None:
        """表示サイズを変え、表示範囲の画像を新しい大きさで要求し直す。"""
        if size == self.thumbnail_size:
            return
        self.thumbnail_size = size
        for item_id, future in tuple(self._tasks.items()):
            if future.cancel():
                self._tasks.pop(item_id, None)
        self._pending.clear()
        if self.items:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.items) - 1, 0))
        self.request_visible()


class FinalizeThumbnailView(QListView):
    """固定サイズのタイル表示を持つビュー。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setWrapping(True)
        self.size_preference = thumbnail_size_preference()
        self.setGridSize(QSize(*grid_size_for(self.size_preference.thumbnail_size)))
        self.setUniformItemSizes(True)
        self.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.size_preference.changed.connect(self._size_changed)

    def _size_changed(self, _level: int) -> None:
        """共有の表示サイズの変更をタイルと画像の大きさへ反映する。"""
        size = self.size_preference.thumbnail_size
        self.setGridSize(QSize(*grid_size_for(size)))
        model = self.model()
        if isinstance(model, FinalizeThumbnailModel):
            model.set_thumbnail_size(size)
        self.doItemsLayout()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        model = self.model()
        if isinstance(model, FinalizeThumbnailModel):
            QTimer.singleShot(0, model.request_visible)


class FinalizeThumbnailDelegate(QStyledItemDelegate):
    """用途、分類、品質、変更状態をサムネイルに重ねる。"""

    def sizeHint(self, option, index) -> QSize:
        model = index.model()
        size = getattr(model, "thumbnail_size", THUMBNAIL_SIZE)
        return QSize(*grid_size_for(size))

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
        image = backend.get_item_image("all", item.item_id, DEFAULT_CHANNEL)
        self.image_label.setPixmap(array_to_pixmap(image))
        layout.addWidget(self.image_label, 1)
        purpose = "学習" if item.usage == "train" else "検証"
        layout.addWidget(QLabel(f"{item.item_id}　{item.source_filename}　用途: {purpose}"))


class DatasetVersionThumbnailWindow(QDialog):
    """確定済み版の画像内容を読み取り専用で確認する。"""

    def __init__(self, parent, backend, version, linked_validation=None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(f"{version.version} のサムネイル")
        self.resize(1040, 760)
        self.backend = backend
        self.version = version
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"{version.version}　{version.created_at:%Y-%m-%d %H:%M}　"
                f"{version.n_images} 件　{version.comment or 'コメントなし'}"
            )
        )
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.models = {}
        self.views = {}
        versions = [(version, version.version)]
        if version.purpose == "train" and linked_validation:
            versions.append((linked_validation, f"基準検証用 {linked_validation.version}"))
        for record, title in versions:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            filters = QHBoxLayout()
            classification = QComboBox()
            classification.addItems(["すべて", "分類A", "分類B", "分類C"])
            quality = QComboBox()
            quality.addItems(["すべて", "良", "可", "不良", "未設定"])
            count = QLabel()
            filters.addWidget(QLabel("画像分類"))
            filters.addWidget(classification)
            filters.addSpacing(14)
            filters.addWidget(QLabel("品質"))
            filters.addWidget(quality)
            filters.addSpacing(14)
            filters.addWidget(ThumbnailSizeSelector())
            filters.addStretch(1)
            filters.addWidget(count)
            page_layout.addLayout(filters)
            view = FinalizeThumbnailView()
            model = FinalizeThumbnailModel(backend, self)
            model.filter_name = "all"
            items = backend.get_dataset_version_items(record.version)
            model.set_items(items, {})
            view.setModel(model)
            view.setItemDelegate(FinalizeThumbnailDelegate(view))
            model.attach_view(view)
            page_layout.addWidget(view, 1)
            classification.currentTextChanged.connect(
                lambda value, m=model, source=items, q=quality, label=count: self._filter(
                    m, source, value, q.currentText(), label
                )
            )
            quality.currentTextChanged.connect(
                lambda value, m=model, source=items, c=classification, label=count: self._filter(
                    m, source, c.currentText(), value, label
                )
            )
            view.clicked.connect(lambda index, m=model: self._preview(m, index))
            self.models[record.version] = model
            self.views[record.version] = view
            self.tabs.addTab(page, title)
            self._update_count(model, count, len(items))

    def _filter(self, model, items, classification, quality, label) -> None:
        """分類と品質で一覧を絞り込む。"""
        model.classification_filter = classification
        model.quality_filter = "未設定" if quality == "未設定" else quality
        if quality == "未設定":
            model.beginResetModel()
            model.items = [
                item
                for item in items
                if (classification == "すべて" or item.classification == classification)
                and not item.quality
            ]
            model.endResetModel()
        else:
            model.set_items(items, {})
        self._update_count(model, label, len(items))
        QTimer.singleShot(0, model.request_visible)

    @staticmethod
    def _update_count(model, label, total) -> None:
        """絞り込み後の件数を表示する。"""
        label.setText(f"{model.rowCount()} / {total} 件")

    def _preview(self, model, index) -> None:
        """選択画像を拡大する。"""
        item = index.data(Qt.ItemDataRole.UserRole)
        if isinstance(item, DataItem):
            ThumbnailPreviewDialog(self, self.backend, item).exec()
