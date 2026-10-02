"""ズーム・パン可能な画像表示部品。"""

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QLabel

from ..theme import Color, set_style
from .image_convert import DisplayMode

__all__ = ["DisplayMode", "ImageView", "ViewSynchronizer"]


class ImageView(QGraphicsView):
    """ホイールズーム、ドラッグパンに対応した画像ビュー。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._scale = 1.0
        self._fit_on_resize = False
        self._fitting = False
        self._syncing = False
        self._drag_position = None
        self.setBackgroundBrush(QBrush(QColor(Color.IMAGE_BG)))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.set_scrollbars_visible(True)
        self.placeholder = QLabel("画像なし", self.viewport())
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_style(self.placeholder, role="imagePlaceholder")
        self.top_left_label = QLabel("", self.viewport())
        self.bottom_right_label = QLabel("", self.viewport())
        for label in (self.top_left_label, self.bottom_right_label):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            set_style(label, role="imageOverlayLabel")
            label.hide()

    @property
    def zoom(self) -> float:
        return self._scale

    def set_image(
        self, pixmap: QPixmap | None, *, fit: bool = True, reference: "ImageView | None" = None
    ) -> None:
        """Pixmap を設定する。fit=False は倍率と画素座標の表示中心を保つ。"""
        synchronizer = getattr(self, "_synchronizer", None)
        old_center = self.mapToScene(self.viewport().rect().center()) if self._pixmap_item else None
        old_zoom = self._scale
        if not fit and synchronizer is not None and synchronizer.last_center is not None:
            old_center = QPointF(synchronizer.last_center)
            old_zoom = synchronizer.last_zoom or old_zoom
        if reference is not None and reference._pixmap_item is not None:
            reference_sync = getattr(reference, "_synchronizer", None)
            old_center = (
                QPointF(reference_sync.last_center)
                if reference_sync is not None and reference_sync.last_center is not None
                else reference.mapToScene(reference.viewport().rect().center())
            )
            old_zoom = (
                reference_sync.last_zoom
                if reference_sync is not None and reference_sync.last_zoom
                else reference.zoom
            )
            self.resetTransform()
            self.scale(old_zoom, old_zoom)
            self._scale = old_zoom
        self.scene().clear()
        self._pixmap_item = None
        has_image = pixmap is not None and not pixmap.isNull()
        self.placeholder.setVisible(not has_image)
        if has_image:
            self._pixmap_item = self.scene().addPixmap(pixmap)
            self.scene().setSceneRect(self._pixmap_item.boundingRect())
            if fit:
                self._fit_on_resize = True
                self.fit_image()
                QTimer.singleShot(0, self._fit_if_pending)
            else:
                self._fit_on_resize = False
                rect = self.scene().sceneRect()
                center = old_center if old_center is not None else rect.center()
                self._center_on_point(center)
        else:
            self._fit_on_resize = False
            self._notify_sync(remember=False)
        self._schedule_sync()

    def set_overlay_labels(self, top_left: str = "", bottom_right: str = "") -> None:
        """画像左上と右下に重ねるラベルを設定する。"""
        for label, text in (
            (self.top_left_label, top_left),
            (self.bottom_right_label, bottom_right),
        ):
            label.setText(text)
            label.setVisible(bool(text))
            label.adjustSize()
        self._position_overlay_labels()

    def _position_overlay_labels(self) -> None:
        """画像枠の内側に小さなラベルを配置する。"""
        self.top_left_label.move(8, 8)
        self.bottom_right_label.move(
            max(8, self.viewport().width() - self.bottom_right_label.width() - 8),
            max(8, self.viewport().height() - self.bottom_right_label.height() - 8),
        )

    def set_scrollbars_visible(self, visible: bool) -> None:
        """スクロールバーを表示または非表示にする。"""
        policy = (
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
            if visible
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setHorizontalScrollBarPolicy(policy)
        self.setVerticalScrollBarPolicy(policy)
        self._schedule_sync()

    def fit_image(self) -> None:
        """画像全体を表示する。"""
        if self._pixmap_item and not self._fitting:
            self._fit_on_resize = True
            self._fitting = True
            try:
                self.resetTransform()
                self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
                self._scale = self.transform().m11()
                self._notify_sync()
            finally:
                self._fitting = False

    def _fit_if_pending(self) -> None:
        """表示後のレイアウトが確定してから全体表示を適用する。"""
        if self._fit_on_resize and self._pixmap_item:
            self.fit_image()

    def zoom_by(self, factor: float) -> None:
        """表示倍率を指定比率で変更する。"""
        self._fit_on_resize = False
        self.scale(factor, factor)
        self._scale *= factor
        self._notify_sync()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """カーソル位置を中心にズームする。"""
        self._fit_on_resize = False
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        self._scale *= factor
        self._notify_sync()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._fit_on_resize = False
            self._drag_position = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_position is not None:
            delta = event.position().toPoint() - self._drag_position
            self._drag_position = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._notify_sync()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_position = None
        self.unsetCursor()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.fit_image()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:
        preserve_center = bool(self._pixmap_item and not self._fit_on_resize)
        center = self.mapToScene(self.viewport().rect().center()) if preserve_center else None
        zoom = self._scale
        super().resizeEvent(event)
        self.placeholder.setGeometry(self.viewport().rect())
        self._position_overlay_labels()
        if self._fit_on_resize and self._pixmap_item:
            self.fit_image()
        elif center is not None:
            self._scale = zoom
            self._center_on_point(center)
        self._schedule_sync()

    def _center_on_point(self, point: QPointF) -> None:
        """指定した画素中心を表示枠中央に保つ。"""
        point = ViewSynchronizer.clamp_center(self, point)
        self.centerOn(point)
        actual = self.mapToScene(self.viewport().rect().center())
        scale = self.transform().m11()
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() + round((point.x() - actual.x()) * scale)
        )
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() + round((point.y() - actual.y()) * scale)
        )

    def _notify_sync(self, expected_synchronizer=None, *, remember: bool = True) -> None:
        if (
            expected_synchronizer is not None
            and getattr(self, "_synchronizer", None) is not expected_synchronizer
        ):
            return
        if not getattr(self, "_syncing", False) and hasattr(self, "_synchronizer"):
            self._synchronizer.sync_from(self, remember=remember)

    def _schedule_sync(self) -> None:
        """Qt のビューポート再配置後にビューを同期する。"""
        synchronizer = getattr(self, "_synchronizer", None)
        if synchronizer is not None:
            QTimer.singleShot(
                0,
                lambda expected=synchronizer: self._notify_sync(expected, remember=False),
            )


class ViewSynchronizer:
    """複数 ImageView の倍率と表示位置を同期する。"""

    def __init__(self, views: list[ImageView], *, last_state=None) -> None:
        self.views = views
        if last_state is not None:
            self.last_center, self.last_zoom = QPointF(last_state[0]), last_state[1]
        elif views and views[0]._pixmap_item is not None:
            self.last_center = views[0].mapToScene(views[0].viewport().rect().center())
            self.last_zoom = views[0].zoom
        else:
            self.last_center, self.last_zoom = None, None
        for view in views:
            view._synchronizer = self

    @staticmethod
    def _center_on_scene_point(view: ImageView, point: QPointF) -> None:
        """整数スクロールバーの丸めを補正してシーン座標を中央に置く。"""
        view.centerOn(point)
        center = view.mapToScene(view.viewport().rect().center())
        scale = view.transform().m11()
        horizontal = view.horizontalScrollBar()
        vertical = view.verticalScrollBar()
        horizontal.setValue(horizontal.value() + round((point.x() - center.x()) * scale))
        vertical.setValue(vertical.value() + round((point.y() - center.y()) * scale))

    @staticmethod
    def clamp_center(view: ImageView, point: QPointF) -> QPointF:
        """画像外へ出る中心だけを、表示枠の半分を考慮して画像内へ戻す。"""
        rect = view.scene().sceneRect()
        scale = max(view.transform().m11(), 1e-6)
        half_width = view.viewport().width() / (2 * scale)
        half_height = view.viewport().height() / (2 * scale)
        minimum_x = rect.left() + half_width
        maximum_x = rect.right() - half_width
        minimum_y = rect.top() + half_height
        maximum_y = rect.bottom() - half_height
        x = (
            rect.center().x()
            if minimum_x > maximum_x
            else min(max(point.x(), minimum_x), maximum_x)
        )
        y = (
            rect.center().y()
            if minimum_y > maximum_y
            else min(max(point.y(), minimum_y), maximum_y)
        )
        return QPointF(x, y)

    def sync_from(self, source: ImageView, *, remember: bool = True) -> None:
        """基準ビューの表示中心シーン座標と倍率を他ビューへ反映する。"""
        if source not in self.views:
            return
        scene_center = source.mapToScene(source.viewport().rect().center())
        if remember and source._pixmap_item is not None:
            self.last_center = QPointF(scene_center)
            self.last_zoom = source.zoom
        for view in self.views:
            if view is source:
                continue
            view._syncing = True
            current = view.transform().m11()
            if current and source.zoom:
                view.scale(source.zoom / current, source.zoom / current)
            view._scale = source.zoom
            self._center_on_scene_point(view, self.clamp_center(view, scene_center))
            view._syncing = False

    def fit_all(self) -> None:
        """先頭ビューを全体表示し、その倍率と中心を全ビューへ適用する。"""
        if not self.views:
            return
        first = self.views[0]
        first.fit_image()
        self.sync_from(first)
        for view in self.views:
            view._schedule_sync()
