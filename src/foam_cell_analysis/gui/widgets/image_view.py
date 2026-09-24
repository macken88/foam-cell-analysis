"""ズーム・パン可能な画像表示部品。"""

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QLabel

from .image_convert import DisplayMode

__all__ = ["DisplayMode", "ImageView", "ViewSynchronizer"]


class ImageView(QGraphicsView):
    """ホイールズーム、ドラッグパンに対応した画像ビュー。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._scale = 1.0
        self._syncing = False
        self._drag_position = None
        self.setBackgroundBrush(Qt.GlobalColor.lightGray)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.set_scrollbars_visible(True)
        self.placeholder = QLabel("画像なし", self.viewport())
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #666; background: #ddd;")

    @property
    def zoom(self) -> float:
        return self._scale

    def set_image(self, pixmap: QPixmap | None) -> None:
        """Pixmap を設定し、無い場合はプレースホルダを表示する。"""
        self.scene().clear()
        self._pixmap_item = None
        has_image = pixmap is not None and not pixmap.isNull()
        self.placeholder.setVisible(not has_image)
        if has_image:
            self._pixmap_item = self.scene().addPixmap(pixmap)
            self.scene().setSceneRect(self._pixmap_item.boundingRect())
            self.fit_image()
        else:
            self._notify_sync()
        self._schedule_sync()

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
        if self._pixmap_item:
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._scale = self.transform().m11()
            self._notify_sync()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """カーソル位置を中心にズームする。"""
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        self._scale *= factor
        self._notify_sync()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
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
        super().resizeEvent(event)
        self.placeholder.setGeometry(self.viewport().rect())
        self._notify_sync()
        self._schedule_sync()

    def _notify_sync(self) -> None:
        if not getattr(self, "_syncing", False) and hasattr(self, "_synchronizer"):
            self._synchronizer.sync_from(self)

    def _schedule_sync(self) -> None:
        """Qt のビューポート再配置後にビューを同期する。"""
        if hasattr(self, "_synchronizer"):
            QTimer.singleShot(0, self._notify_sync)


class ViewSynchronizer:
    """複数 ImageView の倍率と表示位置を同期する。"""

    def __init__(self, views: list[ImageView]) -> None:
        self.views = views
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

    def sync_from(self, source: ImageView) -> None:
        """基準ビューの表示中心シーン座標と倍率を他ビューへ反映する。"""
        scene_center = source.mapToScene(source.viewport().rect().center())
        for view in self.views:
            if view is source:
                continue
            view._syncing = True
            current = view.transform().m11()
            if current and source.zoom:
                view.scale(source.zoom / current, source.zoom / current)
            view._scale = source.zoom
            self._center_on_scene_point(view, scene_center)
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
