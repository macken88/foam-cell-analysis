"""画像表示の共通処理。"""

import numpy as np
from PySide6.QtGui import QColor, QPixmap

from foam_cell_analysis.gui.widgets.chart import LineChart
from foam_cell_analysis.gui.widgets.image_convert import DisplayMode, render, to_binary_separated
from foam_cell_analysis.gui.widgets.image_view import ImageView, ViewSynchronizer


def test_to_binary_separated_removes_8_neighbor_contact():
    labels = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
    binary = to_binary_separated(labels)
    assert binary.tolist() == [[1, 0, 0, 1], [1, 0, 0, 1]]


def test_views_are_synchronized(qtbot):
    first, second = ImageView(), ImageView()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    ViewSynchronizer([first, second])
    first.scale(2, 2)
    first._scale = 2
    first._notify_sync()
    assert second.zoom == 2


def test_image_view_overlay_labels_are_set_and_positioned(qtbot):
    view = ImageView()
    qtbot.addWidget(view)
    view.resize(300, 200)
    view.show()

    view.set_overlay_labels("RC-001 Mask R-CNN", "検出 11 個")

    assert view.top_left_label.text() == "RC-001 Mask R-CNN"
    assert view.bottom_right_label.text() == "検出 11 個"
    assert view.top_left_label.isVisible()
    assert view.bottom_right_label.isVisible()
    assert view.top_left_label.pos().x() < view.bottom_right_label.pos().x()


def test_render_supports_all_display_modes():
    image = np.full((3, 4), 80, dtype=np.uint8)
    labels = np.array([[0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 2, 0]], dtype=np.int32)
    original = render(image, labels, DisplayMode.IMAGE)
    overlay = render(image, labels, DisplayMode.OVERLAY)
    instance = render(image, labels, DisplayMode.INSTANCE_LABEL)
    binary = render(image, labels, DisplayMode.BINARY)
    assert original.shape == overlay.shape == instance.shape == binary.shape == (3, 4, 3)
    assert not np.array_equal(instance[0, 1], instance[1, 2])
    assert set(np.unique(binary)) <= {0, 255}


def test_image_view_uses_mouse_anchor_and_scene_center_sync(qtbot):
    first = ImageView()
    second = ImageView()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.resize(240, 180)
    second.resize(320, 220)
    first.show()
    second.show()
    synchronizer = ViewSynchronizer([first, second])
    first_pixmap = QPixmap(500, 400)
    first_pixmap.fill(QColor("white"))
    second_pixmap = QPixmap(500, 400)
    second_pixmap.fill(QColor("white"))
    first.set_image(first_pixmap)
    second.set_image(second_pixmap)
    first.scale(2.0, 2.0)
    first._scale *= 2.0
    synchronizer.sync_from(first)
    assert first.transformationAnchor() == first.ViewportAnchor.AnchorUnderMouse
    assert second.zoom == first.zoom
    center_a = first.mapToScene(first.viewport().rect().center())
    center_b = second.mapToScene(second.viewport().rect().center())
    assert abs(center_a.x() - center_b.x()) < 2
    assert abs(center_a.y() - center_b.y()) < 2


def test_chart_accepts_sparse_epoch_x_values_and_highlight(qtbot):
    chart = LineChart()
    qtbot.addWidget(chart)
    chart.set_series([("mAP", QColor("#2255aa"), [5, 10, 20], [0.7, 0.8, 0.85])])
    chart.set_highlight("mAP", 20, 0.85)
    assert chart.series[0][2] == [5, 10, 20]
    assert chart.highlight == ("mAP", 20, 0.85)
