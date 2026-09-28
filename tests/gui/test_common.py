"""共通スタイル、ラベル、テーブル、画像ビューの検証。"""

from datetime import datetime

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QPushButton,
    QTableWidget,
)

from foam_cell_analysis.app import apply_style, install_translations
from foam_cell_analysis.gui.labels import (
    candidate_status_label,
    classification_label,
    config_key_label,
    experiment_status_label,
    format_datetime,
    format_score,
    model_type_label,
    quality_filter_label,
)
from foam_cell_analysis.gui.widgets.image_view import ImageView, ViewSynchronizer
from foam_cell_analysis.gui.widgets.table import mark_primary, setup_table


def test_translation_installation_is_safe_without_translation_file(qapp):
    install_translations(qapp)


def test_apply_style_sets_shared_control_styles(qapp):
    apply_style(qapp)
    style = qapp.styleSheet()
    assert 'QPushButton[primary="true"]' in style
    assert "QListWidget#mainSidebar::item:selected" in style
    assert "QTableView QHeaderView::section" in style


@pytest.mark.parametrize(
    ("function", "value", "expected"),
    [
        (model_type_label, "mask_rcnn", "Mask R-CNN"),
        (model_type_label, "cellpose", "Cellpose"),
        (classification_label, "all", "全分類"),
        (quality_filter_label, "all", "すべて"),
        (quality_filter_label, "good", "良のみ"),
        (quality_filter_label, "good_and_acceptable", "良・可"),
        (experiment_status_label, "draft", "下書き"),
        (experiment_status_label, "running", "実行中"),
        (experiment_status_label, "completed", "完了"),
        (experiment_status_label, "failed", "失敗"),
        (experiment_status_label, "stopped", "中断"),
        (candidate_status_label, "candidate", "候補"),
        (candidate_status_label, "evaluating", "評価中"),
        (candidate_status_label, "released", "リリース済み"),
        (candidate_status_label, "rejected", "非採用"),
    ],
)
def test_label_translation_functions(function, value, expected):
    assert function(value) == expected


def test_label_formatting_and_config_key_mapping():
    assert format_score(0.91234) == "0.912"
    assert format_score(None) == "—"
    local_time = (
        datetime.now()
        .astimezone()
        .replace(year=2026, month=9, day=25, hour=10, minute=3, second=0, microsecond=0)
    )
    assert format_datetime(local_time) == "2026-09-25 10:03"
    assert config_key_label("training.learning_rate") == "学習率"
    assert config_key_label("model.rpn.fg_iou_thresh") == "RPN前景IoU閾値"
    assert config_key_label("unregistered.key") == "unregistered.key"


def test_setup_table_and_mark_primary(qapp):
    table = QTableWidget(4, 3)
    setup_table(table, selection_mode=QAbstractItemView.SelectionMode.ExtendedSelection)
    header = table.horizontalHeader()
    assert table.verticalHeader().isHidden()
    assert not table.alternatingRowColors()
    assert table.selectionBehavior() == QAbstractItemView.SelectionBehavior.SelectRows
    assert table.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert not table.wordWrap()
    assert not table.showGrid()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
    assert not header.stretchLastSection()
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Interactive

    second_table = QTableWidget(2, 2)
    setup_table(second_table, stretch_column=0)
    assert (
        second_table.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
    )
    button = QPushButton("保存")
    mark_primary(button)
    assert button.property("primary") is True


def test_image_view_scrollbar_visibility_and_fit_all(qtbot):
    first = ImageView()
    second = ImageView()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.resize(260, 190)
    second.resize(360, 250)
    first.show()
    second.show()
    qtbot.wait(10)
    synchronizer = ViewSynchronizer([first, second])
    pixmaps = [QPixmap(640, 480), QPixmap(640, 480)]
    for pixmap in pixmaps:
        pixmap.fill(QColor("white"))
    first.set_image(pixmaps[0])
    second.set_image(pixmaps[1])
    qtbot.wait(10)
    _assert_same_view(first, second)

    first.set_scrollbars_visible(False)
    assert first.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert first.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    first.set_scrollbars_visible(True)
    assert first.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded

    first.scale(1.4, 1.4)
    first._scale *= 1.4
    synchronizer.fit_all()
    qtbot.wait(10)
    _assert_same_view(first, second)


def _assert_same_view(first: ImageView, second: ImageView) -> None:
    """倍率と表示中心シーン座標が一致することを確認する。"""
    assert first.zoom == pytest.approx(second.zoom)
    first_center = first.mapToScene(first.viewport().rect().center())
    second_center = second.mapToScene(second.viewport().rect().center())
    assert first_center.x() == pytest.approx(second_center.x(), abs=2)
    assert first_center.y() == pytest.approx(second_center.y(), abs=2)
