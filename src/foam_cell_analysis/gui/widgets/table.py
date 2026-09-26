"""Qt の表ウィジェットに共通設定を適用する。"""

from PySide6.QtCore import QItemSelection, QItemSelectionModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QPushButton,
    QTableView,
    QTableWidget,
)

from ..theme import set_style


def restore_row_selection(
    view: QTableView | QTableWidget,
    rows: set[int] | list[int],
    current_row: int | None = None,
) -> None:
    """表の行選択と現在行をまとめて復元する。"""
    model = view.model()
    if model is None or model.columnCount() == 0:
        return
    selection = QItemSelection()
    valid_rows = sorted({row for row in rows if 0 <= row < model.rowCount()})
    for row in valid_rows:
        selection.select(model.index(row, 0), model.index(row, model.columnCount() - 1))
    selection_model = view.selectionModel()
    selection_model.select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)
    if current_row is not None and 0 <= current_row < model.rowCount():
        selection_model.setCurrentIndex(
            model.index(current_row, 0), QItemSelectionModel.SelectionFlag.NoUpdate
        )


def setup_table(
    view: QTableView | QTableWidget,
    stretch_column: int | None = None,
    selection_mode: QAbstractItemView.SelectionMode = (
        QAbstractItemView.SelectionMode.SingleSelection
    ),
) -> None:
    """表の見出し、選択方式、列幅を共通設定する。"""
    view.verticalHeader().setVisible(False)
    view.setAlternatingRowColors(False)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.setSelectionMode(selection_mode)
    view.setWordWrap(False)
    view.setShowGrid(False)

    header = view.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    column_count = view.model().columnCount() if view.model() is not None else 0
    if stretch_column is not None and stretch_column >= column_count:
        raise ValueError("stretch_column が列数の範囲外です")
    view.resizeColumnsToContents()


def mark_primary(button: QPushButton) -> None:
    """ボタンへ主操作用スタイルプロパティを適用する。"""
    set_style(button, primary=True)
