"""Qt の表ウィジェットに共通設定を適用する。"""

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QPushButton,
    QTableView,
    QTableWidget,
)

from ..theme import set_style


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
