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
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    column_count = view.model().columnCount() if view.model() is not None else 0
    target_column = column_count - 1 if stretch_column is None else stretch_column
    if target_column < 0:
        header.setStretchLastSection(True)
    elif target_column >= column_count:
        raise ValueError("stretch_column が列数の範囲外です")
    else:
        header.setStretchLastSection(False)
        header.setSectionResizeMode(target_column, QHeaderView.ResizeMode.Stretch)


def mark_primary(button: QPushButton) -> None:
    """ボタンへ主操作用スタイルプロパティを適用する。"""
    set_style(button, primary=True)
