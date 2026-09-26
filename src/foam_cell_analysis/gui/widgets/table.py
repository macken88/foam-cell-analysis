"""Qt の表ウィジェットに共通設定を適用する。"""

from PySide6.QtCore import QEvent, QItemSelection, QItemSelectionModel, QObject
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QPushButton,
    QTableView,
    QTableWidget,
)

from ..theme import set_style

MAX_CONTENT_COLUMN_WIDTH = 360


class _HeaderResizeTracker(QObject):
    """ヘッダー上のマウスドラッグによる利用者設定幅を記録する。"""

    def __init__(self, view: QTableView | QTableWidget) -> None:
        super().__init__(view)
        self.view = view
        self.header = view.horizontalHeader()
        self.header_viewport = self.header.viewport()
        self.active_section: int | None = None
        self.header.sectionResized.connect(self._section_resized)
        self.header_viewport.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        header = getattr(self, "header", None)
        header_viewport = getattr(self, "header_viewport", None)
        if header is None or header_viewport is None:
            return False
        if watched is header or watched is header_viewport:
            if event.type() == QEvent.Type.MouseButtonPress:
                x = event.position().toPoint().x()
                self.active_section = None
                for section in range(self.header.count()):
                    edge = self.header.sectionViewportPosition(section) + self.header.sectionSize(
                        section
                    )
                    if (
                        abs(x - edge) <= 5
                        and self.header.sectionResizeMode(section)
                        == QHeaderView.ResizeMode.Interactive
                    ):
                        self.active_section = section
                        break
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self.active_section = None
        return False

    def _section_resized(self, section: int, _old: int, _new: int) -> None:
        if self.active_section == section and not getattr(self.view, "_auto_sizing_columns", False):
            self.view._user_sized_columns.add(section)


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
    if not hasattr(view, "_user_sized_columns"):
        view._user_sized_columns = set()
        view._auto_sizing_columns = False
        view._header_resize_tracker = _HeaderResizeTracker(view)
        header.installEventFilter(view._header_resize_tracker)
    view._stretch_column = stretch_column
    view.resizeColumnsToContents()


def fit_table_columns(view: QTableView | QTableWidget) -> None:
    """列を見出し・内容に合わせ、利用者が変えた列幅は保持する。"""
    model = view.model()
    if model is None:
        return
    header = view.horizontalHeader()
    stretch_column = getattr(view, "_stretch_column", None)
    user_sized = getattr(view, "_user_sized_columns", set())
    view._auto_sizing_columns = True
    try:
        for column in range(model.columnCount()):
            if column in user_sized or column == stretch_column:
                continue
            header_width = header.sectionSizeHint(column)
            content_width = view.sizeHintForColumn(column)
            width = min(max(header_width, content_width) + 16, MAX_CONTENT_COLUMN_WIDTH)
            view.setColumnWidth(column, max(24, width))
        if stretch_column is not None and stretch_column not in user_sized:
            other_width = sum(
                view.columnWidth(column)
                for column in range(model.columnCount())
                if column != stretch_column
            )
            remaining = view.viewport().width() - other_width
            natural = min(
                max(header.sectionSizeHint(stretch_column), view.sizeHintForColumn(stretch_column))
                + 16,
                MAX_CONTENT_COLUMN_WIDTH,
            )
            view.setColumnWidth(stretch_column, max(natural, remaining))
    finally:
        view._auto_sizing_columns = False


def mark_primary(button: QPushButton) -> None:
    """ボタンへ主操作用スタイルプロパティを適用する。"""
    set_style(button, primary=True)
