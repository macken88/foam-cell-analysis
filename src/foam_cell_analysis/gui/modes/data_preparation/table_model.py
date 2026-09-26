"""データ準備の一枚表と版履歴用モデル。"""

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import QComboBox, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from ....services.models import DataItem, DatasetVersion
from ...theme import Color, numeric_font

HEADERS = (
    "取り込み元フォルダ",
    "データ識別子",
    "元ファイル名",
    "用途",
    "画像分類",
    "品質",
    "マスク版",
    "⚠",
)
USAGE_TEXT = {"unassigned": "未振り分け", "train": "学習", "val": "検証", "excluded": "不採用"}


class DataPreparationTableModel(QAbstractTableModel):
    """変更・検索・用途で絞り込む作業表モデル。"""

    edit_requested = Signal(str, str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.items: list[DataItem] = []
        self.errors: dict[str, str] = {}
        self.query = ""
        self.usages: set[str] | None = None
        self.classifications: set[str] = set()
        self.source_folder: str | None = None
        self.changed_only = False
        self.errors_only = False
        self._visible_cache: list[DataItem] | None = None

    def set_items(self, items: list[DataItem], errors: dict[str, str] | None = None) -> None:
        """モデル行を入れ替える。"""
        self.beginResetModel()
        self.items = list(items)
        self.errors = errors or {}
        self._visible_cache = None
        self.endResetModel()

    def visible_items(self) -> list[DataItem]:
        """現在の絞り込みに合う項目を返す。"""
        query = self.query.casefold()
        if self._visible_cache is None:
            self._visible_cache = [
                item
                for item in self.items
                if (self.usages is None or item.usage in self.usages)
                and (not self.classifications or item.classification in self.classifications)
                and (self.source_folder is None or item.source_folder == self.source_folder)
                and (not self.changed_only or item.change is not None)
                and (not self.errors_only or item.item_id in self.errors)
                and (
                    not query
                    or query in item.item_id.casefold()
                    or query in item.source_filename.casefold()
                    or query in item.source_relpath.casefold()
                )
            ]
        return self._visible_cache

    def filters_changed(self) -> None:
        """絞り込み条件変更を表へ通知する。"""
        self._visible_cache = None
        self.layoutChanged.emit()

    def update_item_errors(
        self, errors: dict[str, str], item_ids: set[str], *, refresh_layout: bool
    ) -> None:
        """指定項目の検査結果だけ差し替えて表示を更新する。"""
        for item_id in item_ids:
            self.errors.pop(item_id, None)
        self.errors.update(errors)
        if refresh_layout:
            self._visible_cache = None
            self.layoutChanged.emit()
        visible = self.visible_items()
        rows = [row for row, item in enumerate(visible) if item.item_id in item_ids]
        for row in rows:
            self.dataChanged.emit(
                self.index(row, 0),
                self.index(row, len(HEADERS) - 1),
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole],
            )

    def rowCount(self, parent=None) -> int:
        if parent is None:
            return len(self.visible_items())
        return 0 if parent.isValid() else len(self.visible_items())

    def columnCount(self, parent=None) -> int:
        if parent is None:
            return len(HEADERS)
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return HEADERS[section]
        return None

    def item_at(self, row: int) -> DataItem | None:
        visible = self.visible_items()
        return visible[row] if 0 <= row < len(visible) else None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        item = self.item_at(index.row())
        if item is None:
            return None
        column = index.column()
        if role == Qt.ItemDataRole.UserRole:
            return item
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return (
                item.source_folder,
                item.item_id,
                item.source_filename,
                USAGE_TEXT[item.usage],
                item.classification or "未設定",
                item.quality or "未設定",
                item.selected_mask_revision or "なし",
                "⚠" if item.item_id in self.errors else "",
            )[column]
        if role == Qt.ItemDataRole.ToolTipRole and column == 7 and item.item_id in self.errors:
            return self.errors[item.item_id]
        if role == Qt.ItemDataRole.BackgroundRole and item.change:
            return QColor(Color.CHANGED)
        if role == Qt.ItemDataRole.TextAlignmentRole and column in (1, 6, 7):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.FontRole and column in (1, 6):
            font = numeric_font()
            return font
        return None

    def flags(self, index: QModelIndex):
        flags = super().flags(index)
        if index.isValid() and index.column() in (3, 4, 5):
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        item = self.item_at(index.row())
        if item is None:
            return False
        if index.column() == 3:
            usage = {label: key for key, label in USAGE_TEXT.items()}.get(str(value))
            if usage is None:
                return False
            self.edit_requested.emit(item.item_id, "usage", usage)
        else:
            key = "classification" if index.column() == 4 else "quality"
            self.edit_requested.emit(
                item.item_id, key, None if str(value) == "未設定" else str(value)
            )
        return True


class ValueComboDelegate(QStyledItemDelegate):
    """作業表の分類・品質に選択コンボを表示する。"""

    def __init__(self, values: list[str], parent=None):
        super().__init__(parent)
        self.values = values

    def paint(self, painter: QPainter, option, index) -> None:
        item = index.data(Qt.ItemDataRole.UserRole)
        if isinstance(item, DataItem) and item.change:
            color = (
                Color.SELECTION_CHANGED
                if option.state & QStyle.StateFlag.State_Selected
                else Color.CHANGED
            )
            painter.fillRect(option.rect, QColor(color))
            prepared = QStyleOptionViewItem(option)
            if option.state & QStyle.StateFlag.State_Selected:
                prepared.palette.setColor(QPalette.ColorRole.Highlight, QColor(color))
                prepared.palette.setColor(
                    QPalette.ColorRole.HighlightedText, QColor(Color.GRAPHITE)
                )
            super().paint(painter, prepared, index)
            return
        super().paint(painter, option, index)

    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        editor.addItems(self.values)
        return editor

    def setEditorData(self, editor, index):
        editor.setCurrentText(str(index.data(Qt.ItemDataRole.EditRole) or "未設定"))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class DatasetHistoryModel(QAbstractTableModel):
    """確定済みデータセット版の履歴モデル。"""

    HEADERS = ("版", "用途", "件数", "作成日時", "親版", "コメント")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.versions: list[DatasetVersion] = []

    def set_versions(self, versions: list[DatasetVersion]) -> None:
        self.beginResetModel()
        self.versions = list(versions)
        self.endResetModel()

    def rowCount(self, parent=None) -> int:
        if parent is None:
            return len(self.versions)
        return 0 if parent.isValid() else len(self.versions)

    def columnCount(self, parent=None) -> int:
        if parent is None:
            return len(self.HEADERS)
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        record = self.versions[index.row()]
        values = (
            record.version,
            "学習" if record.purpose == "train" else "検証",
            record.n_images,
            record.created_at.strftime("%Y-%m-%d %H:%M"),
            record.parent_version or "—",
            record.comment,
        )
        return values[index.column()]

    edit_requested = Signal(str, str, object)
