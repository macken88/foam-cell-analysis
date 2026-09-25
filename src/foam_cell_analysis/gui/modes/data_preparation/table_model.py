"""データ準備画面の表モデルと編集 delegate。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import (
    QAbstractItemModel,
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QStyledItemDelegate, QStyleOptionViewItem, QWidget

from ....services.models import (
    CheckResult,
    DataItem,
    DatasetVersion,
    ValidationIssue,
    ValidationReport,
)
from ...labels import format_datetime
from ...theme import Color, numeric_font


class DataItemModel(QAbstractTableModel):
    """作業中データと直接編集可能な列を表示する。"""

    headers = [
        "採用",
        "データ識別子",
        "元ファイル名",
        "画像分類",
        "品質",
        "チャンネル",
        "マスク版",
        "変更",
        "検証",
    ]
    classifications = ["分類A", "分類B", "分類C", "未設定"]
    qualities = ["良", "可", "不良", "未設定"]
    edit_requested = Signal(str, dict)

    def __init__(self, on_edit: Callable[[str, dict], None]) -> None:
        super().__init__()
        self.items: list[DataItem] = []
        self.issues: dict[str, int] = {}
        self.edit_requested.connect(on_edit)

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.items)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.headers)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        item = self.items[index.row()]
        column = index.column()
        if role == Qt.ItemDataRole.FontRole and column == 1:
            return numeric_font()
        if role == Qt.ItemDataRole.TextAlignmentRole and column == 1:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.CheckStateRole and column == 0:
            return Qt.CheckState.Checked if item.included else Qt.CheckState.Unchecked
        if role == Qt.ItemDataRole.EditRole:
            if column == 3:
                return item.classification or "未設定"
            if column == 4:
                return item.quality or "未設定"
            if column == 6:
                return item.selected_mask_revision
        if role == Qt.ItemDataRole.DisplayRole:
            values = (
                "",
                item.item_id,
                item.source_filename,
                item.classification or "未設定",
                item.quality or "未設定",
                ", ".join(item.channels),
                item.selected_mask_revision or "—",
                {"added": "追加", "changed": "変更", "excluded": "除外", None: "—"}.get(
                    item.change, item.change
                ),
                f"⚠ {self.issues[item.item_id]}" if self.issues.get(item.item_id) else "OK",
            )
            return values[column]
        if role == Qt.ItemDataRole.BackgroundRole and (
            item.change or self.issues.get(item.item_id)
        ):
            if self.issues.get(item.item_id):
                return QColor(Color.ERROR_BG)
            return QColor(Color.TRAIN_BG)
        if role == Qt.ItemDataRole.ToolTipRole and column == 8 and self.issues.get(item.item_id):
            return f"整合性エラー {self.issues[item.item_id]} 件"
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            return base | Qt.ItemFlag.ItemIsUserCheckable
        if index.column() in (3, 4, 6):
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid():
            return False
        item = self.items[index.row()]
        if index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            self.edit_requested.emit(item.item_id, {"included": value == Qt.CheckState.Checked})
            return True
        field = {3: "classification", 4: "quality", 6: "selected_mask_revision"}.get(index.column())
        if field and role == Qt.ItemDataRole.EditRole:
            normalized = None if value in ("未設定", "") else str(value)
            if getattr(item, field) != normalized:
                self.edit_requested.emit(item.item_id, {field: normalized})
            return True
        return False

    def set_items(self, items: list[DataItem], validation: ValidationReport | None) -> None:
        self.beginResetModel()
        self.items = list(items)
        self.issues = {}
        if validation:
            for issue in validation.errors:
                self.issues[issue.item_id] = self.issues.get(issue.item_id, 0) + 1
        self.endResetModel()

    def item_id_at(self, row: int) -> str | None:
        return self.items[row].item_id if 0 <= row < len(self.items) else None

    def row_for_id(self, item_id: str) -> int:
        return next((row for row, item in enumerate(self.items) if item.item_id == item_id), -1)

    def choices(self, row: int, column: int) -> list[str]:
        if not 0 <= row < len(self.items):
            return []
        if column == 3:
            return self.classifications
        if column == 4:
            return self.qualities
        if column == 6:
            return self.items[row].mask_revisions
        return []


class ItemComboDelegate(QStyledItemDelegate):
    """分類、品質、マスク版を行に応じたコンボで編集する。"""

    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QWidget:
        editor = QComboBox(parent)
        model = index.model()
        if isinstance(model, QSortFilterProxyModel):
            index = model.mapToSource(index)
            model = model.sourceModel()
        source = model
        if isinstance(source, DataItemModel):
            editor.addItems(source.choices(index.row(), index.column()))
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        editor.setCurrentText(str(index.data(Qt.ItemDataRole.EditRole) or ""))

    def setModelData(self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex) -> None:
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class ItemFilterProxy(QSortFilterProxyModel):
    """作業データを分類、品質、変更、エラー、文字列で絞り込む。"""

    def __init__(self) -> None:
        super().__init__()
        self.classification = "すべて"
        self.quality = "すべて"
        self.changed_only = False
        self.errors_only = False
        self.search_text = ""
        self.setDynamicSortFilter(True)

    def set_filters(
        self,
        classification: str,
        quality: str,
        changed_only: bool,
        errors_only: bool,
        search_text: str,
    ) -> None:
        """条件をまとめて更新し、互換性のある方法で再絞り込みする。"""
        values = (classification, quality, changed_only, errors_only, search_text)
        if hasattr(self, "beginFilterChange"):
            self.beginFilterChange()
            self._assign_filters(values)
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
        else:
            self._assign_filters(values)
            self.invalidateRowsFilter()

    def _assign_filters(self, values: tuple[str, str, bool, bool, str]) -> None:
        (
            self.classification,
            self.quality,
            self.changed_only,
            self.errors_only,
            self.search_text,
        ) = values

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        if not isinstance(model, DataItemModel):
            return True
        item = model.items[source_row]
        if self.classification not in ("すべて", item.classification or "未設定"):
            return False
        if self.quality not in ("すべて", item.quality or "未設定"):
            return False
        if self.changed_only and item.change is None:
            return False
        if self.errors_only and not model.issues.get(item.item_id):
            return False
        if (
            self.search_text
            and self.search_text not in f"{item.item_id} {item.source_filename}".casefold()
        ):
            return False
        return True


class ErrorTableModel(QAbstractTableModel):
    """検査エラー表。"""

    headers = ["重要度", "データ識別子", "検査項目", "内容"]
    item_activated = Signal(str)

    def __init__(self, on_activate: Callable[[str], None]) -> None:
        super().__init__()
        self.issues = []
        self.item_activated.connect(on_activate)

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.issues)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        issue = self.issues[index.row()]
        if role == Qt.ItemDataRole.FontRole and index.column() == 1:
            return numeric_font()
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() == 1:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        return (issue.severity, issue.item_id, issue.check, issue.message)[index.column()]

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def set_issues(self, issues: list[ValidationIssue]) -> None:
        self.beginResetModel()
        self.issues = list(issues)
        self.endResetModel()

    def activate(self, row: int) -> None:
        if 0 <= row < len(self.issues):
            self.item_activated.emit(self.issues[row].item_id)


class HistoryTableModel(QAbstractTableModel):
    """確定済みデータセット版の読み取り専用表。"""

    headers = ["版", "用途", "親版", "作成日時", "画像数", "状態", "アーカイブ状態", "コメント"]
    archive_status_labels = {
        "NOT_CREATED": "未作成",
        "COMPLETED": "作成済み",
        "FAILED": "失敗",
    }

    def __init__(self) -> None:
        super().__init__()
        self.versions: list[DatasetVersion] = []

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.versions)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        version = self.versions[index.row()]
        if role == Qt.ItemDataRole.FontRole and index.column() in (0, 2, 3, 4):
            return numeric_font()
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() in (0, 2, 3, 4):
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        values = (
            version.version,
            "学習用" if version.purpose == "train" else "検証用",
            version.parent_version or "—",
            format_datetime(version.created_at),
            version.n_images,
            {"RELEASED": "確定済み"}.get(version.status, version.status),
            self.archive_status_labels.get(version.archive_status, version.archive_status),
            version.comment,
        )
        return values[index.column()]

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def set_versions(self, versions: list[DatasetVersion]) -> None:
        self.beginResetModel()
        self.versions = list(reversed(versions))
        self.endResetModel()

    def version_at(self, row: int) -> DatasetVersion | None:
        return self.versions[row] if 0 <= row < len(self.versions) else None


class CheckTableModel(QAbstractTableModel):
    """整合性検査項目の結果を表示する。"""

    headers = ["検査項目", "状態", "件数"]

    def __init__(self) -> None:
        super().__init__()
        self.checks: list[CheckResult] = []

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.checks)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        check = self.checks[index.row()]
        if role == Qt.ItemDataRole.FontRole and index.column() == 2:
            return numeric_font()
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() == 2:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        return (check.label, check.status, check.count)[index.column()]

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def set_checks(self, checks: list[CheckResult]) -> None:
        self.beginResetModel()
        self.checks = list(checks)
        self.endResetModel()
