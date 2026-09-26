"""学習キュー用の model/view データモデルと編集デリゲート。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QEvent, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QStyledItemDelegate,
)

from ...context import AppContext
from ...labels import (
    classification_label,
    experiment_status_label,
    model_type_label,
    quality_filter_label,
)
from ...theme import Color, numeric_font

FIELDS = [
    ("experiment.description", "説明", "実験"),
    ("experiment.study_id", "実験群", "実験"),
    ("data.dataset_version", "データセット版", "データ"),
    ("data.classification", "画像分類", "データ"),
    ("data.quality_filter", "品質条件", "データ"),
    ("data.cv.n_folds", "分割数", "データ"),
    ("data.cv.stratify_by_classification", "画像分類で層別", "データ"),
    ("data.cv.group_by_source_folder", "フォルダ単位で分割", "データ"),
    ("data.seed", "乱数シード", "データ"),
    ("model.type", "モデル種類", "モデル"),
    ("model.pretrained_weights", "事前学習済み重み", "モデル"),
    ("model.pretrained_model", "事前学習済みモデル（Cellpose）", "モデル"),
    ("training.epochs", "エポック数", "共通学習設定"),
    ("training.batch_size", "バッチサイズ", "共通学習設定"),
    ("training.learning_rate", "学習率", "共通学習設定"),
    ("training.weight_decay", "重み減衰", "共通学習設定"),
    ("model.optimizer", "最適化手法（Cellpose）", "共通学習設定"),
    ("training.early_stopping.enabled", "早期終了（有効）", "共通学習設定"),
    ("training.early_stopping.patience", "待機エポック数", "共通学習設定"),
    ("augmentation.profile", "プロファイル", "データ拡張"),
]

FIXED_HEADERS = ("順番", "状態", "実験識別子")
CHOICE_OPTIONS = {
    "data.dataset_version": "datasets",
    "data.classification": "classifications",
    "model.pretrained_weights": "pretrained",
    "model.pretrained_model": "cellpose_models",
    "model.optimizer": "optimizers",
    "augmentation.profile": "profiles",
}
NUMERIC_PATHS = {
    path
    for path, _label, _group in FIELDS
    if path
    not in {
        "experiment.description",
        "experiment.study_id",
        "data.classification",
        "data.quality_filter",
        "model.type",
        "model.pretrained_weights",
        "model.pretrained_model",
        "model.optimizer",
        "training.early_stopping.enabled",
        "augmentation.profile",
        "data.cv.stratify_by_classification",
        "data.cv.group_by_source_folder",
    }
}


def _get_path(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


class TrainingQueueModel(QAbstractTableModel):
    """実験キューを編集部品から分離して表示する。"""

    edit_failed = Signal(str)

    def __init__(self, ctx: AppContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.entries = []
        self._issues: dict[str, tuple[list[str], list[str]]] = {}
        self.refresh()

    @property
    def fixed_column_count(self) -> int:
        return len(FIXED_HEADERS)

    def column_for_path(self, path: str) -> int:
        return self.fixed_column_count + next(
            index for index, field in enumerate(FIELDS) if field[0] == path
        )

    def row_for_id(self, experiment_id: str) -> int:
        return next(
            (row for row, entry in enumerate(self.entries) if entry.experiment_id == experiment_id),
            -1,
        )

    def rowCount(self, parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.entries)

    def columnCount(self, parent=None) -> int:
        return (
            0 if parent is not None and parent.isValid() else self.fixed_column_count + len(FIELDS)
        )

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            if section < self.fixed_column_count:
                return FIXED_HEADERS[section]
            return FIELDS[section - self.fixed_column_count][1]
        if role == Qt.ItemDataRole.ToolTipRole and orientation == Qt.Orientation.Horizontal:
            if section >= self.fixed_column_count:
                return FIELDS[section - self.fixed_column_count][2]
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        entry = self.entries[index.row()]
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() >= self.fixed_column_count and entry.status == "queued":
            path = FIELDS[index.column() - self.fixed_column_count][0]
            if _get_path(entry.config.values, path) is not None:
                flags |= Qt.ItemFlag.ItemIsEditable
                if isinstance(_get_path(entry.config.values, path), bool):
                    flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.entries):
            return None
        entry = self.entries[index.row()]
        errors, warnings = self._issues.get(entry.experiment_id, ([], []))
        if role == Qt.ItemDataRole.ToolTipRole:
            return "\n".join(errors or warnings) or None
        if role == Qt.ItemDataRole.BackgroundRole:
            if errors and entry.status == "queued":
                return QColor(Color.ERROR_BG)
            if index.column() >= self.fixed_column_count:
                value = _get_path(
                    entry.config.values, FIELDS[index.column() - self.fixed_column_count][0]
                )
                if value is None:
                    return QColor(Color.IDLE_BG)
        if role == Qt.ItemDataRole.FontRole and (
            index.column() in (0, 2)
            or (
                index.column() >= self.fixed_column_count
                and FIELDS[index.column() - self.fixed_column_count][0] in NUMERIC_PATHS
            )
        ):
            return numeric_font()
        if role == Qt.ItemDataRole.TextAlignmentRole and (
            index.column() in (0, 2)
            or (
                index.column() >= self.fixed_column_count
                and FIELDS[index.column() - self.fixed_column_count][0] in NUMERIC_PATHS
            )
        ):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if index.column() == 0:
            if role == Qt.ItemDataRole.DisplayRole:
                order = str(index.row() + 1)
                return f"⚠ {order}" if errors and entry.status == "queued" else order
            return None
        if index.column() == 1 and role == Qt.ItemDataRole.DisplayRole:
            if entry.status == "queued" and errors:
                return "設定エラー"
            if entry.status == "queued" and warnings:
                return "△ 待機"
            state = experiment_status_label(entry.status)
            if entry.status == "running":
                state += f"（{self._progress(entry)}）"
            return state
        if index.column() == 2:
            return entry.experiment_id if role == Qt.ItemDataRole.DisplayRole else None
        path = FIELDS[index.column() - self.fixed_column_count][0]
        value = _get_path(entry.config.values, path)
        if role == Qt.ItemDataRole.CheckStateRole and isinstance(value, bool):
            return Qt.CheckState.Checked if value else Qt.CheckState.Unchecked
        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(value, bool) or value is None:
                return ""
            return self._display_value(path, value)
        return None

    def _progress(self, entry) -> str:
        if entry.phase == "final_training":
            total = entry.selected_epoch or entry.total_epochs
            return f"最終学習・epoch {entry.current_epoch}/{total}"
        folds = entry.config.values.get("data", {}).get("cv", {}).get("n_folds", 5)
        fold = max(entry.fold_histories, default=1)
        return f"分割 {fold}/{folds}・epoch {entry.current_epoch}/{entry.total_epochs}"

    @staticmethod
    def _display_value(path: str, value: Any) -> str:
        if path == "model.type":
            return model_type_label(value)
        if path == "data.classification":
            return classification_label(value) if value != "all" else "全分類"
        if path == "data.quality_filter":
            return quality_filter_label(value)
        if path == "model.pretrained_weights":
            return {"coco": "COCO", "imagenet": "ImageNet"}.get(value, str(value))
        return str(value)

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or index.column() < self.fixed_column_count:
            return False
        entry = self.entries[index.row()]
        if entry.status != "queued":
            return False
        path = FIELDS[index.column() - self.fixed_column_count][0]
        config = deepcopy(entry.config.values)
        try:
            if role == Qt.ItemDataRole.CheckStateRole:
                value = value == Qt.CheckState.Checked
            elif path in NUMERIC_PATHS:
                old = _get_path(config, path)
                value = int(value) if isinstance(old, int) else float(value)
            if path == "model.type":
                if value not in {"mask_rcnn", "cellpose"}:
                    return False
                new_model = deepcopy(self.ctx.backend.default_experiment_config(value)["model"])
                new_model["type"] = value
                config["model"] = new_model
            else:
                _set_path(config, path, value)
            self.ctx.backend.update_training_queue_item(entry.experiment_id, config)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            self.edit_failed.emit(str(error))
            return False
        self.refresh()
        self.dataChanged.emit(
            self.index(index.row(), 0),
            self.index(index.row(), self.columnCount() - 1),
        )
        return True

    def refresh(self) -> bool:
        """行構成が変わったときだけ reset し、進捗は状態セルだけ更新する。"""
        entries = self.ctx.backend.list_training_queue()
        ids = [entry.experiment_id for entry in entries]
        old_ids = [entry.experiment_id for entry in self.entries]
        old_issues = self._issues
        changed_structure = ids != old_ids
        if changed_structure:
            self.beginResetModel()
            self.entries = entries
            self.endResetModel()
        else:
            self.entries = entries
        issues = {}
        for entry in entries:
            try:
                validation = self.ctx.backend.validate_experiment_config(entry.config.values)
            except Exception as error:
                validation = [{"level": "error", "message": str(error)}]
            issues[entry.experiment_id] = (
                [item["message"] for item in validation if item["level"] == "error"],
                [item["message"] for item in validation if item["level"] == "warning"],
            )
        self._issues = issues
        if not changed_structure and entries:
            self.dataChanged.emit(
                self.index(0, 1),
                self.index(len(entries) - 1, 1),
                [
                    Qt.ItemDataRole.DisplayRole,
                    Qt.ItemDataRole.ToolTipRole,
                    Qt.ItemDataRole.BackgroundRole,
                ],
            )
            for row, entry in enumerate(entries):
                if old_issues.get(entry.experiment_id) != issues[entry.experiment_id]:
                    self.dataChanged.emit(
                        self.index(row, 0),
                        self.index(row, self.columnCount() - 1),
                        [
                            Qt.ItemDataRole.DisplayRole,
                            Qt.ItemDataRole.ToolTipRole,
                            Qt.ItemDataRole.BackgroundRole,
                        ],
                    )
        return changed_structure


class TrainingQueueDelegate(QStyledItemDelegate):
    """表のセルが編集状態になったときだけ小さな editor を生成する。"""

    def __init__(self, model: TrainingQueueModel, parent=None):
        super().__init__(parent)
        self.queue_model = model

    def editorEvent(self, event, model, option, index):
        if index.column() >= self.queue_model.fixed_column_count:
            path = FIELDS[index.column() - self.queue_model.fixed_column_count][0]
            value = _get_path(self.queue_model.entries[index.row()].config.values, path)
            if isinstance(value, bool) and event.type() == QEvent.Type.MouseButtonRelease:
                next_state = Qt.CheckState.Unchecked if value else Qt.CheckState.Checked
                return model.setData(index, next_state, Qt.ItemDataRole.CheckStateRole)
        return super().editorEvent(event, model, option, index)

    def createEditor(self, parent, option, index):
        path = FIELDS[index.column() - self.queue_model.fixed_column_count][0]
        value = _get_path(self.queue_model.entries[index.row()].config.values, path)
        if isinstance(value, bool):
            return None
        if path == "model.type":
            editor = QComboBox(parent)
            editor.addItem("Mask R-CNN", "mask_rcnn")
            editor.addItem("Cellpose", "cellpose")
            return editor
        if path in CHOICE_OPTIONS or path == "data.quality_filter":
            editor = QComboBox(parent)
            options = self.queue_model.ctx.backend.list_training_options()
            choices = (
                ["all", *options.get("classifications", [])]
                if path == "data.classification"
                else ["all", "good_only", "good_and_acceptable"]
                if path == "data.quality_filter"
                else options.get(CHOICE_OPTIONS[path], [])
            )
            for choice in choices:
                editor.addItem(self.queue_model._display_value(path, choice), choice)
            return editor
        if path in NUMERIC_PATHS:
            if isinstance(value, float):
                editor = QDoubleSpinBox(parent)
                editor.setRange(-1e12, 1e12)
                editor.setDecimals(
                    12 if path in {"training.learning_rate", "training.weight_decay"} else 6
                )
                editor.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
                editor.setKeyboardTracking(False)
                return editor
            editor = QSpinBox(parent)
            editor.setRange(-1_000_000, 1_000_000)
            editor.setKeyboardTracking(False)
            return editor
        return QLineEdit(parent)

    def setEditorData(self, editor, index):
        path = FIELDS[index.column() - self.queue_model.fixed_column_count][0]
        value = _get_path(self.queue_model.entries[index.row()].config.values, path)
        if isinstance(editor, QComboBox):
            selected = editor.findData(value)
            editor.setCurrentIndex(max(0, selected))
        elif isinstance(editor, QSpinBox | QDoubleSpinBox):
            editor.setValue(value)
        elif isinstance(editor, QLineEdit):
            editor.setText(str(value))

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            value = editor.currentData()
        elif isinstance(editor, QSpinBox | QDoubleSpinBox):
            value = editor.value()
        elif isinstance(editor, QLineEdit):
            value = editor.text()
        else:
            return
        model.setData(index, value, Qt.ItemDataRole.EditRole)
