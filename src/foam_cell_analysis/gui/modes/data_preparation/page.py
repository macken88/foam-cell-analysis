"""作業データを表・サムネイルで整えて同時確定する画面。"""

from __future__ import annotations

from collections import Counter

from PySide6.QtCore import QEvent, QItemSelectionModel, Qt
from PySide6.QtGui import QAction, QBrush, QKeySequence, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ....services.models import DataItem, ImportCandidate
from ...context import AppContext
from ...keymap_dialog import KeymapDialog
from ...shortcuts import ShortcutMap
from ...theme import Color, numeric_font, set_style
from ...widgets.image_convert import DisplayMode, array_to_pixmap, render
from ...widgets.image_view import ImageView
from ...widgets.marks import USAGE_MARKS, CountChip, KeyHintBar, TagDelegate
from ...widgets.page_base import BasePage
from .dialogs import (
    AutoTriageDialog,
    ContinuousTriageDialog,
    DatasetFinalizeDialog,
    ImportDialog,
    ImportSettingsDialog,
)
from .table_model import (
    USAGE_TEXT,
    DataPreparationTableModel,
    DatasetHistoryModel,
    ValueComboDelegate,
)
from .thumbnails import (
    FolderHeader,
    ThumbnailDelegate,
    ThumbnailListView,
    ThumbnailModel,
    configure_thumbnail_view,
)


class _EditCommand(QUndoCommand):
    """選択項目への一括変更を一つの取り消し操作にする。"""

    def __init__(self, page: DataPreparationPage, item_ids: list[str], changes: dict, text: str):
        super().__init__(text)
        self.page = page
        self.item_ids = list(item_ids)
        self.changes = changes
        self.before = {
            item.item_id: {key: getattr(item, key) for key in changes}
            for item in page.items
            if item.item_id in item_ids
        }
        self.before_flags = {
            item.item_id: (item.change, item.previous_change)
            for item in page.items
            if item.item_id in item_ids
        }

    def redo(self) -> None:
        self.page._apply_changes(self.item_ids, self.changes)

    def undo(self) -> None:
        for item_id, values in self.before.items():
            self.page.ctx.backend.update_item("all", item_id, **values)
        self.page._restore_change_flags(self.before_flags)
        self.page.refresh()


class _UsageDelegate(TagDelegate):
    """用途タグを表示し、用途コンボで編集する。"""

    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        editor.addItems(list(self.mapping))
        return editor

    def paint(self, painter, option, index) -> None:
        item = index.data(Qt.ItemDataRole.UserRole)
        if (
            isinstance(item, DataItem)
            and item.change
            and not (option.state & QStyle.StateFlag.State_Selected)
        ):
            painter.fillRect(option.rect, Color.CHANGED)
        super().paint(painter, option, index)

    def setEditorData(self, editor, index) -> None:
        editor.setCurrentText(str(index.data(Qt.ItemDataRole.EditRole) or ""))

    def setModelData(self, editor, model, index) -> None:
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class _ChangedRowDelegate(QStyledItemDelegate):
    """変更行の背景色を表のすべてのセルへ伝える。"""

    def paint(self, painter, option, index) -> None:
        item = index.data(Qt.ItemDataRole.UserRole)
        prepared = QStyleOptionViewItem(option)
        if isinstance(item, DataItem) and item.change:
            prepared.backgroundBrush = QBrush(Color.CHANGED)
        super().paint(painter, prepared, index)


class _TriageCommand(QUndoCommand):
    """実行済みの自動振り分けを一括で取り消す。"""

    def __init__(
        self,
        page,
        before: dict[str, str],
        after: dict[str, str],
        before_flags: dict[str, tuple[str | None, str | None]],
    ):
        super().__init__("自動振り分け")
        self.page = page
        self.before = before
        self.after = after
        self.before_flags = before_flags
        self.first = True

    def redo(self):
        if self.first:
            self.first = False
            return
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)
        self.page._restore_change_flags(self.before_flags)
        self.page.refresh()

    def _apply(self, values):
        for item_id, usage in values.items():
            self.page.ctx.backend.update_item("all", item_id, usage=usage)
        self.page.refresh()


class DataPreparationPage(BasePage):
    """作業表、プレビュー、絞り込み、キー操作をまとめて表示する。"""

    def __init__(self, ctx: AppContext, parent=None, show_heading: bool = True) -> None:
        super().__init__(
            ctx, "作業中データ", "画像を確認して用途・分類・品質を決めます。", parent, show_heading
        )
        self.shortcuts = ShortcutMap()
        self.undo_stack = QUndoStack(self)
        self.items = ctx.backend.get_working_items()
        self.model = DataPreparationTableModel(self)
        self.thumbnail_model = ThumbnailModel(ctx.backend, self)
        self._selected_ids: list[str] = []
        self._syncing_selection = False
        root = self.content_layout
        root.setSpacing(6)
        self.base_label = QLabel()
        self.base_label.setFont(numeric_font(9))
        root.addWidget(self.base_label)
        toolbar = QHBoxLayout()
        self.import_button = QPushButton("取り込み…")
        self.auto_button = QPushButton("自動振り分け…")
        self.excel_button = QPushButton("Excel ▾")
        self.other_button = QPushButton("その他 ▾")
        self.finalize_button = QPushButton("確定…")
        self.finalize_button.setProperty("primary", True)
        for button in (self.import_button, self.auto_button, self.excel_button, self.other_button):
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.finalize_button)
        root.addLayout(toolbar)
        chips = QHBoxLayout()
        self.chips: dict[str, CountChip] = {}
        chip_defs = [
            ("all", "すべて", "plain"),
            ("unassigned", "未振り分け", "unassigned"),
            ("train", "学習", "train"),
            ("val", "検証", "val"),
            ("excluded", "不採用", "excluded"),
            ("errors", "⚠ エラー", "error"),
            ("changed", "変更あり", "plain"),
        ]
        chips.setSpacing(6)
        for key, label, usage in chip_defs:
            chip = CountChip(label, 0, usage)
            chip.clicked.connect(lambda checked=False, name=key: self._filter_usage(name))
            chips.addWidget(chip)
            self.chips[key] = chip
        self.chips["all"].setChecked(True)
        separator = QFrame()
        separator.setProperty("role", "chipSeparator")
        separator.setFixedSize(1, 22)
        chips.addWidget(separator)
        for classification in ("分類A", "分類B", "分類C"):
            chip = CountChip(classification, 0)
            chip.clicked.connect(
                lambda checked=False, value=classification: self._filter_class(value)
            )
            chips.addWidget(chip)
            self.chips[classification] = chip
        for chip in self.chips.values():
            chip.toggled.connect(lambda checked, target=chip: self._style_chip(target, checked))
            self._style_chip(chip, chip.isChecked())
        for name, key in (
            ("all", "filter_all"),
            ("unassigned", "filter_unassigned"),
            ("train", "filter_train"),
            ("val", "filter_val"),
            ("excluded", "filter_excluded"),
            ("errors", "filter_errors"),
        ):
            self.chips[name].setToolTip(self.shortcuts[key])
        chips.addStretch(1)
        root.addLayout(chips)
        filter_controls = QHBoxLayout()
        filter_controls.setSpacing(8)
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(155)
        self.source_combo.addItem("取り込み元: すべて", None)
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        filter_controls.addWidget(self.source_combo)
        self.search = QLineEdit()
        self.search.setMinimumWidth(175)
        self.search.setPlaceholderText("識別子・ファイル名を検索（Ctrl+F）")
        self.search.textChanged.connect(self._search_changed)
        filter_controls.addWidget(self.search, 1)
        root.addLayout(filter_controls)
        self._make_menus()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.views = QStackedWidget()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setItemDelegate(_ChangedRowDelegate(self.table))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().hide()
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for column, width in enumerate((140, 108, 125, 90, 78, 62, 74, 32)):
            self.table.setColumnWidth(column, width)
        usage_mapping = {
            USAGE_TEXT[key]: (values[1], values[2], values[3])
            for key, values in USAGE_MARKS.items()
        }
        self.table.setItemDelegateForColumn(3, _UsageDelegate(usage_mapping, self.table))
        self.table.setItemDelegateForColumn(
            4, ValueComboDelegate(["未設定", *ctx.backend.classifications], self.table)
        )
        self.table.setItemDelegateForColumn(
            5, ValueComboDelegate(["未設定", "良", "可", "不良"], self.table)
        )
        self.model.edit_requested.connect(self._table_edit_requested)
        self.thumbnail = ThumbnailListView()
        self.thumbnail.setModel(self.thumbnail_model)
        self.thumbnail.setItemDelegate(ThumbnailDelegate(self.thumbnail))
        configure_thumbnail_view(self.thumbnail)
        self.views.addWidget(self.table)
        self.views.addWidget(self.thumbnail)
        self.views.setCurrentIndex(1)
        self.splitter.addWidget(self.views)
        self.preview_panel = QWidget()
        preview = QVBoxLayout(self.preview_panel)
        preview.setContentsMargins(8, 0, 0, 0)
        self.preview_meta = QLabel("画像を選択してください")
        self.preview_meta.setFont(numeric_font(9))
        preview.addWidget(self.preview_meta)
        self.image_view = ImageView()
        self.image_view.setMinimumWidth(300)
        preview.addWidget(self.image_view, 1)
        self.image_caption = QLabel("画像表示形式・チャンネル")
        set_style(self.image_caption, role="note")
        preview.addWidget(self.image_caption)
        edits = QHBoxLayout()
        self.usage_combo = QComboBox()
        for key, label in USAGE_TEXT.items():
            self.usage_combo.addItem(label, key)
        self.class_combo = QComboBox()
        self.class_combo.addItem("未設定", None)
        for value in ctx.backend.classifications:
            self.class_combo.addItem(value, value)
        self.quality_combo = QComboBox()
        self.quality_combo.addItem("未設定", None)
        for value in ("良", "可", "不良"):
            self.quality_combo.addItem(value, value)
        self.mask_combo = QComboBox()
        self.display_combo = QComboBox()
        self.display_combo.addItems(["原画像", "オーバーレイ", "インスタンスラベル", "二値化"])
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["A", "B", "C"])
        for combo in (self.usage_combo, self.class_combo, self.quality_combo, self.mask_combo):
            combo.currentIndexChanged.connect(
                lambda _index, control=combo: self._edit_selection(control)
            )
        for label, combo in (
            ("用途", self.usage_combo),
            ("分類", self.class_combo),
            ("品質", self.quality_combo),
            ("マスク", self.mask_combo),
        ):
            edits.addWidget(QLabel(label))
            edits.addWidget(combo)
        preview.addLayout(edits)
        display = QHBoxLayout()
        display.addWidget(self.display_combo)
        display.addWidget(self.channel_combo)
        preview.addLayout(display)
        self.selection_note = QLabel("0 件選択中。変更は選択中のすべてに適用")
        preview.addWidget(self.selection_note)
        self.splitter.addWidget(self.preview_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([900, 588])
        root.addWidget(self.splitter, 1)
        self.hints = KeyHintBar(self.shortcuts.hint_items())
        root.addWidget(self.hints)
        self.table.selectionModel().selectionChanged.connect(
            lambda *_: self._selection_changed(self.table)
        )
        self.thumbnail.selectionModel().selectionChanged.connect(
            lambda *_: self._selection_changed(self.thumbnail)
        )
        self.table.installEventFilter(self)
        self.table.viewport().installEventFilter(self)
        self.thumbnail.installEventFilter(self)
        self.thumbnail.viewport().installEventFilter(self)
        self.import_button.clicked.connect(self.import_data)
        self.auto_button.clicked.connect(self.auto_triage)
        self.finalize_button.clicked.connect(self.finalize)
        self.display_combo.currentIndexChanged.connect(self._show_preview)
        self.channel_combo.currentIndexChanged.connect(self._show_preview)
        self.refresh()

    def _make_menus(self) -> None:
        excel = QMenu(self)
        excel.addAction("Excel 出力…", self.export_excel)
        excel.addAction("Excel 取込…", self.import_excel)
        self.excel_button.setMenu(excel)
        other = QMenu(self)
        other.addAction("連続振り分け…", self.open_triage)
        other.addAction("キー割り当て…", self.open_keymap)
        other.addAction("表示を切り替え", self.toggle_view)
        other.addAction("元に戻す", self.undo_stack.undo)
        other.addAction("やり直す", self.undo_stack.redo)
        self.other_button.setMenu(other)
        self._shortcut_actions = {}
        for action, callback in (
            ("import", self.import_data),
            ("auto_triage", self.auto_triage),
            ("export_excel", self.export_excel),
            ("import_excel", self.import_excel),
            ("finalize", self.finalize),
            ("undo", self.undo_stack.undo),
            ("redo", self.undo_stack.redo),
            ("search", self.search.setFocus),
        ):
            shortcut = QAction(self)
            shortcut.setShortcut(QKeySequence(self.shortcuts[action]))
            shortcut.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.triggered.connect(callback)
            self.addAction(shortcut)
            self._shortcut_actions[action] = shortcut
        for name, key in (
            ("all", "filter_all"),
            ("unassigned", "filter_unassigned"),
            ("train", "filter_train"),
            ("val", "filter_val"),
            ("excluded", "filter_excluded"),
            ("errors", "filter_errors"),
        ):
            action = QAction(self)
            action.setShortcut(QKeySequence(self.shortcuts[key]))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(lambda checked=False, value=name: self._filter_usage(value))
            self.addAction(action)
            self._shortcut_actions[key] = action

    def _apply_changes(self, item_ids: list[str], changes: dict) -> None:
        if "usage" in changes and len(changes) == 1:
            self.ctx.backend.set_usage(item_ids, changes["usage"])
        else:
            self.ctx.backend.update_items(item_ids, **changes)
        self.refresh()
        if item_ids:
            names = {
                "train": "学習",
                "val": "検証",
                "excluded": "不採用",
                "unassigned": "未振り分け",
            }
            label = names.get(
                changes.get("usage"),
                str(changes.get("classification", changes.get("quality", "変更"))),
            )
            first = next((item.item_id for item in self.items if item.item_id in item_ids), "")
            suffix = f" ほか {len(item_ids) - 1}件" if len(item_ids) > 1 else ""
            self.ctx.status.show_message(
                f"{first}{suffix} を {label} にしました（Ctrl+Z で元に戻す）"
            )

    def _restore_change_flags(self, values: dict[str, tuple[str | None, str | None]]) -> None:
        """Undo 実行前の差分表示状態へ戻す。"""
        by_id = {item.item_id: item for item in self.ctx.backend.get_working_items()}
        for item_id, (change, previous_change) in values.items():
            if item_id in by_id:
                by_id[item_id].change = change
                by_id[item_id].previous_change = previous_change

    def _table_edit_requested(self, item_id: str, field: str, value) -> None:
        self._change([item_id], **{field: value})

    def _change(self, item_ids: list[str], **changes) -> None:
        if item_ids:
            self.undo_stack.push(_EditCommand(self, item_ids, changes, "作業データの変更"))

    def _selected_ids_from(self, view) -> list[str]:
        ids = []
        for index in view.selectionModel().selectedRows():
            if view is self.table:
                item = self.model.item_at(index.row())
            else:
                item = self.thumbnail_model.data(index, Qt.ItemDataRole.UserRole)
            if isinstance(item, DataItem):
                ids.append(item.item_id)
        return ids

    def _selection_changed(self, source) -> None:
        if self._syncing_selection:
            return
        self._syncing_selection = True
        ids = self._selected_ids_from(source)
        self._selected_ids = ids
        other = self.thumbnail if source is self.table else self.table
        selection_model = other.selectionModel()
        old_block = selection_model.blockSignals(True)
        other.clearSelection()
        wanted = set(ids)
        if source is self.table:
            for item in self.model.visible_items():
                if item.item_id in wanted:
                    self.thumbnail.selectionModel().select(
                        self._index_for_thumb(item), QItemSelectionModel.SelectionFlag.Select
                    )
        else:
            for row, item in enumerate(self.model.visible_items()):
                if item.item_id in wanted:
                    self.table.selectRow(row)
        selection_model.blockSignals(old_block)
        self.selection_note.setText(f"{len(ids)} 件選択中。変更は選択中のすべてに適用")
        self._show_preview()
        self._syncing_selection = False

    def _index_for_thumb(self, item: DataItem):
        for row, value in enumerate(self.thumbnail_model.rows):
            if isinstance(value, DataItem) and value.item_id == item.item_id:
                return self.thumbnail_model.index(row, 0)
        return self.thumbnail_model.index(-1, 0)

    def _edit_selection(self, control) -> None:
        if not self._selected_ids or control.currentIndex() < 0:
            return
        if control is self.usage_combo:
            self._change(self._selected_ids, usage=control.currentData())
        elif control is self.class_combo:
            self._change(self._selected_ids, classification=control.currentData())
        elif control is self.quality_combo:
            self._change(self._selected_ids, quality=control.currentData())
        elif control is self.mask_combo:
            self._change(self._selected_ids, selected_mask_revision=control.currentText())

    def _show_preview(self, *_args) -> None:
        if not self._selected_ids:
            self.image_view.set_image(None)
            return
        item = next((item for item in self.items if item.item_id == self._selected_ids[0]), None)
        if not item:
            return
        self.preview_meta.setText(
            f"{item.item_id}　{item.source_filename}\n取り込み元 {item.source_folder}"
        )
        self.usage_combo.blockSignals(True)
        self.class_combo.blockSignals(True)
        self.quality_combo.blockSignals(True)
        self.mask_combo.blockSignals(True)
        self.usage_combo.setCurrentIndex(self.usage_combo.findData(item.usage))
        self.class_combo.setCurrentIndex(max(0, self.class_combo.findData(item.classification)))
        self.quality_combo.setCurrentIndex(max(0, self.quality_combo.findData(item.quality)))
        self.mask_combo.clear()
        self.mask_combo.addItems(item.mask_revisions)
        self.mask_combo.setCurrentText(item.selected_mask_revision)
        self.usage_combo.blockSignals(False)
        self.class_combo.blockSignals(False)
        self.quality_combo.blockSignals(False)
        self.mask_combo.blockSignals(False)
        channel = self.channel_combo.currentText() or (item.channels[0] if item.channels else "A")
        image = self.ctx.backend.get_item_image("all", item.item_id, channel)
        mode = (
            DisplayMode.IMAGE,
            DisplayMode.OVERLAY,
            DisplayMode.INSTANCE_LABEL,
            DisplayMode.BINARY,
        )[self.display_combo.currentIndex()]
        labels = (
            self.ctx.backend.get_item_mask("all", item.item_id, item.selected_mask_revision)
            if item.mask_revisions
            else None
        )
        self.image_view.set_image(array_to_pixmap(render(image, labels, mode)))
        self.image_caption.setText(f"{self.display_combo.currentText()} ・ チャンネル {channel}")

    def _filter_usage(self, name: str) -> None:
        mapping = {
            "all": None,
            "unassigned": {"unassigned"},
            "train": {"train"},
            "val": {"val"},
            "excluded": {"excluded"},
            "errors": set(),
            "changed": None,
        }
        self.model.usages = mapping[name]
        self.model.errors_only = name == "errors"
        self.model.changed_only = name == "changed"
        self.refresh_views()
        for key in ("all", "unassigned", "train", "val", "excluded", "errors", "changed"):
            self.chips[key].setChecked(key == name)

    @staticmethod
    def _style_chip(chip: CountChip, checked: bool) -> None:
        """テーマの名前付き状態でチップの文字色を切り替える。"""
        state = "activeChipText" if checked else ""
        set_style(chip.name_label, state=state)
        set_style(chip.count_label, state=state)

    def _filter_class(self, value: str) -> None:
        if value in self.model.classifications:
            self.model.classifications.remove(value)
        else:
            self.model.classifications.add(value)
        self.chips[value].setChecked(value in self.model.classifications)
        self.refresh_views()

    def _source_changed(self, index: int) -> None:
        self.model.source_folder = self.source_combo.currentData()
        self.refresh_views()

    def _search_changed(self, text: str) -> None:
        self.model.query = text
        self.refresh_views()

    def refresh_views(self) -> None:
        self.model.layoutChanged.emit()
        self.thumbnail_model.set_items(self.model.visible_items(), self.model.errors)

    def refresh(self) -> None:
        self.items = self.ctx.backend.get_working_items()
        report = self.ctx.backend.validate_items()
        errors = {issue.item_id: issue.message for issue in report.errors}
        self.model.set_items(self.items, errors)
        self.base_label.setText(
            f"ベース版　学習 {self.ctx.backend.working['all'].base_train_version} / "
            f"検証 {self.ctx.backend.working['all'].base_val_version}"
        )
        counts = Counter(item.usage for item in self.items)
        for key, value in (
            ("all", len(self.items)),
            ("unassigned", counts["unassigned"]),
            ("train", counts["train"]),
            ("val", counts["val"]),
            ("excluded", counts["excluded"]),
            ("errors", len(errors)),
            ("changed", sum(item.change is not None for item in self.items)),
        ):
            self.chips[key].count_label.setText(str(value))
        for classification, chip in (
            ("分類A", self.chips["分類A"]),
            ("分類B", self.chips["分類B"]),
            ("分類C", self.chips["分類C"]),
        ):
            chip.count_label.setText(
                str(sum(item.classification == classification for item in self.items))
            )
        selected_folder = self.source_combo.currentData()
        folders = sorted({item.source_folder for item in self.items})
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("取り込み元: すべて", None)
        for folder in folders:
            self.source_combo.addItem(folder, folder)
        self.source_combo.setCurrentIndex(max(0, self.source_combo.findData(selected_folder)))
        self.source_combo.blockSignals(False)
        self.thumbnail_model.set_items(self.model.visible_items(), errors)
        self.finalize_button.setEnabled(not errors)
        self.finalize_button.setToolTip("整合性エラーを解消してください" if errors else "")
        if not self._selected_ids and self.model.visible_items():
            self._selected_ids = [self.model.visible_items()[0].item_id]
        if self._selected_ids:
            first = next(
                (
                    item
                    for item in self.model.visible_items()
                    if item.item_id == self._selected_ids[0]
                ),
                None,
            )
            if first:
                self.table.selectRow(self.model.visible_items().index(first))
                first_index = self._index_for_thumb(first)
                self.thumbnail.selectionModel().select(
                    first_index,
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
                header_row = first_index.row() - 1
                if header_row >= 0 and isinstance(
                    self.thumbnail_model.rows[header_row], FolderHeader
                ):
                    first_index = self.thumbnail_model.index(header_row, 0)
                self.thumbnail.scrollTo(first_index, QListView.ScrollHint.PositionAtTop)
            self._show_preview()

    def import_data(self) -> None:
        dialog = ImportDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            candidates = self.ctx.backend.scan_import_folders(
                dialog.image_dirs, dialog.mask_dir, None
            )
        except ValueError as error:
            QMessageBox.warning(self, "取り込みエラー", str(error))
            return
        folders: dict[str, list[ImportCandidate]] = {}
        for item in candidates:
            folders.setdefault(item.source_relpath.rsplit("/", 1)[0], []).append(item)
        settings = ImportSettingsDialog(self, folders)
        if settings.exec() != QDialog.DialogCode.Accepted:
            return
        self.ctx.backend.import_folders(candidates, settings.values)
        self.model.usages = {"unassigned"}
        self.model.source_folder = next(iter(folders), None)
        self.refresh()

    def auto_triage(self) -> None:
        dialog = AutoTriageDialog(
            self, self.items, selected_ids=self._selected_ids, backend=self.ctx.backend
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.values()
            target_selected = settings.pop("target_selected")
            target_ids = self._selected_ids if target_selected else None
            before = {
                item.item_id: item.usage
                for item in self.items
                if item.usage == "unassigned" and (target_ids is None or item.item_id in target_ids)
            }
            before_flags = {
                item.item_id: (item.change, item.previous_change)
                for item in self.items
                if item.item_id in before
            }
            result = self.ctx.backend.apply_auto_split(settings, target_ids)
            after = {item.item_id: item.usage for item in self.items if item.item_id in before}
            self.undo_stack.push(_TriageCommand(self, before, after, before_flags))
            self._change_status(f"{sum(result.values())} 件を振り分けました")

    def _change_status(self, message: str) -> None:
        self.ctx.status.show_message(message)
        self.refresh()

    def open_triage(self) -> None:
        targets = self.model.visible_items()
        dialog = ContinuousTriageDialog(
            self,
            targets,
            lambda item_id, **changes: self._change([item_id], **changes),
            self.ctx.backend,
        )
        dialog.exec()
        self.refresh()

    def finalize(self) -> None:
        dialog = DatasetFinalizeDialog(self, self.ctx.backend)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            versions = dialog.created
            label = (
                " と ".join(item.version for item in versions) if versions else "変更はありません"
            )
            self._change_status(f"{label} を確定しました")

    def export_excel(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "作業データを Excel に出力", "working_dataset.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        try:
            from openpyxl import Workbook
            from openpyxl.worksheet.datavalidation import DataValidation

            book = Workbook()
            sheet = book.active
            sheet.title = "作業データ"
            sheet.append(
                [
                    "データ識別子",
                    "取り込み元フォルダ",
                    "元ファイル名",
                    "用途",
                    "画像分類",
                    "品質",
                    "マスク版",
                ]
            )
            for item in self.model.visible_items():
                sheet.append(
                    [
                        item.item_id,
                        item.source_folder,
                        item.source_filename,
                        USAGE_TEXT[item.usage],
                        item.classification or "",
                        item.quality or "",
                        item.selected_mask_revision,
                    ]
                )
            for col, formula in (
                (4, '"未振り分け,学習,検証,不採用"'),
                (5, '"分類A,分類B,分類C"'),
                (6, '"良,可,不良"'),
            ):
                rule = DataValidation(type="list", formula1=formula, allow_blank=True)
                sheet.add_data_validation(rule)
                rule.add(f"{chr(64 + col)}2:{chr(64 + col)}{sheet.max_row}")
            book.save(path)
        except ImportError:
            QMessageBox.warning(self, "Excel 出力", "Excel 出力に必要なライブラリがありません")

    def import_excel(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "作業データを Excel から取り込む", "", "Excel (*.xlsx)"
        )
        if not path:
            return
        try:
            from openpyxl import load_workbook

            sheet = load_workbook(path, read_only=True, data_only=True).active
            rows = list(sheet.iter_rows(min_row=2, values_only=True))
            updates = []
            by_id = {item.item_id: item for item in self.items}
            usage_map = {v: k for k, v in USAGE_TEXT.items()}
            for row in rows:
                if row[0] not in by_id:
                    continue
                if row[3] not in usage_map:
                    raise ValueError(f"{row[0]} の用途が正しくありません")
                updates.append(
                    (
                        str(row[0]),
                        {
                            "usage": usage_map[row[3]],
                            "classification": row[4] or None,
                            "quality": row[5] or None,
                            "selected_mask_revision": row[6] or "",
                        },
                    )
                )
            command = _ExcelCommand(self, updates)
            self.undo_stack.push(command)
        except (ImportError, OSError, ValueError) as error:
            QMessageBox.warning(self, "Excel 取込エラー", str(error))

    def open_keymap(self) -> None:
        if KeymapDialog(self, self.shortcuts).exec() == QDialog.DialogCode.Accepted:
            for name, action in self._shortcut_actions.items():
                action.setShortcut(QKeySequence(self.shortcuts[name]))
            self.hints.deleteLater()
            self.hints = KeyHintBar(self.shortcuts.hint_items())
            self.content_layout.addWidget(self.hints)

    def choose_classification(self) -> None:
        """選択項目へ分類をまとめて設定する。"""
        if not self._selected_ids:
            return
        value, accepted = QInputDialog.getItem(
            self, "画像分類", "画像分類", self.ctx.backend.classifications, 0, False
        )
        if accepted:
            self._change(self._selected_ids, classification=value)

    def show_key_help(self) -> None:
        """現在のキー割り当てを表示する。"""
        details = "\n".join(f"{key}　{label}" for key, label in self.shortcuts.hint_items())
        QMessageBox.information(self, "キー操作", details)

    def toggle_view(self) -> None:
        self.views.setCurrentIndex(1 - self.views.currentIndex())

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.KeyPress and watched in (
            self.table,
            self.table.viewport(),
            self.thumbnail,
            self.thumbnail.viewport(),
        ):
            key = event.key()
            key_text = event.text().upper()
            for action, field, value in (
                ("usage_train", "usage", "train"),
                ("usage_val", "usage", "val"),
                ("usage_excluded", "usage", "excluded"),
                ("usage_unassigned", "usage", "unassigned"),
                ("quality_good", "quality", "良"),
                ("quality_ok", "quality", "可"),
                ("quality_bad", "quality", "不良"),
            ):
                if key_text and key_text == self.shortcuts[action].upper() and self._selected_ids:
                    self._change(self._selected_ids, **{field: value})
                    return True
            if key_text == self.shortcuts["toggle_view"].upper():
                self.toggle_view()
                return True
            if key_text == self.shortcuts["class_dialog"].upper() and self._selected_ids:
                self.choose_classification()
                return True
            if key_text == self.shortcuts["help"]:
                self.show_key_help()
                return True
            if key == Qt.Key.Key_Space:
                self.select_next_unassigned()
                return True
            if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
                self.open_triage()
                return True
            if self._selected_ids:
                for index in range(9):
                    if key_text == self.shortcuts[f"class_{index + 1}"]:
                        if index < len(self.ctx.backend.classifications):
                            self._change(
                                self._selected_ids,
                                classification=self.ctx.backend.classifications[index],
                            )
                        return True
        return super().eventFilter(watched, event)

    def select_next_unassigned(self) -> None:
        """次の未振り分け画像へ選択を移す。"""
        visible = self.model.visible_items()
        if not visible:
            return
        current = next(
            (index for index, item in enumerate(visible) if item.item_id in self._selected_ids), -1
        )
        for offset in range(1, len(visible) + 1):
            index = (current + offset) % len(visible)
            if visible[index].usage == "unassigned":
                item = visible[index]
                self._selected_ids = [item.item_id]
                self.table.selectRow(index)
                self.thumbnail.selectionModel().clearSelection()
                self.thumbnail.selectionModel().select(
                    self._index_for_thumb(item),
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
                self._show_preview()
                return

    def on_enter(self, params: dict) -> None:
        self.refresh()


class _ExcelCommand(QUndoCommand):
    """Excel 取込の変更全体を取り消し可能にする。"""

    def __init__(self, page, updates):
        super().__init__("Excel からの変更")
        self.page = page
        self.updates = updates
        self.before = {
            item.item_id: (
                item.usage,
                item.classification,
                item.quality,
                item.selected_mask_revision,
            )
            for item in page.items
            if any(item.item_id == key for key, _ in updates)
        }
        self.before_flags = {
            item.item_id: (item.change, item.previous_change)
            for item in page.items
            if any(item.item_id == key for key, _ in updates)
        }

    def redo(self):
        for item_id, changes in self.updates:
            self.page.ctx.backend.update_item("all", item_id, **changes)
        self.page.refresh()

    def undo(self):
        for item_id, values in self.before.items():
            self.page.ctx.backend.update_item(
                "all",
                item_id,
                usage=values[0],
                classification=values[1],
                quality=values[2],
                selected_mask_revision=values[3],
            )
        self.page._restore_change_flags(self.before_flags)
        self.page.refresh()


class DatasetHistoryPage(BasePage):
    """学習用・検証用を一緒に確認する版履歴ページ。"""

    def __init__(self, ctx: AppContext, parent=None, show_heading: bool = False) -> None:
        super().__init__(
            ctx,
            "データセット版履歴",
            "確定済みの学習用版・検証用版を確認します。",
            parent,
            show_heading,
        )
        self.model = DatasetHistoryModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.verticalHeader().hide()
        self.table.setSortingEnabled(False)
        self.content_layout.addWidget(self.table)
        self.refresh()

    def refresh(self) -> None:
        self.model.set_versions(self.ctx.backend.list_dataset_versions())

    def on_enter(self, params: dict) -> None:
        self.refresh()
