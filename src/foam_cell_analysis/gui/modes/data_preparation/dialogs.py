"""データ取り込み、自動振り分け、確定の各ダイアログ。"""

from collections import defaultdict

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ....services.models import DataItem, ImportCandidate
from ...context import DEFAULT_CHANNEL
from ...settings import app_settings
from ...theme import Color, numeric_font, set_style
from ...widgets.image_convert import DisplayMode, array_to_pixmap, render
from ...widgets.image_view import ImageView
from ...widgets.marks import DisplayToggle
from .finalize_thumbnails import (
    FinalizeThumbnailDelegate,
    FinalizeThumbnailModel,
    FinalizeThumbnailView,
    ThumbnailPreviewDialog,
)


class ImportDialog(QDialog):
    """先頭チャンネルの取り込み元フォルダを選ぶ。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("データ取り込み")
        self.setMinimumSize(640, 220)
        self.image_dirs: dict[str, str] = {}
        self.mask_dir = ""
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("画像フォルダを選択してください。取り込み後に各フォルダの統一項目を設定します。")
        )
        form = QFormLayout()
        self.image_edit = QLineEdit()
        browse_image = QPushButton("参照…")
        image_row = QHBoxLayout()
        image_row.addWidget(self.image_edit, 1)
        image_row.addWidget(browse_image)
        browse_image.clicked.connect(lambda: self._browse(self.image_edit))
        form.addRow("画像フォルダ", image_row)
        self.mask_edit = QLineEdit()
        browse_mask = QPushButton("参照…")
        mask_row = QHBoxLayout()
        mask_row.addWidget(self.mask_edit, 1)
        mask_row.addWidget(browse_mask)
        browse_mask.clicked.connect(lambda: self._browse(self.mask_edit))
        form.addRow("マスクフォルダ", mask_row)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("次へ")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if path:
            edit.setText(path)

    def _accept(self) -> None:
        image_dir = self.image_edit.text().strip()
        self.image_dirs = {DEFAULT_CHANNEL: image_dir} if image_dir else {}
        self.mask_dir = self.mask_edit.text().strip()
        if image_dir:
            self.accept()


class ImportSettingsDialog(QDialog):
    """取り込み単位の値を各画像項目へ書き込む。"""

    def __init__(self, parent, folders: dict[str, list[ImportCandidate]]) -> None:
        super().__init__(parent)
        self.setWindowTitle("フォルダごとの統一項目")
        self.setMinimumSize(700, 360)
        self.values: dict[str, dict[str, str | None]] = {}
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("統一する項目にチェックし、フォルダごとの値を選んでください。"))
        self.apply_usage = QCheckBox("用途")
        self.apply_usage.setChecked(True)
        self.apply_classification = QCheckBox("画像分類")
        self.apply_classification.setChecked(True)
        self.apply_quality = QCheckBox("品質")
        self.apply_quality.setChecked(True)
        options = QHBoxLayout()
        options.addWidget(self.apply_usage)
        options.addWidget(self.apply_classification)
        options.addWidget(self.apply_quality)
        options.addStretch(1)
        layout.addLayout(options)
        self.table = QTableWidget(len(folders), 5)
        self.table.setHorizontalHeaderLabels(["フォルダ", "画像数", "用途", "画像分類", "品質"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.resizeColumnsToContents()
        self.rows = []
        settings = app_settings()
        for row, (folder, candidates) in enumerate(folders.items()):
            self.table.setItem(row, 0, QTableWidgetItem(folder))
            self.table.setItem(row, 1, QTableWidgetItem(str(len(candidates))))
            usage = QComboBox()
            usage.addItem("未振り分け", "unassigned")
            usage.addItem("学習", "train")
            usage.addItem("検証", "val")
            usage.addItem("不採用", "excluded")
            classification = QComboBox()
            classification.addItem("未設定", "")
            for value in ("分類A", "分類B", "分類C"):
                classification.addItem(value, value)
            quality = QComboBox()
            quality.addItem("未設定", "")
            for value in ("良", "可", "不良"):
                quality.addItem(value, value)
            defaults = settings.value("dataPreparation/lastImport", {}) or {}
            usage.setCurrentIndex(max(0, usage.findData(defaults.get("usage", "unassigned"))))
            classification.setCurrentIndex(
                max(0, classification.findData(defaults.get("classification", "")))
            )
            quality.setCurrentIndex(max(0, quality.findData(defaults.get("quality", ""))))
            self.table.setCellWidget(row, 2, usage)
            self.table.setCellWidget(row, 3, classification)
            self.table.setCellWidget(row, 4, quality)
            self.rows.append((folder, usage, classification, quality))
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("選択分を取り込む")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        settings = app_settings()
        last = settings.value("dataPreparation/lastImport", {}) or {}
        for folder, usage, classification, quality in self.rows:
            values = {}
            if self.apply_usage.isChecked():
                values["usage"] = usage.currentData()
            if self.apply_classification.isChecked():
                values["classification"] = classification.currentData() or None
            if self.apply_quality.isChecked():
                values["quality"] = quality.currentData() or None
            self.values[folder] = values
            if self.apply_usage.isChecked():
                last["usage"] = usage.currentData()
            if self.apply_classification.isChecked():
                last["classification"] = classification.currentData()
            if self.apply_quality.isChecked():
                last["quality"] = quality.currentData()
        settings.setValue("dataPreparation/lastImport", last)
        self.accept()


class AutoTriageDialog(QDialog):
    """設定変更をプレビュー表へ即時反映する自動振り分け画面。"""

    def __init__(
        self,
        parent,
        items: list[DataItem],
        settings: dict | None = None,
        selected_ids: list[str] | None = None,
        backend=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("自動振り分け")
        self.setMinimumSize(860, 520)
        self.items = items
        self.backend = backend
        saved = app_settings().value("dataPreparation/autoTriage", {}) or {}
        self.settings = {**saved, **(settings or {})}
        self.selected_ids = set(selected_ids or [])
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.target_unassigned = QRadioButton(
            f"未振り分け（{sum(item.usage == 'unassigned' for item in items)}件）"
        )
        self.target_unassigned.setChecked(True)
        self.target_selected = QRadioButton(f"選択中（{len(self.selected_ids)}件）")
        self.target_selected.setEnabled(bool(self.selected_ids))
        target_row = QHBoxLayout()
        target_row.addWidget(self.target_unassigned)
        target_row.addWidget(self.target_selected)
        target_row.addStretch(1)
        form.addRow("対象", target_row)
        self.selected_note = QLabel("選択中を対象にした場合、現在の用途を上書きして振り分けます。")
        self.selected_note.setVisible(bool(self.selected_ids))
        form.addRow("", self.selected_note)
        self.ratio = QSpinBox()
        self.ratio.setRange(0, 100)
        self.ratio.setSuffix(" %")
        self.ratio.setValue(int(self.settings.get("validation_ratio", 20)))
        self.ratio.setFont(numeric_font())
        self.ratio.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.by_folder = QCheckBox("取り込み元フォルダごと")
        self.by_folder.setChecked(bool(self.settings.get("by_folder", False)))
        self.stratify = QCheckBox("画像分類ごとに割合をそろえる")
        self.stratify.setChecked(bool(self.settings.get("stratify", True)))
        self.bad_to_excluded = QCheckBox("不採用にする")
        self.bad_to_excluded.setChecked(bool(self.settings.get("bad_quality_to_excluded", False)))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(int(self.settings.get("seed", 42)))
        self.seed.setFont(numeric_font())
        self.seed.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        form.addRow("検証の割合", self.ratio)
        form.addRow("振り分け単位", self.by_folder)
        form.addRow("層化", self.stratify)
        form.addRow("品質が「不良」のもの", self.bad_to_excluded)
        form.addRow("乱数シード", self.seed)
        layout.addLayout(form)
        self.preview = QTableWidget(0, 4)
        self.preview.setHorizontalHeaderLabels(
            ["分類", "学習（現在 → 実行後）", "検証（現在 → 実行後）", "検証の割合"]
        )
        self.preview.verticalHeader().hide()
        self.preview.horizontalHeader().setStretchLastSection(False)
        self.preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.preview.setColumnWidth(0, 125)
        self.preview.setColumnWidth(1, 245)
        self.preview.setColumnWidth(2, 245)
        self.preview.setColumnWidth(3, 130)
        layout.addWidget(self.preview)
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        for widget in (self.ratio, self.by_folder, self.stratify, self.bad_to_excluded, self.seed):
            signal = widget.valueChanged if isinstance(widget, QSpinBox) else widget.toggled
            signal.connect(self.update_preview)
        self.target_unassigned.toggled.connect(self.update_preview)
        self.target_selected.toggled.connect(self.update_preview)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.apply_button.setText("振り分ける")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.update_preview()

    def values(self) -> dict:
        """現在の設定を辞書で返す。"""
        return {
            "validation_ratio": self.ratio.value(),
            "by_folder": self.by_folder.isChecked(),
            "stratify": self.stratify.isChecked(),
            "bad_quality_to_excluded": self.bad_to_excluded.isChecked(),
            "seed": self.seed.value(),
            "target_selected": self.target_selected.isChecked(),
            "include_assigned": self.target_selected.isChecked(),
        }

    def _accept(self) -> None:
        """次回起動用に前回の条件を保存する。"""
        app_settings().setValue("dataPreparation/autoTriage", self.values())
        self.accept()

    def update_preview(self, *_args) -> None:
        """層別件数と目標との差を再計算する。"""
        target_ids = self.selected_ids if self.target_selected.isChecked() else None
        candidates = [
            item
            for item in self.items
            if (target_ids is not None or item.usage == "unassigned")
            and (target_ids is None or item.item_id in target_ids)
        ]
        by_class: dict[str, list[DataItem]] = defaultdict(list)
        for item in candidates:
            by_class[item.classification or "未設定"].append(item)
        self.preview.setRowCount(len(by_class))
        ratio = self.ratio.value()
        settings = self.values()
        settings.pop("target_selected", None)
        assignments = (
            self.backend.preview_auto_split_assignments(
                settings, list(target_ids) if target_ids is not None else None
            )
            if self.backend
            else {}
        )
        classification_order = [*self.backend.classifications, "未設定"] if self.backend else []
        ordered_classes = [
            (name, by_class[name]) for name in classification_order if name in by_class
        ]
        ordered_classes.extend(
            (name, values) for name, values in by_class.items() if name not in classification_order
        )
        for row, (classification, values) in enumerate(ordered_classes):
            train_after = (
                sum(assignments.get(item.item_id) == "train" for item in values)
                if assignments
                else len(values) - round(len(values) * ratio / 100)
            )
            val_after = (
                sum(assignments.get(item.item_id) == "val" for item in values)
                if assignments
                else round(len(values) * ratio / 100)
            )
            replaced_ids = target_ids or set()
            train_now = sum(
                i.usage == "train"
                and i.classification == classification
                and i.item_id not in replaced_ids
                for i in self.items
            )
            val_now = sum(
                i.usage == "val"
                and i.classification == classification
                and i.item_id not in replaced_ids
                for i in self.items
            )
            train_total = train_now + train_after
            val_total = val_now + val_after
            after_ratio = (
                val_total / (train_total + val_total) * 100 if train_total + val_total else 0
            )
            vals = (
                classification,
                f"{train_now} → {train_total}",
                f"{val_now} → {val_total}",
                f"{after_ratio:.1f}%",
            )
            for col, value in enumerate(vals):
                cell = QTableWidgetItem(value)
                if col > 0:
                    cell.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                    cell.setFont(numeric_font())
                self.preview.setItem(row, col, cell)
            if abs(after_ratio - ratio) > 10:
                self.preview.item(row, 3).setBackground(QColor(Color.CHANGED))
        result = (
            self.backend.preview_auto_split(
                settings, list(target_ids) if target_ids is not None else None
            )
            if self.backend
            else {}
        )
        if result:
            self.count_label.setText(
                f"対象 {len(candidates)} 件　実行後: 学習 {result.get('train', 0)} 件・"
                f"検証 {result.get('val', 0)} 件・不採用 {result.get('excluded', 0)} 件"
            )
        else:
            self.count_label.setText(f"対象 {len(candidates)} 件を振り分けます")
        self.apply_button.setText(f"{len(candidates)}件を振り分ける")


class DatasetFinalizeDialog(QDialog):
    """学習・検証の変更版を一括確定する。"""

    def __init__(self, parent, backend) -> None:
        super().__init__(parent)
        self.setWindowTitle("データセットを確定")
        self.setMinimumSize(960, 680)
        self.backend = backend
        self.created = []
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        self.finalize_tabs = tabs
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        self.table = QTableWidget(2, 7)
        self.table.setHorizontalHeaderLabels(
            ["用途", "新しい版", "親版", "件数", "追加", "除外", "変更"]
        )
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.resizeColumnsToContents()
        changed_purposes = []
        summaries = backend.summarize_finalize()
        for row, purpose in enumerate(("train", "val")):
            summary = summaries[purpose]
            label = "学習" if purpose == "train" else "検証"
            base = backend.get_working_dataset(purpose).base_version
            has_changes = bool(summary.get("has_changes", True))
            if has_changes:
                changed_purposes.append(label)
            new_version = (
                str(summary["next_version"]) if has_changes else f"変更なし（{base} のまま）"
            )
            values = (
                label,
                new_version,
                base,
                str(summary["n_images"]),
                f"+{summary['added']}" if has_changes else "0",
                str(summary["removed"]) if has_changes else "0",
                str(summary["changed"]) if has_changes else "0",
            )
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
        overview_layout.addWidget(self.table)
        items = backend.get_working_items()
        n_unassigned = sum(item.usage == "unassigned" for item in items)
        overview_layout.addWidget(
            QLabel(
                f"未振り分け {n_unassigned} 件は今回の版に含まれません。"
                "学習用版には同時に発行する検証用版を自動で紐付けます。"
            )
        )
        self.comment = QLineEdit()
        overview_layout.addWidget(QLabel("コメント"))
        overview_layout.addWidget(self.comment)
        self.archive = QCheckBox("確定後にアーカイブを作成する")
        overview_layout.addWidget(self.archive)
        self.errors = backend.validate_items().errors
        self.error_label = QLabel(
            "" if not self.errors else f"整合性エラー {len(self.errors)} 件を解消してください"
        )
        if self.errors:
            set_style(self.error_label, state="error")
        overview_layout.addWidget(self.error_label)
        tabs.addTab(overview, "概要")
        self.thumbnail_filter = QComboBox()
        self.thumbnail_filter.addItems(["追加・変更のみ", "すべて", "⚠ のあるもの"])
        thumbnail_page = QWidget()
        thumbnail_layout = QVBoxLayout(thumbnail_page)
        thumbnail_layout.addWidget(
            QLabel("今回の版に入る画像を用途ごとに確認できます。画像をクリックすると拡大します。")
        )
        thumbnail_layout.addWidget(self.thumbnail_filter)
        self.thumbnail_tabs = QTabWidget()
        self.thumbnail_models = {}
        self.thumbnail_views = {}
        self.thumbnail_stacks = {}
        self.thumbnail_empty_labels = {}
        self.thumbnail_source_items = {}
        working_items = backend.get_working_items()
        item_errors = {issue.item_id: issue.message for issue in self.errors}
        for purpose, label in (("train", "学習用"), ("val", "検証用")):
            items_for_purpose = [item for item in working_items if item.usage == purpose]
            self.thumbnail_source_items[purpose] = items_for_purpose
            model = FinalizeThumbnailModel(backend, self)
            model.set_items(items_for_purpose, item_errors)
            view = FinalizeThumbnailView()
            view.setModel(model)
            view.setItemDelegate(FinalizeThumbnailDelegate(view))
            view.setUniformItemSizes(True)
            model.attach_view(view)
            view.clicked.connect(lambda index, source=model: self._preview_item(source, index))
            self.thumbnail_models[purpose] = model
            self.thumbnail_views[purpose] = view
            stack = QStackedWidget()
            empty_label = QLabel()
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            set_style(empty_label, role="note", state="idle")
            stack.addWidget(view)
            stack.addWidget(empty_label)
            self.thumbnail_stacks[purpose] = stack
            self.thumbnail_empty_labels[purpose] = empty_label
            self.thumbnail_tabs.addTab(stack, f"{label} ({len(items_for_purpose)} 件)")
        self.thumbnail_filter.currentIndexChanged.connect(self._filter_thumbnails)
        self._filter_thumbnails(0)
        thumbnail_layout.addWidget(self.thumbnail_tabs, 1)
        tabs.addTab(thumbnail_page, "サムネイルで確認")
        layout.addWidget(tabs, 1)
        buttons = QHBoxLayout()
        self.error_summary_button = QPushButton()
        self.error_summary_button.setVisible(bool(self.errors))
        self.error_summary_button.clicked.connect(self._show_error_rows)
        buttons.addWidget(self.error_summary_button)
        buttons.addStretch(1)
        self.confirm_button = QPushButton()
        self.confirm_button.setText(
            " と ".join(f"{name}用版" for name in changed_purposes) + "を作成"
            if changed_purposes
            else "確定"
        )
        self.confirm_button.setEnabled(not self.errors and bool(changed_purposes))
        self.cancel_button = QPushButton("キャンセル")
        self.confirm_button.clicked.connect(self._accept)
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.confirm_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self._update_error_summary()

    def _update_error_summary(self) -> None:
        """無効な確定操作の理由を件数付きで示す。"""
        count = len(self.errors)
        self.error_summary_button.setText(f"⚠ エラー {count} 件を直すと確定できます")
        set_style(self.error_summary_button, usage="error")
        self.error_summary_button.setVisible(count > 0)

    def _show_error_rows(self) -> None:
        """確定ダイアログを閉じ、エラー行だけを表示する。"""
        self.reject()
        parent = self.parent()
        if hasattr(parent, "_filter_usage"):
            parent._filter_usage("errors")

    def _filter_thumbnails(self, index: int) -> None:
        """用途ごとの一覧へ選択中の絞り込みを適用する。"""
        filter_name = ("changed", "all", "errors")[index]
        for purpose, model in self.thumbnail_models.items():
            model.filter_name = filter_name
            model.set_items(self.thumbnail_source_items[purpose], model.errors)
            visible_count = model.rowCount()
            total_count = len(self.thumbnail_source_items[purpose])
            label = "学習用" if purpose == "train" else "検証用"
            if filter_name == "all":
                self.thumbnail_tabs.setTabText(
                    self.thumbnail_tabs.indexOf(self.thumbnail_stacks[purpose]),
                    f"{label} ({total_count} 件)",
                )
            else:
                self.thumbnail_tabs.setTabText(
                    self.thumbnail_tabs.indexOf(self.thumbnail_stacks[purpose]),
                    f"{label} ({visible_count} / {total_count} 件)",
                )
            if visible_count:
                self.thumbnail_stacks[purpose].setCurrentIndex(0)
            else:
                empty_message = {
                    "changed": "追加・変更された画像はありません",
                    "errors": "整合性エラーのある画像はありません",
                    "all": "表示できる画像はありません",
                }[filter_name]
                self.thumbnail_empty_labels[purpose].setText(empty_message)
                self.thumbnail_stacks[purpose].setCurrentIndex(1)

    def _preview_item(self, model, index) -> None:
        """選択したサムネイルを簡易拡大表示する。"""
        item = model.data(index, Qt.ItemDataRole.UserRole)
        if isinstance(item, DataItem):
            ThumbnailPreviewDialog(self, self.backend, item).exec()

    def _accept(self) -> None:
        self.created = self.backend.finalize_working(
            self.comment.text().strip(), self.archive.isChecked()
        )
        self.accept()


class ContinuousTriageDialog(QDialog):
    """一枚ずつ用途・分類・品質を設定する連続振り分け画面。"""

    def __init__(
        self,
        parent,
        items: list[DataItem],
        on_update,
        backend=None,
        shortcuts=None,
        display_preference=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("連続振り分け")
        self.source_items = list(items)
        self.items = [item for item in items if item.usage == "unassigned"]
        self.index = 0
        self.on_update = on_update
        self.backend = backend
        self.shortcuts = shortcuts
        self.display_preference = display_preference or parent.ctx.display
        self.display_mode = self._configured_display_mode()
        self.current_item_id: str | None = None
        self.setMinimumSize(900, 580)
        layout = QVBoxLayout(self)
        self.counter = QLabel()
        self.item_label = QLabel()
        self.item_label.setFont(numeric_font(14))
        layout.addWidget(self.counter)
        layout.addWidget(self.item_label)
        self.display_toggle = DisplayToggle(self.display_preference)
        self.display_toggle.alternate_selected.connect(self._toggle_mode)
        layout.addWidget(self.display_toggle)
        if shortcuts:
            shortcuts.changed.connect(self._shortcuts_changed)
        self.target_unassigned = QRadioButton("未振り分けのみ")
        self.target_unassigned.setChecked(True)
        self.target_filtered = QRadioButton("絞り込み中のすべて")
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("対象"))
        target_row.addWidget(self.target_unassigned)
        target_row.addWidget(self.target_filtered)
        target_row.addStretch(1)
        layout.addLayout(target_row)
        body = QHBoxLayout()
        self.image = ImageView()
        self.image.setMinimumSize(580, 380)
        self.image.installEventFilter(self)
        self.image.viewport().installEventFilter(self)
        body.addWidget(self.image, 1)
        self.filmstrip = QListWidget()
        self.filmstrip.setMaximumWidth(190)
        for item in self.items:
            self.filmstrip.addItem(f"{item.item_id}\n{item.source_filename}")
        self.filmstrip.currentRowChanged.connect(self._select_row)
        body.addWidget(self.filmstrip)
        layout.addLayout(body, 1)
        row = QHBoxLayout()
        self.usage_buttons = {}
        for usage, text, action in (
            ("train", "学習", "usage_train"),
            ("val", "検証", "usage_val"),
            ("excluded", "不採用", "usage_excluded"),
            ("unassigned", "未振り分け", "usage_unassigned"),
        ):
            button = QPushButton(text)
            self.usage_buttons[usage] = (button, action, text)
            button.clicked.connect(lambda _=False, value=usage: self.assign(value))
            row.addWidget(button)
        layout.addLayout(row)
        meta = QHBoxLayout()
        meta.setSpacing(20)
        self.classification = QComboBox()
        self.classification.addItem("未設定", None)
        for value in ("分類A", "分類B", "分類C"):
            self.classification.addItem(value, value)
        self.quality = QComboBox()
        self.quality.addItem("未設定", None)
        for value in ("良", "可", "不良"):
            self.quality.addItem(value, value)
        for label, combo in (("分類", self.classification), ("品質", self.quality)):
            pair = QHBoxLayout()
            pair.setSpacing(6)
            pair.addWidget(QLabel(label))
            pair.addWidget(combo)
            meta.addLayout(pair)
        meta.addStretch(1)
        layout.addLayout(meta)
        self.classification.currentIndexChanged.connect(
            lambda: self._edit_metadata("classification", self.classification.currentData())
        )
        self.quality.currentIndexChanged.connect(
            lambda: self._edit_metadata("quality", self.quality.currentData())
        )
        self.next_box = QCheckBox("振り分けたら次へ進む")
        self.next_box.setChecked(True)
        layout.addWidget(self.next_box)
        self.target_unassigned.toggled.connect(lambda checked: checked and self._set_target(False))
        self.target_filtered.toggled.connect(lambda checked: checked and self._set_target(True))
        if shortcuts:
            self._shortcuts_changed()
            self.display_preference.changed.connect(self._display_changed)
        cancel = QPushButton("閉じる")
        cancel.clicked.connect(self.accept)
        layout.addWidget(cancel)
        self._show_item()

    def _select_row(self, row: int) -> None:
        if 0 <= row < len(self.items):
            self.index = row
            self._show_item()

    def _set_target(self, include_assigned: bool) -> None:
        """対象ラジオに応じてフィルムストリップを切り替える。"""
        self.items = (
            list(self.source_items)
            if include_assigned
            else [item for item in self.source_items if item.usage == "unassigned"]
        )
        self.index = 0
        self._refresh_filmstrip()
        self._show_item()

    def _refresh_filmstrip(self) -> None:
        """対象画像に合わせてフィルムストリップを作り直す。"""
        self.filmstrip.clear()
        for item in self.items:
            self.filmstrip.addItem(f"{item.item_id}\n{item.source_filename}")

    def _edit_metadata(self, field: str, value) -> None:
        if self.items and self.index < len(self.items):
            self.on_update(self.items[self.index].item_id, **{field: value})
            self._show_item()

    def _shortcuts_changed(self) -> None:
        """共有キー変更を用途ボタンへ反映する。"""
        if hasattr(self, "usage_buttons"):
            for button, action, text in self.usage_buttons.values():
                key = self.shortcuts.display_key(self.shortcuts[action])
                button.setText(f"{text}　{key}")

    def assign(self, usage: str) -> None:
        if self.items:
            item = self.items[self.index]
            current_id = item.item_id
            self.on_update(item.item_id, usage=usage)
            if self.next_box.isChecked():
                if self.target_unassigned.isChecked() and self.backend:
                    source_ids = {source.item_id for source in self.source_items}
                    self.source_items = [
                        candidate
                        for candidate in self.backend.get_working_items()
                        if candidate.item_id in source_ids
                    ]
                    self.items = [
                        candidate
                        for candidate in self.source_items
                        if candidate.usage == "unassigned"
                    ]
                    self._refresh_filmstrip()
                    next_index = next(
                        (
                            index
                            for index, candidate in enumerate(self.items)
                            if candidate.item_id != current_id
                        ),
                        None,
                    )
                    self.index = next_index if next_index is not None else 0
                elif self.items:
                    self.index = (self.index + 1) % len(self.items)
            self.current_item_id = current_id
            self._show_item()

    def _show_item(self) -> None:
        if self.items:
            item = self.items[self.index]
            self.current_item_id = item.item_id
            remaining = sum(i.usage == "unassigned" for i in self.items)
            self.counter.setText(f"未振り分け 残り {remaining} / {len(self.items)}")
            item_label = f"{item.item_id}　"
            item_label += item.source_filename
            self.item_label.setText(item_label)
            for combo, value in (
                (self.classification, item.classification),
                (self.quality, item.quality),
            ):
                combo.blockSignals(True)
                combo.setCurrentIndex(max(0, combo.findData(value)))
                combo.blockSignals(False)
            if self.backend:
                channel = DEFAULT_CHANNEL
                self.current_channel = channel
                image = self.backend.get_item_image("all", item.item_id, channel)
                labels = (
                    self.backend.get_item_mask("all", item.item_id, item.selected_mask_revision)
                    if item.mask_revisions
                    else None
                )
                self.image.set_image(array_to_pixmap(render(image, labels, self.display_mode)))
                QTimer.singleShot(0, self.image.fit_image)
            if self.filmstrip.currentRow() != self.index:
                self.filmstrip.blockSignals(True)
                self.filmstrip.setCurrentRow(self.index)
                self.filmstrip.blockSignals(False)
        else:
            self.counter.setText("対象画像はありません")
            self.item_label.clear()
            self.image.set_image(None)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """用途キーで分類し、Enter または Esc で戻る。"""
        if self.shortcuts:
            for action, usage in (
                ("usage_train", "train"),
                ("usage_val", "val"),
                ("usage_excluded", "excluded"),
                ("usage_unassigned", "unassigned"),
            ):
                if self.shortcuts.matches(action, event):
                    self.assign(usage)
                    event.accept()
                    return
        if self._handle_view_shortcut(event):
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if (
            watched in (self.image, self.image.viewport())
            and event.type() == QEvent.Type.KeyPress
            and self._handle_view_shortcut(event)
        ):
            return True
        return super().eventFilter(watched, event)

    def _configured_display_mode(self) -> DisplayMode:
        """共有設定の表示名を描画モードへ変換する。"""
        return {
            "オーバーレイ": DisplayMode.OVERLAY,
            "インスタンスラベル": DisplayMode.INSTANCE_LABEL,
            "二値マスク": DisplayMode.BINARY,
        }[self.display_preference.value]

    def _toggle_mode(self, alternate: bool) -> None:
        self.display_mode = self._configured_display_mode() if alternate else DisplayMode.IMAGE
        self._show_item()

    def _display_changed(self, _name: str) -> None:
        """共有表示名の変更を現在の画像へ反映する。"""
        if self.display_toggle.is_alternate:
            self.display_mode = self._configured_display_mode()
            self._show_item()

    def _shortcuts_changed(self) -> None:
        """用途ボタンに現在のキー割り当てを反映する。"""
        if not self.shortcuts:
            return
        for button, action, text in self.usage_buttons.values():
            key = self.shortcuts.display_key(self.shortcuts[action])
            button.setText(f"{text}　{key}")
            button.setToolTip(f"{text}（{key}）")

    def _handle_view_shortcut(self, event: QKeyEvent) -> bool:
        """二択表示とズームのショートカットを処理する。"""
        if self.shortcuts and self.shortcuts.matches("display_mode", event):
            self.display_toggle.set_alternate(not self.display_toggle.is_alternate)
            event.accept()
            return True
        if self.shortcuts and self.shortcuts.matches("previous_image", event):
            self._select_row(max(0, self.index - 1))
            event.accept()
            return True
        if self.shortcuts and self.shortcuts.matches("next_image", event):
            self._select_row(min(len(self.items) - 1, self.index + 1))
            event.accept()
            return True
        if self.shortcuts and self.shortcuts.matches("zoom_in", event):
            self.image.zoom_by(1.2)
            event.accept()
            return True
        if self.shortcuts and self.shortcuts.matches("zoom_out", event):
            self.image.zoom_by(1 / 1.2)
            event.accept()
            return True
        if self.shortcuts and self.shortcuts.matches("fit_view", event):
            self.image.fit_image()
            event.accept()
            return True
        return False
