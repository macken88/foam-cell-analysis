"""データ取り込み、自動振り分け、確定の各ダイアログ。"""

from collections import defaultdict

from PySide6.QtCore import QSettings, Qt, QTimer
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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ....services.models import DataItem, ImportCandidate
from ...theme import Color, numeric_font, set_style
from ...widgets.image_convert import DisplayMode, array_to_pixmap, render
from ...widgets.image_view import ImageView


class ImportDialog(QDialog):
    """複数の取り込み元フォルダと画像チャンネルを選ぶ。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("データ取り込み")
        self.setMinimumSize(640, 300)
        self.image_dirs: dict[str, str] = {}
        self.mask_dir = ""
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "画像チャンネルごとのフォルダを選択してください。取り込み後に各フォルダの統一項目を設定します。"
            )
        )
        form = QFormLayout()
        self.channel_rows = {}
        for channel in ("A", "B", "C"):
            edit = QLineEdit()
            browse = QPushButton("参照…")
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(browse)
            browse.clicked.connect(lambda _=False, e=edit: self._browse(e))
            form.addRow(f"チャンネル {channel}", row)
            self.channel_rows[channel] = edit
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
        self.image_dirs = {
            key: edit.text().strip()
            for key, edit in self.channel_rows.items()
            if edit.text().strip()
        }
        self.mask_dir = self.mask_edit.text().strip()
        if self.image_dirs.get("A"):
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
        self.table.horizontalHeader().setStretchLastSection(True)
        self.rows = []
        settings = QSettings("FoamCellAnalysis", "FoamCellAnalysis")
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
        settings = QSettings("FoamCellAnalysis", "FoamCellAnalysis")
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
        saved = (
            QSettings("FoamCellAnalysis", "FoamCellAnalysis").value(
                "dataPreparation/autoTriage", {}
            )
            or {}
        )
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
        self.ratio = QSpinBox()
        self.ratio.setRange(0, 100)
        self.ratio.setSuffix(" %")
        self.ratio.setValue(int(self.settings.get("validation_ratio", 20)))
        self.ratio.setFont(numeric_font())
        self.by_folder = QCheckBox("取り込み元フォルダごと")
        self.by_folder.setChecked(bool(self.settings.get("by_folder", False)))
        self.stratify = QCheckBox("画像分類ごとに割合をそろえる")
        self.stratify.setChecked(bool(self.settings.get("stratify", True)))
        self.bad_to_excluded = QCheckBox("不採用にする")
        self.bad_to_excluded.setChecked(bool(self.settings.get("bad_quality_to_excluded", False)))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(int(self.settings.get("seed", 42)))
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
        self.preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
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
        }

    def _accept(self) -> None:
        """次回起動用に前回の条件を保存する。"""
        QSettings("FoamCellAnalysis", "FoamCellAnalysis").setValue(
            "dataPreparation/autoTriage", self.values()
        )
        self.accept()

    def update_preview(self, *_args) -> None:
        """層別件数と目標との差を再計算する。"""
        target_ids = self.selected_ids if self.target_selected.isChecked() else None
        candidates = [
            item
            for item in self.items
            if item.usage == "unassigned" and (target_ids is None or item.item_id in target_ids)
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
        for row, (classification, values) in enumerate(by_class.items()):
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
            train_now = sum(
                i.usage == "train" and i.classification == classification for i in self.items
            )
            val_now = sum(
                i.usage == "val" and i.classification == classification for i in self.items
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
                self.preview.setItem(row, col, QTableWidgetItem(value))
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
        self.setMinimumSize(720, 420)
        self.backend = backend
        self.created = []
        layout = QVBoxLayout(self)
        self.table = QTableWidget(2, 7)
        self.table.setHorizontalHeaderLabels(
            ["用途", "新しい版", "親版", "件数", "追加", "除外", "変更"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        changed_purposes = []
        for row, purpose in enumerate(("train", "val")):
            summary = backend.summarize_finalize()[purpose]
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
        layout.addWidget(self.table)
        items = backend.get_working_items()
        n_unassigned = sum(item.usage == "unassigned" for item in items)
        layout.addWidget(
            QLabel(
                f"未振り分け {n_unassigned} 件は今回の版に含まれません。"
                "学習用版には同時に発行する検証用版を自動で紐付けます。"
            )
        )
        self.comment = QLineEdit()
        layout.addWidget(QLabel("コメント"))
        layout.addWidget(self.comment)
        self.archive = QCheckBox("確定後にアーカイブを作成する")
        layout.addWidget(self.archive)
        self.errors = backend.validate_items().errors
        self.error_label = QLabel(
            "" if not self.errors else f"整合性エラー {len(self.errors)} 件を解消してください"
        )
        if self.errors:
            set_style(self.error_label, state="error")
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.confirm_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.confirm_button.setText(
            " と ".join(f"{name}用版" for name in changed_purposes) + "を作成"
            if changed_purposes
            else "確定"
        )
        self.confirm_button.setEnabled(not self.errors and bool(changed_purposes))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        self.created = self.backend.finalize_working(
            self.comment.text().strip(), self.archive.isChecked()
        )
        self.accept()


class ContinuousTriageDialog(QDialog):
    """一枚ずつ用途・分類・品質を設定する連続振り分け画面。"""

    def __init__(self, parent, items: list[DataItem], on_update, backend=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("連続振り分け")
        self.source_items = list(items)
        self.items = [item for item in items if item.usage == "unassigned"]
        self.index = 0
        self.on_update = on_update
        self.backend = backend
        self.setMinimumSize(900, 580)
        layout = QVBoxLayout(self)
        self.counter = QLabel()
        self.item_label = QLabel()
        self.item_label.setFont(numeric_font(14))
        layout.addWidget(self.counter)
        layout.addWidget(self.item_label)
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
        body.addWidget(self.image, 1)
        self.filmstrip = QListWidget()
        self.filmstrip.setMaximumWidth(190)
        for item in self.items:
            self.filmstrip.addItem(f"{item.item_id}\n{item.source_filename}")
        self.filmstrip.currentRowChanged.connect(self._select_row)
        body.addWidget(self.filmstrip)
        layout.addLayout(body, 1)
        row = QHBoxLayout()
        for usage, text in (("train", "学習"), ("val", "検証"), ("excluded", "不採用")):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, value=usage: self.assign(value))
            row.addWidget(button)
        layout.addLayout(row)
        meta = QHBoxLayout()
        self.classification = QComboBox()
        self.classification.addItem("未設定", None)
        for value in ("分類A", "分類B", "分類C"):
            self.classification.addItem(value, value)
        self.quality = QComboBox()
        self.quality.addItem("未設定", None)
        for value in ("良", "可", "不良"):
            self.quality.addItem(value, value)
        meta.addWidget(QLabel("分類"))
        meta.addWidget(self.classification)
        meta.addWidget(QLabel("品質"))
        meta.addWidget(self.quality)
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
        self.filmstrip.clear()
        for item in self.items:
            self.filmstrip.addItem(f"{item.item_id}\n{item.source_filename}")
        self._show_item()

    def _edit_metadata(self, field: str, value) -> None:
        if self.items and self.index < len(self.items):
            self.on_update(self.items[self.index].item_id, **{field: value})
            self._show_item()

    def assign(self, usage: str) -> None:
        if self.items:
            item = self.items[self.index]
            self.on_update(item.item_id, usage=usage)
            if self.next_box.isChecked():
                self.index = (self.index + 1) % len(self.items)
            self._show_item()

    def _show_item(self) -> None:
        if self.items:
            item = self.items[self.index]
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
                channel = item.channels[0] if item.channels else "A"
                image = self.backend.get_item_image("all", item.item_id, channel)
                labels = (
                    self.backend.get_item_mask("all", item.item_id, item.selected_mask_revision)
                    if item.mask_revisions
                    else None
                )
                self.image.set_image(array_to_pixmap(render(image, labels, DisplayMode.OVERLAY)))
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
        mapping = {"q": "train", "w": "val", "e": "excluded"}
        usage = mapping.get(event.text().casefold())
        if usage:
            self.assign(usage)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
            event.accept()
            return
        super().keyPressEvent(event)
