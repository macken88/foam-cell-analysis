"""データ準備画面。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ....services.models import DataItem, ImportCandidate, ValidationReport
from ...context import AppContext
from ...jobs import FakeJob
from ...labels import format_datetime
from ...widgets.form import FormSection
from ...widgets.image_convert import DisplayMode, array_to_pixmap, render
from ...widgets.image_view import ImageView
from ...widgets.page_base import BasePage
from ...widgets.table import mark_primary, setup_table
from .dialogs import (
    ArchiveDialog,
    DatasetFinalizeDialog,
    ExcelExportDialog,
    ExcelImportDialog,
    ImportDialog,
    MaskRevisionDialog,
)
from .table_model import (
    CheckTableModel,
    DataItemModel,
    ErrorTableModel,
    HistoryTableModel,
    ItemComboDelegate,
    ItemFilterProxy,
)


class DataPreparationPage(BasePage):
    """作業データの編集、検証、確定を行う。"""

    classifications = ["すべて", "分類A", "分類B", "分類C", "未設定"]
    qualities = ["すべて", "良", "可", "不良", "未設定"]
    display_modes = {
        "原画像": DisplayMode.IMAGE,
        "オーバーレイ": DisplayMode.OVERLAY,
        "インスタンスラベル": DisplayMode.INSTANCE_LABEL,
    }

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(
            ctx,
            "データ準備",
            "作業データを編集し、整合性を確認してデータセット版を確定します。",
            parent,
        )
        self.purpose = "train"
        self.dataset = None
        self.selected_item_id: str | None = None
        self.last_validation: ValidationReport | None = None
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        top = QHBoxLayout()
        self.purpose_combo = QComboBox()
        self.purpose_combo.addItem("学習用", "train")
        self.purpose_combo.addItem("検証用", "val")
        self.purpose_combo.currentIndexChanged.connect(self._change_purpose)
        top.addWidget(QLabel("対象:"))
        top.addWidget(self.purpose_combo)
        self.state_label = QLabel()
        self.saved_label = QLabel()
        top.addSpacing(20)
        top.addWidget(QLabel("状態:"))
        top.addWidget(self.state_label)
        top.addStretch(1)
        top.addWidget(self.saved_label)
        self.content_layout.addLayout(top)

        summary = FormSection("現在の作業中データセット")
        summary_line = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        summary_line.addWidget(self.summary_label, 1)
        summary.add_row("概要", self.summary_label)
        buttons = QHBoxLayout()
        self.import_button = QPushButton("データ取り込み…")
        self.export_button = QPushButton("Excel出力…")
        self.excel_button = QPushButton("Excel取込…")
        self.validate_button = QPushButton("整合性確認を実行")
        self.finalize_button = QPushButton("データセットを確定…")
        mark_primary(self.validate_button)
        mark_primary(self.finalize_button)
        for button in (
            self.import_button,
            self.export_button,
            self.excel_button,
            self.validate_button,
            self.finalize_button,
        ):
            buttons.addWidget(button)
        summary.add_row("操作", _layout_widget(buttons))
        self.content_layout.addWidget(summary)
        self.import_button.clicked.connect(self._open_import)
        self.export_button.clicked.connect(self._open_export)
        self.excel_button.clicked.connect(self._open_excel_import)
        self.validate_button.clicked.connect(self.start_validation)
        self.finalize_button.clicked.connect(self._open_finalize)

        self.tabs = QTabWidget()
        self.content_layout.addWidget(self.tabs, 1)
        self._build_working_tab()
        self._build_validation_tab()
        self._build_history_tab()

    def _build_working_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        filters = QHBoxLayout()
        self.class_filter = _combo(self.classifications)
        self.quality_filter = _combo(self.qualities)
        self.changed_filter = QCheckBox("変更ありのみ")
        self.errors_filter = QCheckBox("エラーありのみ")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("識別子・ファイル名を検索")
        for label, widget in (
            ("画像分類", self.class_filter),
            ("品質", self.quality_filter),
            ("", self.changed_filter),
            ("", self.errors_filter),
            ("検索", self.search_edit),
        ):
            if label:
                filters.addWidget(QLabel(label))
            filters.addWidget(widget)
        layout.addLayout(filters)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.item_model = DataItemModel(self._edit_item)
        self.proxy = ItemFilterProxy()
        self.proxy.setSourceModel(self.item_model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        setup_table(
            self.table,
            stretch_column=2,
            selection_mode=QTableView.SelectionMode.ExtendedSelection,
        )
        self.table.setItemDelegateForColumn(3, ItemComboDelegate(self.table))
        self.table.setItemDelegateForColumn(4, ItemComboDelegate(self.table))
        self.table.setItemDelegateForColumn(6, ItemComboDelegate(self.table))
        self.table.setSortingEnabled(True)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        split.addWidget(self.table)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_controls = QHBoxLayout()
        self.channel_combo = _combo(["A", "B", "C"])
        self.display_combo = _combo(list(self.display_modes))
        self.mask_combo = QComboBox()
        preview_controls.addWidget(QLabel("チャンネル"))
        preview_controls.addWidget(self.channel_combo)
        preview_controls.addWidget(QLabel("表示形式"))
        preview_controls.addWidget(self.display_combo)
        preview_layout.addLayout(preview_controls)
        self.image_view = ImageView()
        preview_layout.addWidget(self.image_view, 1)
        self.item_info = QLabel("行を選択するとプレビューを表示します")
        self.item_info.setWordWrap(True)
        preview_layout.addWidget(self.item_info)
        self.mask_button = QPushButton("新しいマスク版を取り込む…")
        preview_layout.addWidget(QLabel("表示するマスク版"))
        preview_layout.addWidget(self.mask_combo)
        preview_layout.addWidget(self.mask_button)
        self.mask_button.clicked.connect(self._open_mask_revision)
        self.channel_combo.currentTextChanged.connect(self._refresh_preview)
        self.display_combo.currentTextChanged.connect(self._refresh_preview)
        self.mask_combo.currentTextChanged.connect(self._refresh_preview)
        split.addWidget(preview)
        split.setStretchFactor(0, 6)
        split.setStretchFactor(1, 4)
        layout.addWidget(split, 1)

        actions = QHBoxLayout()
        self.exclude_button = QPushButton("選択を除外")
        self.include_button = QPushButton("選択を採用に戻す")
        self.bulk_class_button = QPushButton("画像分類を一括設定")
        self.bulk_quality_button = QPushButton("品質を一括設定")
        for button in (
            self.exclude_button,
            self.include_button,
            self.bulk_class_button,
            self.bulk_quality_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.exclude_button.clicked.connect(lambda: self._bulk_update(included=False))
        self.include_button.clicked.connect(lambda: self._bulk_update(included=True))
        self.bulk_class_button.clicked.connect(lambda: self._bulk_choose("classification"))
        self.bulk_quality_button.clicked.connect(lambda: self._bulk_choose("quality"))
        self.tabs.addTab(tab, "作業中データ")
        self.class_filter.currentTextChanged.connect(self._apply_filters)
        self.quality_filter.currentTextChanged.connect(self._apply_filters)
        self.changed_filter.toggled.connect(self._apply_filters)
        self.errors_filter.toggled.connect(self._apply_filters)
        self.search_edit.textChanged.connect(self._apply_filters)
        self.table.selectionModel().selectionChanged.connect(self._update_action_state)

    def _build_validation_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.checklist = QTableView()
        self.checklist.setMaximumHeight(210)
        self.check_model = CheckTableModel()
        self.checklist.setModel(self.check_model)
        setup_table(self.checklist, stretch_column=0)
        layout.addWidget(QLabel("整合性確認項目"))
        layout.addWidget(self.checklist)
        self.error_table = QTableView()
        self.error_model = ErrorTableModel(self._error_double_clicked)
        self.error_table.setModel(self.error_model)
        setup_table(self.error_table, stretch_column=3)
        self.error_table.doubleClicked.connect(lambda index: self.error_model.activate(index.row()))
        layout.addWidget(QLabel("エラー一覧"))
        layout.addWidget(self.error_table, 1)
        self.tabs.addTab(tab, "整合性確認")

    def _build_history_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        split = QSplitter(Qt.Orientation.Vertical)
        self.history_model = HistoryTableModel()
        self.history_table = QTableView()
        self.history_table.setModel(self.history_model)
        setup_table(self.history_table, stretch_column=7)
        self.history_table.selectionModel().selectionChanged.connect(self._update_archive_button)
        split.addWidget(self.history_table)
        self.version_detail = QLabel("版を選択すると詳細を表示します")
        self.version_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.version_detail.setWordWrap(True)
        split.addWidget(self.version_detail)
        layout.addWidget(split, 1)
        self.archive_button = QPushButton("アーカイブ作成…")
        self.archive_button.setEnabled(False)
        self.archive_button.clicked.connect(self._open_archive)
        layout.addWidget(self.archive_button)
        self.history_table.selectionModel().selectionChanged.connect(self._show_version_detail)
        self.tabs.addTab(tab, "データセット版履歴")

    def on_enter(self, params: dict[str, Any]) -> None:
        """ページ遷移時に対象データを再読込する。"""
        self.refresh()

    def refresh(self) -> None:
        """Backend の最新状態を表示する。"""
        self.dataset = self.ctx.backend.get_working_dataset(self.purpose)
        self.state_label.setText(
            {"WORKING": "作業中", "VALIDATED": "検証済み"}.get(
                self.dataset.state, self.dataset.state
            )
        )
        self.saved_label.setText(f"自動保存済み {format_datetime(self.dataset.last_saved_at)}")
        summary = self.ctx.backend.summarize_working_changes(self.purpose)
        error_count = len(self.dataset.validation.errors) if self.dataset.validation else 0
        self.summary_label.setText(
            f"ベース版: {self.dataset.base_version}　登録画像数: {summary['n_images']}　"
            f"マスク数: {summary['n_masks']}\n未入力メタデータ: {summary['missing_metadata']}　"
            f"整合性エラー: {error_count}　"
            f"追加 +{summary['added']}　除外 -{summary['removed']}　変更 {summary['changed']}"
        )
        self.finalize_button.setEnabled(self.dataset.state == "VALIDATED")
        self.last_validation = self.dataset.validation
        self.item_model.set_items(self.dataset.items, self.dataset.validation)
        self.table.resizeColumnsToContents()
        for column, width in enumerate((46, 95, 120, 80, 55, 65, 75, 50, 50)):
            self.table.setColumnWidth(column, width)
        self._apply_filters()
        if self.item_model.rowCount() and not self.table.selectionModel().selectedRows():
            first_item_id = self.item_model.item_id_at(0)
            if first_item_id:
                self._select_item(first_item_id)
        self._populate_validation(self.dataset.validation)
        self.history_model.set_versions(self.ctx.backend.list_dataset_versions(self.purpose))
        self._update_action_state()
        self._update_archive_button()

    def _change_purpose(self) -> None:
        self.purpose = str(self.purpose_combo.currentData())
        self.selected_item_id = None
        self.refresh()

    def _edit_item(self, item_id: str, changes: dict[str, Any]) -> None:
        self.ctx.backend.update_item(self.purpose, item_id, **changes)
        self.ctx.status.notify_saved(self.ctx.backend.get_last_saved_at(self.purpose))
        self.refresh()
        self._select_item(item_id)

    def _apply_filters(self) -> None:
        self.proxy.set_filters(
            classification=self.class_filter.currentText(),
            quality=self.quality_filter.currentText(),
            changed_only=self.changed_filter.isChecked(),
            errors_only=self.errors_filter.isChecked(),
            search_text=self.search_edit.text().strip().casefold(),
        )

    def _selected_ids(self) -> list[str]:
        ids = set()
        for index in self.table.selectionModel().selectedRows():
            source = self.proxy.mapToSource(index)
            item_id = self.item_model.item_id_at(source.row())
            if item_id:
                ids.add(item_id)
        return sorted(ids)

    def _bulk_update(self, **changes: Any) -> None:
        item_ids = self._selected_ids()
        if not item_ids:
            return
        self.ctx.backend.bulk_update_items(self.purpose, item_ids, **changes)
        self.ctx.status.notify_saved(self.ctx.backend.get_last_saved_at(self.purpose))
        self.refresh()

    def _bulk_choose(self, field: str) -> None:
        choices = self.classifications[1:] if field == "classification" else self.qualities[1:]
        from PySide6.QtWidgets import QInputDialog

        value, ok = QInputDialog.getItem(self, "一括設定", "設定値", choices, 0, False)
        if ok:
            self._bulk_update(**{field: None if value == "未設定" else value})

    def _selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            source = self.proxy.mapToSource(rows[0])
            self.selected_item_id = self.item_model.item_id_at(source.row())
        else:
            self.selected_item_id = None
        self._refresh_preview()
        self._update_action_state()

    def _update_action_state(self, *_args: Any) -> None:
        selected = self._selected_ids() if hasattr(self, "table") else []
        selected_items = [
            item
            for item in (self.dataset.items if self.dataset else [])
            if item.item_id in selected
        ]
        self.exclude_button.setEnabled(any(item.included for item in selected_items))
        self.include_button.setEnabled(any(not item.included for item in selected_items))
        self.bulk_class_button.setEnabled(bool(selected))
        self.bulk_quality_button.setEnabled(bool(selected))
        self.mask_button.setEnabled(len(selected) == 1)

    def _item(self) -> DataItem | None:
        if not self.selected_item_id or self.dataset is None:
            return None
        return next(
            (item for item in self.dataset.items if item.item_id == self.selected_item_id), None
        )

    def _refresh_preview(self, *_args: Any) -> None:
        item = self._item()
        if item is None:
            self.image_view.set_image(None)
            return
        channels = item.channels or ["A"]
        channel = self.channel_combo.currentText()
        if channel not in channels:
            channel = channels[0]
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(["A", "B", "C"])
        self.channel_combo.setCurrentText(channel)
        self.channel_combo.blockSignals(False)
        self.mask_combo.blockSignals(True)
        self.mask_combo.clear()
        self.mask_combo.addItems(item.mask_revisions)
        self.mask_combo.setCurrentText(item.selected_mask_revision)
        self.mask_combo.blockSignals(False)
        try:
            image = self.ctx.backend.get_item_image(self.purpose, item.item_id, channel)
            mask = (
                self.ctx.backend.get_item_mask(
                    self.purpose, item.item_id, self.mask_combo.currentText()
                )
                if self.mask_combo.count()
                else None
            )
            mode = self.display_modes[self.display_combo.currentText()]
            self.image_view.set_image(array_to_pixmap(render(image, mask, mode)))
        except (KeyError, ValueError):
            self.image_view.set_image(None)
        self.item_info.setText(
            f"{item.item_id}　SHA-256: {item.sha256[:16]}\n元の相対パス: {item.source_relpath}"
        )

    def _open_mask_revision(self) -> None:
        item = self._item()
        if item is None:
            return
        dialog = MaskRevisionDialog(self, item.source_filename)
        if dialog.exec() and dialog.mask_path:
            revision = self.ctx.backend.add_mask_revision(self.purpose, item.item_id)
            self.ctx.status.notify_saved(self.ctx.backend.get_last_saved_at(self.purpose))
            self.refresh()
            self.mask_combo.setCurrentText(revision)

    def _populate_validation(self, report: ValidationReport | None) -> None:
        checks = report.checks if report else []
        issues = report.errors if report else []
        self.check_model.set_checks(checks)
        self.error_model.set_issues(issues)

    def start_validation(self) -> None:
        """FakeJob で整合性確認を行う。"""
        self.validate_button.setEnabled(False)
        self.ctx.status.show_message("整合性確認を実行中です")
        job = FakeJob("整合性確認", total_steps=3, interval_ms=800)

        def done(_ok: bool, message: str) -> None:
            self.validate_button.setEnabled(True)
            self.last_validation = self.ctx.backend.validate_working_dataset(self.purpose)
            self.refresh()
            self.ctx.status.show_message(message)

        job.finished.connect(done)
        self.ctx.jobs.start(job)

    def _error_double_clicked(self, item_id: str) -> None:
        self.tabs.setCurrentIndex(0)
        self._select_item(item_id)

    def _select_item(self, item_id: str) -> None:
        row = self.item_model.row_for_id(item_id)
        if row < 0:
            return
        proxy_index = self.proxy.mapFromSource(self.item_model.index(row, 0))
        if proxy_index.isValid():
            self.table.selectRow(proxy_index.row())
            self.table.scrollTo(proxy_index)

    def _open_import(self) -> None:
        dialog = ImportDialog(self)
        if not dialog.exec() or not dialog.selected_candidates:
            return
        candidates = dialog.selected_candidates
        classification = dialog.classification
        job = FakeJob("データ取り込み", 3, 500)
        job.finished.connect(
            lambda ok, msg: self._finish_import(ok, msg, candidates, classification)
        )
        self.ctx.jobs.start(job)

    def _finish_import(
        self,
        ok: bool,
        message: str,
        candidates: list[ImportCandidate],
        classification: str | None,
    ) -> None:
        if ok:
            self.ctx.backend.import_items(self.purpose, candidates, classification)
            self.ctx.status.notify_saved(self.ctx.backend.get_last_saved_at(self.purpose))
            self.refresh()
        self.ctx.status.show_message(message)

    def _open_export(self) -> None:
        dialog = ExcelExportDialog(self)
        if dialog.exec() and dialog.output_path:
            QMessageBox.information(
                self, "Excel出力", "出力しました（モック）。実ファイルは作成していません。"
            )

    def _open_excel_import(self) -> None:
        dialog = ExcelImportDialog(self, self.ctx.backend, self.purpose)
        if dialog.exec() and dialog.approved_changes:
            self.ctx.backend.apply_excel_import(self.purpose, dialog.approved_changes)
            self.ctx.status.notify_saved(self.ctx.backend.get_last_saved_at(self.purpose))
            self.refresh()

    def _open_finalize(self) -> None:
        if self.dataset.state != "VALIDATED":
            return
        dialog = DatasetFinalizeDialog(self, self.ctx.backend, self.purpose)
        if not dialog.exec():
            return
        version = dialog.apply()
        self.ctx.status.show_message(f"{version.version} を確定しました")
        self.refresh()
        answer = QMessageBox.question(self, "アーカイブ作成", "続けてアーカイブを作成しますか？")
        if answer == QMessageBox.StandardButton.Yes:
            self._run_archive(version.version)

    def _run_archive(self, version: str) -> None:
        dialog = ArchiveDialog(self, version)
        if not dialog.exec() or not dialog.output_path:
            return
        job = FakeJob("アーカイブ作成", 4, 700)

        def done(ok: bool, message: str) -> None:
            result = self.ctx.backend.record_archive_result(version, dialog.output_path, ok)
            archive_state = HistoryTableModel.archive_status_labels.get(
                result.archive_status, result.archive_status
            )
            self.ctx.status.show_message(
                f"{result.version}.zip: {archive_state}　SHA-256: {result.archive_sha256 or 'なし'}"
            )
            self.refresh()

        job.finished.connect(done)
        self.ctx.jobs.start(job)

    def _show_version_detail(self, *_args: Any) -> None:
        row = self.history_table.currentIndex().row()
        version = self.history_model.version_at(row)
        if version is None:
            return
        archive_status = HistoryTableModel.archive_status_labels.get(
            version.archive_status, version.archive_status
        )
        self.version_detail.setText(
            f"版: {version.version}\n用途: {'学習用' if version.purpose == 'train' else '検証用'}"
            f"\n親版: {version.parent_version or '—'}"
            f"\n作成日時: {format_datetime(version.created_at)}"
            f"\n画像数: {version.n_images}"
            f"\n状態: {'確定済み' if version.status == 'RELEASED' else version.status}"
            f"\nアーカイブ: "
            f"{archive_status}"
            f"\nSHA-256: {version.archive_sha256 or '—'}"
            f"\n基準検証版: {version.base_validation_version or '—'}"
            f"\nコメント: {version.comment or '—'}"
        )

    def _update_archive_button(self, *_args: Any) -> None:
        if not hasattr(self, "archive_button"):
            return
        selected = self.history_table.selectionModel().selectedRows()
        self.archive_button.setEnabled(len(selected) == 1)

    def _open_archive(self) -> None:
        row = self.history_table.currentIndex().row()
        version = self.history_model.version_at(row)
        if version:
            self._run_archive(version.version)


def _combo(values: list[str]) -> QComboBox:
    combo = QComboBox()
    combo.addItems(values)
    return combo


def _layout_widget(layout: QHBoxLayout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget
