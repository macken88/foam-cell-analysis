"""データ準備モードの操作ダイアログ。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ....services.backend import Backend
from ....services.models import DatasetVersion, ImportCandidate
from ...theme import numeric_font, set_style
from ...widgets.table import mark_primary, setup_table


class FolderRow(QWidget):
    """フォルダパスと参照ボタンの入力行。"""

    def __init__(self, parent: QWidget, title: str, directory: bool = True) -> None:
        super().__init__(parent)
        self.directory = directory
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.browse_button = QPushButton("参照…")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_button)
        self.browse_button.clicked.connect(lambda: self.choose(title))

    @property
    def path(self) -> str:
        return self.path_edit.text()

    def choose(self, title: str) -> None:
        if self.directory:
            path = QFileDialog.getExistingDirectory(self, title)
        else:
            path, _ = QFileDialog.getOpenFileName(self, title)
        if path:
            self.path_edit.setText(path)


class ImportDialog(QDialog):
    """取り込み候補を検索して選択する。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("データ取り込み")
        self.setMinimumSize(800, 560)
        self.resize(840, 600)
        self.selected_candidates: list[ImportCandidate] = []
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.image_dirs = {channel: FolderRow(self, f"チャンネル{channel}") for channel in "ABC"}
        for channel, row in self.image_dirs.items():
            form.addRow(f"画像フォルダ {channel}{'（必須）' if channel == 'A' else ''}", row)
        self.mask_dir = FolderRow(self, "マスクフォルダ")
        form.addRow("マスクフォルダ", self.mask_dir)
        self.classification_combo = QComboBox()
        self.classification_combo.addItems(["未設定", "分類A", "分類B", "分類C"])
        form.addRow("画像分類の初期値", self.classification_combo)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.scan_button = QPushButton("検索")
        buttons.addWidget(self.scan_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["選択", "ファイル名", "チャンネル", "マスク"])
        setup_table(self.table, stretch_column=1)
        layout.addWidget(self.table, 1)
        footer = QHBoxLayout()
        footer.addStretch(1)
        cancel = QPushButton("キャンセル")
        self.import_button = QPushButton("選択分を取り込む")
        mark_primary(self.import_button)
        self.import_button.setEnabled(False)
        footer.addWidget(cancel)
        footer.addWidget(self.import_button)
        layout.addLayout(footer)
        cancel.clicked.connect(self.reject)
        self.scan_button.clicked.connect(self.scan)
        self.import_button.clicked.connect(self.accept_selection)

    @property
    def classification(self) -> str | None:
        value = self.classification_combo.currentText()
        return None if value == "未設定" else value

    def scan(self) -> None:
        image_dirs = {key: row.path for key, row in self.image_dirs.items() if row.path}
        if "A" not in image_dirs:
            QMessageBox.warning(
                self, "入力が必要です", "チャンネルAの画像フォルダを指定してください。"
            )
            return
        try:
            candidates = self.parent().ctx.backend.scan_import_source(
                image_dirs, self.mask_dir.path
            )
        except ValueError as error:
            QMessageBox.warning(self, "検索できません", str(error))
            return
        self._candidates = candidates
        self.table.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            checkbox = QTableWidgetItem()
            checkbox.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(row, 0, checkbox)
            self.table.setItem(row, 1, QTableWidgetItem(candidate.source_filename))
            self.table.setItem(row, 2, QTableWidgetItem(", ".join(candidate.channels)))
            self.table.setItem(
                row, 3, QTableWidgetItem("あり" if candidate.mask_available else "なし")
            )
        self.import_button.setEnabled(bool(candidates))

    def accept_selection(self) -> None:
        self.selected_candidates = [
            candidate
            for row, candidate in enumerate(getattr(self, "_candidates", []))
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked
        ]
        if self.selected_candidates:
            self.accept()


class ExcelExportDialog(QDialog):
    """Excel出力先を選ぶモックダイアログ。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Excel出力")
        self.setMinimumSize(520, 220)
        self.output_path = ""
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.destination = FolderRow(self, "Excel出力先")
        layout.addWidget(QLabel("出力先フォルダを選択してください。実ファイルは作成しません。"))
        form.addRow("保存先フォルダ", self.destination)
        layout.addLayout(form)
        footer = _footer(self, "出力")
        mark_primary(footer.itemAt(2).widget())
        layout.addLayout(footer)
        footer.itemAt(2).widget().clicked.connect(self.accept_export)

    def accept_export(self) -> None:
        if self.destination.path:
            self.output_path = self.destination.path
            self.accept()


class ExcelImportDialog(QDialog):
    """Excel取込の事前確認と適用可否を扱う。"""

    def __init__(self, parent: QWidget, backend: Backend, purpose: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Excel取込")
        self.setMinimumSize(760, 470)
        self.backend = backend
        self.purpose = purpose
        self.approved_changes: list[dict[str, Any]] = []
        self._changes: list[dict[str, Any]] = []
        layout = QVBoxLayout(self)
        self.file_row = FolderRow(self, "Excelファイル", directory=False)
        layout.addWidget(QLabel("ファイル名に error を含む場合はエラーとなり、反映しません。"))
        form = QFormLayout()
        form.addRow("Excelファイル", self.file_row)
        layout.addLayout(form)
        self.preview_button = QPushButton("取込前チェック")
        layout.addWidget(self.preview_button)
        self.result_label = QLabel("ファイルを選択して取込前チェックを実行してください。")
        layout.addWidget(self.result_label)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["データ識別子", "列", "変更前", "変更後"])
        setup_table(self.table, stretch_column=3)
        layout.addWidget(self.table, 1)
        footer = _footer(self, "変更を取り込む")
        self.apply_button = footer.itemAt(2).widget()
        mark_primary(self.apply_button)
        self.apply_button.setEnabled(False)
        layout.addLayout(footer)
        self.preview_button.clicked.connect(self.preview)
        self.apply_button.clicked.connect(self.apply_preview)

    def preview(self) -> None:
        filename = self.file_row.path
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Excelファイルを選択", "", "Excel (*.xlsx *.xlsm)"
            )
            if filename:
                self.file_row.path_edit.setText(filename)
        if not filename:
            return
        result = self.backend.preview_excel_import(filename)
        self._changes = list(result.get("changes", []))
        self.table.setRowCount(0)
        if not result.get("ok"):
            errors = "、".join(result.get("errors", []))
            self.result_label.setText(f"エラーのため中断しました: {errors}")
            self.apply_button.setEnabled(False)
            return
        self.table.setRowCount(len(self._changes))
        for row, change in enumerate(self._changes):
            values = [change["item_id"], change["field"], change.get("before", ""), change["after"]]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setFont(numeric_font())
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, column, item)
        self.result_label.setText(f"変更予定 {len(self._changes)} 件")
        self.apply_button.setEnabled(bool(self._changes))

    def apply_preview(self) -> None:
        self.approved_changes = list(self._changes)
        if self.approved_changes:
            self.accept()


class DatasetFinalizeDialog(QDialog):
    """検証済み作業版の確定内容を提示する。"""

    def __init__(self, parent: QWidget, backend: Backend, purpose: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("新しいデータセットを作成")
        self.setMinimumSize(480, 300)
        self.resize(520, 320)
        self.backend = backend
        self.purpose = purpose
        self.version: DatasetVersion | None = None
        self.summary = backend.summarize_working_changes(purpose)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        dataset = backend.get_working_dataset(purpose)
        next_version = str(self.summary["next_version"])
        form.addRow("用途", QLabel("学習用" if purpose == "train" else "検証用"))
        form.addRow("版", _readonly(next_version))
        form.addRow("親版", _readonly(dataset.base_version))
        form.addRow(
            "画像数・変更",
            QLabel(
                f"{self.summary['n_images']}　追加 +{self.summary['added']}　"
                f"除外 -{self.summary['removed']}　変更 {self.summary['changed']}"
            ),
        )
        self.validation_combo = QComboBox()
        self.validation_combo.addItem("選択なし", None)
        if purpose == "train":
            validation_versions = backend.list_validation_versions()
            for record in validation_versions:
                self.validation_combo.addItem(record.version, record.version)
            if validation_versions:
                latest = validation_versions[-1].version
                self.validation_combo.setCurrentIndex(self.validation_combo.findData(latest))
            form.addRow("基準検証用データセット", self.validation_combo)
        self.duplicate_label = QLabel("")
        self.duplicate_label.setWordWrap(True)
        form.addRow("重複チェック", self.duplicate_label)
        self.comment_edit = QLineEdit()
        form.addRow("コメント", self.comment_edit)
        layout.addLayout(form)
        footer = _footer(self, f"{next_version} を作成")
        self.create_button = footer.itemAt(2).widget()
        mark_primary(self.create_button)
        layout.addLayout(footer)
        self.validation_combo.currentIndexChanged.connect(self.check_duplicates)
        self.create_button.clicked.connect(self.accept)
        self.check_duplicates()

    @property
    def base_validation_version(self) -> str | None:
        return self.validation_combo.currentData() if self.purpose == "train" else None

    def check_duplicates(self) -> list[tuple[str, str]]:
        duplicates = self.backend.check_dataset_duplicates(
            self.purpose, self.base_validation_version
        )
        if duplicates:
            set_style(self.duplicate_label, state="error")
            self.duplicate_label.setText(
                f"検証用データセットとの重複 {len(duplicates)} 件\n"
                + "\n".join(f"{item_id}（{reason}）" for item_id, reason in duplicates)
            )
        else:
            set_style(self.duplicate_label, state="ok")
            self.duplicate_label.setText("検証用データセットとの重複 0 件　✔")
        self.create_button.setEnabled(not duplicates)
        return duplicates

    def apply(self) -> DatasetVersion:
        """入力内容を Backend に適用し、作成版を返す。"""
        if self.check_duplicates():
            raise ValueError("検証用データセットと重複しています")
        self.version = self.backend.finalize_dataset(
            self.purpose,
            self.comment_edit.text().strip(),
            self.base_validation_version,
        )
        return self.version


class ArchiveDialog(QDialog):
    """アーカイブ出力先を選択する。"""

    def __init__(self, parent: QWidget, version: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("アーカイブ作成")
        self.setMinimumSize(560, 230)
        self.resize(600, 260)
        self.version = version
        self.output_path = ""
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(f"{version} のアーカイブ出力先を選択してください。実ファイルは作りません。")
        )
        self.destination = FolderRow(self, "出力先")
        form = QFormLayout()
        form.addRow("出力先フォルダ", self.destination)
        layout.addLayout(form)
        footer = _footer(self, "作成")
        mark_primary(footer.itemAt(2).widget())
        layout.addLayout(footer)
        footer.itemAt(2).widget().clicked.connect(self.accept_archive)

    def accept_archive(self) -> None:
        if self.destination.path:
            self.output_path = self.destination.path
            self.accept()


class MaskRevisionDialog(QDialog):
    """既存マスクを上書きせず新しい版として取り込む。"""

    def __init__(self, parent: QWidget, filename: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("新しいマスク版を取り込む")
        self.setMinimumSize(580, 240)
        self.resize(620, 280)
        self.mask_path = ""
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(f"対象: {filename}\n選択したファイルを新しいマスク版として記録します。")
        )
        self.source = FolderRow(self, "マスクファイル", directory=False)
        form = QFormLayout()
        form.addRow("マスクファイル", self.source)
        layout.addLayout(form)
        footer = _footer(self, "取り込む")
        mark_primary(footer.itemAt(2).widget())
        layout.addLayout(footer)
        footer.itemAt(2).widget().clicked.connect(self.accept_mask)

    def accept_mask(self) -> None:
        if self.source.path:
            self.mask_path = self.source.path
            self.accept()


def _footer(dialog: QDialog, accept_label: str) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.addStretch(1)
    cancel = QPushButton("キャンセル")
    accept = QPushButton(accept_label)
    cancel.clicked.connect(dialog.reject)
    layout.addWidget(cancel)
    layout.addWidget(accept)
    return layout


def _readonly(value: str) -> QLabel:
    label = QLabel(value)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    set_style(label, role="readonly")
    return label
