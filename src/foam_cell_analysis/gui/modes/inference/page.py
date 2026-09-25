"""本番推論画面。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...jobs import FakeJob
from ...navigation import PageId
from ...theme import Color, numeric_font, set_style
from ...widgets.form import FormSection
from ...widgets.image_convert import DisplayMode, array_to_pixmap, render
from ...widgets.image_view import ImageView
from ...widgets.marks import STATUS_MARKS, TagDelegate
from ...widgets.page_base import BasePage
from ...widgets.table import mark_primary, setup_table


@dataclass
class InferenceInput:
    """推論対象の画像と行状態。"""

    filename: str
    classification: str
    model_id: str
    status: str = "待機"
    image: object | None = None
    labels: object | None = None


class InferencePage(BasePage):
    """分類別のリリースモデルで画像を推論する。"""

    classifications = ("分類A", "分類B", "分類C")
    modes = {
        "オーバーレイ": DisplayMode.OVERLAY,
        "インスタンスラベル": DisplayMode.INSTANCE_LABEL,
        "二値マスク": DisplayMode.BINARY,
    }

    def __init__(self, ctx, parent=None, *, show_heading: bool = True) -> None:
        super().__init__(
            ctx,
            "本番推論",
            "画像分類に応じたリリース済みモデルで推論します。",
            parent,
            show_heading=show_heading,
        )
        self.inputs: list[InferenceInput] = []
        self.output_path = ""
        self._build_ui()
        self.refresh_routing()

    def _build_ui(self) -> None:
        input_section = FormSection("入力画像")
        controls = QHBoxLayout()
        self.add_files_button = QPushButton("画像を追加…")
        self.add_folder_button = QPushButton("フォルダを追加…")
        self.remove_button = QPushButton("選択を削除")
        controls.addWidget(self.add_files_button)
        controls.addWidget(self.add_folder_button)
        controls.addWidget(self.remove_button)
        controls.addStretch(1)
        input_section.form.addRow(controls)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["ファイル名", "画像分類", "適用モデル", "状態", "検出数"]
        )
        setup_table(
            self.table,
            stretch_column=0,
            selection_mode=QTableWidget.SelectionMode.ExtendedSelection,
        )
        self.table.setItemDelegateForColumn(
            3,
            TagDelegate({**STATUS_MARKS, "待機": (Color.IDLE_BG, Color.SLATE, False)}, self.table),
        )
        input_section.form.addRow(self.table)

        self.output_section = FormSection("出力設定")
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.path_label = QLabel("未選択")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.browse_output_button = QPushButton("参照…")
        path_layout.addWidget(self.path_label, 1)
        path_layout.addWidget(self.browse_output_button)
        self.output_section.add_row("出力先", path_row)
        self.instance_check = QCheckBox("インスタンスラベルマスク")
        self.instance_check.setChecked(True)
        self.binary_check = QCheckBox("粒子解析用二値マスク")
        self.binary_check.setChecked(True)
        self.overlay_check = QCheckBox("オーバーレイ画像も保存")
        self.output_section.add_row(
            "保存内容",
            self._checkbox_row(self.instance_check, self.binary_check, self.overlay_check),
        )
        format_row = QWidget()
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        self.png_radio = QRadioButton("PNG")
        self.tiff_radio = QRadioButton("TIFF")
        self.png_radio.setChecked(True)
        format_layout.addWidget(self.png_radio)
        format_layout.addWidget(self.tiff_radio)
        format_layout.addStretch(1)
        self.output_section.add_row("ファイル形式", format_row)
        self.run_button = QPushButton("推論を実行")
        self.run_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        mark_primary(self.run_button)
        actions = QWidget()
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addStretch(1)
        action_layout.addWidget(self.run_button)
        self.output_section.add_row("", actions)

        preview_section = FormSection("結果プレビュー")
        self.display_group = QButtonGroup(self)
        self.display_buttons: dict[str, QPushButton] = {}
        display_row = QWidget()
        display_layout = QHBoxLayout(display_row)
        display_layout.setContentsMargins(0, 0, 0, 0)
        for index, label in enumerate(self.modes):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            set_style(button, role="segment")
            if index == 0:
                button.setChecked(True)
            self.display_group.addButton(button)
            self.display_buttons[label] = button
            display_layout.addWidget(button)
        display_layout.addStretch(1)
        preview_section.add_row("", display_row)
        self.image_view = ImageView()
        self.image_view.setMinimumSize(360, 320)
        preview_section.add_row("", self.image_view)
        self.detection_count = QLabel("検出数: —")
        self.image_view.set_overlay_labels("", "")
        preview_section.form.addRow(self.detection_count)

        self.error_banner = QWidget()
        set_style(self.error_banner, role="errorBanner")
        error_layout = QHBoxLayout(self.error_banner)
        error_layout.setContentsMargins(8, 4, 8, 4)
        self.error_text = QLabel()
        self.route_button = QPushButton("振り分け画面へ")
        error_layout.addWidget(self.error_text, 1)
        error_layout.addWidget(self.route_button)
        self.error_banner.hide()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.addWidget(input_section, 2)
        left_layout.addWidget(self.error_banner)
        left_layout.addWidget(self.output_section, 1)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.addWidget(preview_section)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        self.content_layout.addWidget(splitter, 1)

        self.add_files_button.clicked.connect(self.choose_images)
        self.add_folder_button.clicked.connect(self.choose_folder)
        self.remove_button.clicked.connect(self.remove_selected)
        self.browse_output_button.clicked.connect(self.choose_output)
        self.run_button.clicked.connect(self.run_inference)
        self.route_button.clicked.connect(self.open_routing)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.display_group.buttonToggled.connect(
            lambda _button, checked: checked and self._render_selected()
        )

    @staticmethod
    def _checkbox_row(*checks: QCheckBox) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        for check in checks:
            layout.addWidget(check)
        return widget

    def on_enter(self, params: dict) -> None:
        """ページ表示時に最新のモデル振り分けを反映する。"""
        self.refresh_routing()

    def refresh_routing(self) -> None:
        """入力行の適用モデルを現在の振り分けで更新する。"""
        self.routing = self.ctx.backend.get_routing()
        self.models = {model.model_id: model for model in self.ctx.backend.list_released_models()}
        for entry in self.inputs:
            entry.model_id = self.routing.get(entry.classification) or ""
        self._refresh_table()

    def add_images(self, paths: list[str | Path]) -> None:
        """ファイルダイアログなしで画像ファイル名を入力に追加する。"""
        for path in paths:
            filename = Path(path).name
            if filename:
                self.inputs.append(InferenceInput(filename, self.classifications[0], ""))
                self.inputs[-1].model_id = self.routing.get(self.classifications[0]) or ""
        self._refresh_table()

    def choose_images(self) -> None:
        """複数の画像ファイルを選択して追加する。"""
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "画像を追加", "", "画像ファイル (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        self.add_images(filenames)

    def choose_folder(self) -> None:
        """フォルダ内の画像ファイルを追加する。"""
        folder = QFileDialog.getExistingDirectory(self, "画像フォルダを追加")
        if folder:
            paths = sorted(
                path
                for path in Path(folder).iterdir()
                if path.is_file()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            )
            self.add_images(paths)

    def choose_output(self) -> None:
        """出力先フォルダを選択する。"""
        folder = QFileDialog.getExistingDirectory(self, "出力先フォルダを選択")
        if folder:
            self.output_path = folder
            self.path_label.setText(folder)

    def remove_selected(self) -> None:
        """選択行を入力一覧から削除する。"""
        rows = sorted(
            {index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True
        )
        for row in rows:
            del self.inputs[row]
        self._refresh_table()

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.inputs))
        for row, entry in enumerate(self.inputs):
            self.table.setItem(row, 0, QTableWidgetItem(entry.filename))
            classification = QComboBox()
            classification.addItems(self.classifications)
            classification.setCurrentText(entry.classification)
            classification.currentTextChanged.connect(
                lambda value, index=row: self._classification_changed(index, value)
            )
            self.table.setCellWidget(row, 1, classification)
            model_item = QTableWidgetItem(entry.model_id or "未割り当て")
            model_item.setFont(numeric_font())
            model_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            model_item.setFlags(model_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            model_item.setToolTip("画像分類とモデル振り分けから自動決定")
            self.table.setItem(row, 2, model_item)
            status_item = QTableWidgetItem(entry.status)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 3, status_item)
            count = len(set(entry.labels.ravel()) - {0}) if entry.labels is not None else None
            count_item = QTableWidgetItem(str(count) if count is not None else "—")
            count_item.setFont(numeric_font())
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 4, count_item)
        self.table.resizeColumnsToContents()
        self._update_run_state()

    def _classification_changed(self, row: int, value: str) -> None:
        if row >= len(self.inputs):
            return
        entry = self.inputs[row]
        entry.classification = value
        entry.model_id = self.routing.get(value) or ""
        model_item = self.table.item(row, 2)
        if model_item:
            model_item.setText(entry.model_id or "未割り当て")
        self._update_run_state()
        self._render_selected()

    def _update_run_state(self) -> None:
        missing = [entry for entry in self.inputs if not entry.model_id]
        self.run_button.setEnabled(bool(self.inputs) and not missing)
        runnable = sum(bool(entry.model_id) for entry in self.inputs)
        self.run_button.setText(f"推論を実行（{runnable} 枚）")
        self.error_banner.setVisible(bool(missing))
        missing_types = sorted({entry.classification for entry in missing})
        if missing:
            filenames = "、".join(entry.filename for entry in missing[:3])
            suffix = " ほか" if len(missing) > 3 else ""
            self.error_text.setText(
                f"{'、'.join(missing_types)} にモデルが未割り当てのため、"
                f"{filenames}{suffix} は推論できません。"
            )
        else:
            self.error_text.clear()
        self.run_button.setToolTip("モデル振り分けが未設定の画像があります" if missing else "")

    def run_inference(self) -> None:
        """入力画像を順に処理する FakeJob を開始する。"""
        if not self.inputs:
            return
        if not self.output_path:
            QMessageBox.warning(
                self, "出力先未選択", "推論結果の出力先フォルダを選択してください。"
            )
            self.ctx.status.show_message("推論結果の出力先フォルダを選択してください")
            return
        missing = [entry for entry in self.inputs if not entry.model_id]
        if missing:
            QMessageBox.warning(
                self, "モデル未割り当て", "未割り当て分類があります。振り分け画面へ進んでください。"
            )
            return
        for entry in self.inputs:
            entry.status = "待機"
            entry.image = None
            entry.labels = None
        current_index = 0

        def advance(step: int) -> None:
            nonlocal current_index
            row_index = (step - 1) // 2
            if step % 2 == 1:
                current_index = row_index
                entry = self.inputs[current_index]
                entry.status = "実行中"
                entry.image, entry.labels = self.ctx.backend.get_inference_result(
                    entry.filename, entry.model_id
                )
                self._refresh_table()
                self.table.selectRow(current_index)
            else:
                self.inputs[row_index].status = "完了"
                current_index = row_index + 1
                self._refresh_table()
                if current_index < len(self.inputs):
                    self.table.selectRow(current_index)
                self._update_run_state()

        job = FakeJob(
            "本番推論", total_steps=len(self.inputs) * 2, interval_ms=200, on_step=advance
        )
        job.finished.connect(lambda ok, message: self._job_finished(ok, message))
        self.run_button.setEnabled(False)
        self.ctx.jobs.start(job)

    def _job_finished(self, ok: bool, message: str) -> None:
        if not ok:
            for entry in self.inputs:
                if entry.status == "実行中":
                    entry.status = "中断"
        self._refresh_table()
        self.ctx.status.show_message(message)

    def _selection_changed(self) -> None:
        self.remove_button.setEnabled(bool(self.table.selectionModel().selectedRows()))
        self._render_selected()

    def _render_selected(self, *_args) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.image_view.set_image(None)
            self.detection_count.setText("検出数: —")
            self.image_view.set_overlay_labels("", "")
            return
        entry = self.inputs[rows[0].row()]
        if entry.status != "完了" or entry.image is None or entry.labels is None:
            self.image_view.set_image(None)
            self.detection_count.setText("検出数: —")
            self.image_view.set_overlay_labels(entry.filename, entry.model_id or "未割り当て")
            return
        selected_mode = next(
            label for label, button in self.display_buttons.items() if button.isChecked()
        )
        mode = self.modes[selected_mode]
        self.image_view.set_image(array_to_pixmap(render(entry.image, entry.labels, mode)))
        count = len(set(entry.labels.ravel()) - {0})
        self.detection_count.setText(f"検出数: {count}")
        self.image_view.set_overlay_labels(
            f"{entry.filename}　{entry.model_id}", f"検出 {count} 個"
        )

    def open_routing(self) -> None:
        """リリース済みモデル・振り分け画面を開く。"""
        self.ctx.navigator.navigate(PageId.RELEASED_MODELS)
