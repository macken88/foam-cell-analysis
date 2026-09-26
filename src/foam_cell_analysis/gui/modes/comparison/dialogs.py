"""候補操作用ダイアログ。"""

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ....services.models import Candidate, ExternalResult
from ...context import AppContext
from ...labels import config_key_label, format_score, model_type_label
from ...theme import numeric_font
from ...widgets.table import mark_primary, setup_table


class CandidateDialog(QDialog):
    """実験チェックポイントと推論設定を候補に登録する。"""

    def __init__(self, ctx: AppContext, parent=None, preset: dict | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("リリース候補を追加")
        self.setMinimumSize(560, 620)
        self.experiment = QComboBox()
        self.checkpoint = QComboBox()
        self.model_type = QLineEdit()
        self.preprocessing = QPlainTextEdit()
        self.preprocessing.setReadOnly(True)
        self.preprocessing.setMaximumHeight(120)
        self.model_type.setReadOnly(True)
        self.config = QComboBox()
        self.use_existing = QRadioButton("既存の推論設定を使う")
        self.use_new = QRadioButton("新しい推論設定を作る")
        self.use_existing.setChecked(True)
        self.config_values = QPlainTextEdit()
        self.config_values.setReadOnly(True)
        self.config_values.setMaximumHeight(80)
        self.comment = QLineEdit()
        form = QFormLayout()
        for label, widget in (
            ("完了した実験", self.experiment),
            ("途中保存モデル", self.checkpoint),
            ("モデル種類", self.model_type),
            ("前処理設定", self.preprocessing),
            ("コメント", self.comment),
        ):
            form.addRow(label, widget)
        self.params = QWidget()
        self.params_form = QFormLayout(self.params)
        self.fields: dict[str, QWidget] = {}
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        inference_form = QFormLayout()
        inference_form.addRow(self.use_existing, self.config)
        inference_form.addRow("既存設定の値", self.config_values)
        inference_form.addRow(self.use_new)
        layout.addLayout(inference_form)
        layout.addWidget(self.params)
        layout.addWidget(self.buttons)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("候補を追加")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        mark_primary(self.buttons.button(QDialogButtonBox.StandardButton.Ok))
        self.experiments = {
            e.experiment_id: e for e in ctx.backend.list_experiments() if e.status == "completed"
        }
        for experiment_id in self.experiments:
            self.experiment.addItem(experiment_id)
        self.experiment.currentTextChanged.connect(self._experiment_changed)
        self.checkpoint.currentTextChanged.connect(self._refresh_configs)
        self.use_existing.toggled.connect(self._toggle_config_mode)
        self.config.currentTextChanged.connect(self._show_config_values)
        self._experiment_changed(self.experiment.currentText())
        if preset:
            index = self.experiment.findText(preset.get("experiment_id", ""))
            if index >= 0:
                self.experiment.setCurrentIndex(index)
                self.checkpoint.setCurrentText(preset.get("checkpoint", ""))
        self._toggle_config_mode()

    def _experiment_changed(self, experiment_id: str) -> None:
        experiment = self.experiments.get(experiment_id)
        self.checkpoint.clear()
        if not experiment:
            return
        self.checkpoint.addItems([c.name for c in experiment.checkpoints])
        self.model_type.setText(model_type_label(experiment.model_type))
        self.model_type_value = experiment.model_type
        self.preprocessing.setPlainText(
            self._format_mapping(experiment.config.values.get("model", {}))
        )
        self._build_params(experiment.model_type)
        self._refresh_configs()

    def _build_params(self, model_type: str) -> None:
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self.fields.clear()
        specs = (
            [
                ("box_score_thresh", "検出スコア閾値", 0.5),
                ("box_nms_thresh", "Box NMS閾値", 0.5),
                ("box_detections_per_img", "最大検出数", 100),
            ]
            if model_type == "mask_rcnn"
            else [
                ("cellprob_threshold", "セル確率閾値", 0.0),
                ("flow_threshold", "フロー閾値", 0.4),
            ]
        )
        for key, label, default in specs:
            if key == "box_detections_per_img":
                widget = QSpinBox()
                widget.setRange(1, 10000)
                widget.setValue(int(default))
            else:
                widget = QDoubleSpinBox()
                widget.setRange(-10.0, 10.0)
                widget.setSingleStep(0.05)
                widget.setValue(default)
            self.fields[key] = widget
            self.params_form.addRow(label, widget)

    @staticmethod
    def _format_mapping(values: dict[str, Any], prefix: str = "") -> str:
        lines = []
        for key, value in values.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                lines.extend(CandidateDialog._format_mapping(value, name).splitlines())
            else:
                display_key = config_key_label(name)
                if display_key == name:
                    display_key = config_key_label(f"model.{name}")
                lines.append(f"{display_key}: {CandidateDialog._format_value(value)}")
        return "\n".join(lines)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, str) and value in {"mask_rcnn", "cellpose"}:
            return model_type_label(value)
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        if isinstance(value, bool):
            return "有効" if value else "無効"
        return str(value)

    def _refresh_configs(self, _value: str = "") -> None:
        model_type = getattr(self, "model_type_value", "")
        current = self.config.currentText()
        self.config.clear()
        self.config.addItems(
            [item.config_id for item in self.ctx.backend.list_inference_configs(model_type)]
        )
        if self.config.findText(current) >= 0:
            self.config.setCurrentText(current)
        self._show_config_values(self.config.currentText())

    def _show_config_values(self, config_id: str) -> None:
        config = next(
            (c for c in self.ctx.backend.list_inference_configs() if c.config_id == config_id), None
        )
        self.config_values.setPlainText(
            self._format_mapping(config.params) if config else "設定なし"
        )

    def _toggle_config_mode(self, _checked: bool = False) -> None:
        existing = self.use_existing.isChecked()
        self.config.setEnabled(existing)
        self.config_values.setEnabled(existing)
        for widget in self.fields.values():
            widget.setEnabled(not existing)

    def apply(self) -> Candidate:
        """入力値から候補を登録し、重複時は例外を伝える。"""
        config_id = self.config.currentText()
        if self.use_new.isChecked():
            values = {key: widget.value() for key, widget in self.fields.items()}
            config = self.ctx.backend.create_inference_config(self.model_type_value, values)
            config_id = config.config_id
        if not config_id:
            raise ValueError("推論設定を選択してください")
        return self.ctx.backend.add_candidate(
            self.experiment.currentText(),
            self.checkpoint.currentText(),
            config_id,
            self.comment.text(),
        )


class EvaluationDialog(QDialog):
    """評価値と外部解析結果を表示・保存する。"""

    def __init__(self, ctx: AppContext, candidate: Candidate, validation: str, parent=None) -> None:
        super().__init__(parent)
        self.ctx, self.candidate, self.validation = ctx, candidate, validation
        self.read_only = candidate.status == "released"
        self.setWindowTitle(f"詳細評価 - {candidate.candidate_id}")
        self.setMinimumSize(800, 640)
        self.resize(800, 640)
        self.metrics = QTableWidget(2, 5)
        self.metrics.setHorizontalHeaderLabels(["評価項目", "全体", "分類A", "分類B", "分類C"])
        setup_table(self.metrics, stretch_column=4)
        evaluation = candidate.evaluations[validation]
        classes = ["分類A", "分類B", "分類C"]
        all_count = sum(value[1] for value in evaluation.per_class.values())
        metric_values = ["平均適合率 (mAP)", format_score(evaluation.overall_map)]
        count_values = ["対象件数", str(all_count)]
        for classification in classes:
            score, count = evaluation.per_class.get(classification, (None, 0))
            metric_values.append(format_score(score))
            count_values.append(str(count))
        for column, value in enumerate(metric_values):
            item = QTableWidgetItem(value)
            if column:
                item.setFont(numeric_font())
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.metrics.setItem(0, column, item)
        for column, value in enumerate(count_values):
            item = QTableWidgetItem(value)
            if column:
                item.setFont(numeric_font())
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.metrics.setItem(1, column, item)
        self.metrics.setFixedHeight(
            self.metrics.horizontalHeader().height()
            + sum(self.metrics.rowHeight(row) for row in range(self.metrics.rowCount()))
            + self.metrics.frameWidth() * 2
            + 2
        )
        self.results = QTableWidget(0, 3)
        self.results.setHorizontalHeaderLabels(["項目名", "値", "単位"])
        setup_table(self.results, stretch_column=0)
        if self.read_only:
            self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for item in candidate.external_results:
            self._append_result(item.name, item.value, item.unit)
        self.add_row = QPushButton("行を追加")
        self.remove_row = QPushButton("選択行を削除")
        self.software, self.date, self.comment = (
            QLineEdit(candidate.external_software),
            QLineEdit(candidate.external_date),
            QLineEdit(candidate.comment),
        )
        form = QFormLayout()
        form.addRow("解析ソフト名", self.software)
        form.addRow("解析日", self.date)
        form.addRow("コメント", self.comment)
        self.add_row.clicked.connect(lambda: self._append_result("", "", ""))
        self.remove_row.clicked.connect(
            lambda: (
                self.results.removeRow(self.results.currentRow())
                if self.results.currentRow() >= 0
                else None
            )
        )
        controls = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        controls.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        controls.button(QDialogButtonBox.StandardButton.Close).setText("閉じる")
        mark_primary(controls.button(QDialogButtonBox.StandardButton.Save))
        controls.button(QDialogButtonBox.StandardButton.Save).clicked.connect(self._save)
        controls.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("アプリ内では自動算出しない補助評価です。"))
        layout.addWidget(self.metrics)
        layout.addWidget(QLabel("外部粒子解析結果"))
        result_actions = QHBoxLayout()
        result_actions.addStretch(1)
        result_actions.addWidget(self.add_row)
        result_actions.addWidget(self.remove_row)
        layout.addLayout(result_actions)
        layout.addWidget(self.results)
        layout.addLayout(form)
        layout.addWidget(controls)
        for widget in (self.add_row, self.remove_row, self.software, self.date, self.comment):
            widget.setEnabled(not self.read_only)
        if self.read_only:
            controls.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)

    def _append_result(self, name: str, value: str | float, unit: str) -> None:
        row = self.results.rowCount()
        self.results.insertRow(row)
        for col, cell_value in enumerate((name, value, unit)):
            item = QTableWidgetItem(str(cell_value))
            if col == 1:
                item.setFont(numeric_font())
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results.setItem(row, col, item)

    def apply(self) -> Candidate:
        """表の外部評価を Backend に保存する。"""
        results = [
            ExternalResult(
                self.results.item(row, 0).text(),
                self.results.item(row, 1).text(),
                self.results.item(row, 2).text(),
            )
            for row in range(self.results.rowCount())
        ]
        return self.ctx.backend.save_external_results(
            self.candidate.candidate_id,
            results,
            self.software.text(),
            self.date.text(),
            self.comment.text(),
        )

    def _save(self) -> None:
        self.apply()
        self.accept()


class ReleaseDialog(QDialog):
    """リリース登録コメントを受け取る。"""

    def __init__(self, candidate: Candidate, parent=None) -> None:
        super().__init__(parent)
        self.candidate = candidate
        self.setWindowTitle("リリース済みモデル登録")
        self.setMinimumSize(560, 360)
        self.comment = QLineEdit(candidate.comment)
        layout = QFormLayout(self)
        layout.addRow("候補", QLabel(candidate.candidate_id))
        layout.addRow("実験", QLabel(candidate.experiment_id))
        layout.addRow("途中保存モデル", QLabel(candidate.checkpoint))
        layout.addRow("コメント", self.comment)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "リリース済みモデルとして登録"
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        mark_primary(self.buttons.button(QDialogButtonBox.StandardButton.Ok))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow(self.buttons)


class MaskExportDialog(QDialog):
    """粒子解析向けマスク出力条件を入力する。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("粒子解析用マスク出力")
        self.setMinimumSize(560, 360)
        self.output_dir = QLineEdit()
        self.browse = QPushButton("参照…")
        output_row = QWidget()
        output_layout = QGridLayout(output_row)
        output_layout.addWidget(self.output_dir, 0, 0)
        output_layout.addWidget(self.browse, 0, 1)
        self.instance_mask = QCheckBox("インスタンスラベルマスク")
        self.instance_mask.setChecked(True)
        self.binary_mask = QCheckBox("粒子解析用二値マスク")
        self.file_format = QComboBox()
        self.file_format.addItems(["PNG", "TIFF"])
        self.scope = QComboBox()
        self.scope.addItems(["検証用データセット全体", "画像分類で絞り込み"])
        self.classification = QComboBox()
        self.classification.addItems(["分類A", "分類B", "分類C"])
        self.classification.setEnabled(False)
        self.scope.currentIndexChanged.connect(
            lambda index: self.classification.setEnabled(index == 1)
        )
        self.browse.clicked.connect(self._browse)
        form = QFormLayout(self)
        form.addRow("出力先フォルダ", output_row)
        form.addRow("出力内容", self.instance_mask)
        form.addRow("", self.binary_mask)
        form.addRow("ファイル形式", self.file_format)
        form.addRow("対象", self.scope)
        form.addRow("画像分類", self.classification)
        form.addRow("", QLabel("後処理は隣接気泡の分離のみ（8近傍接触の境界を背景化）"))
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("出力を開始")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        mark_primary(self.buttons.button(QDialogButtonBox.StandardButton.Ok))
        form.addRow(self.buttons)

    def values(self) -> dict[str, str | bool]:
        """選択された出力条件を返す。"""
        return {
            "output_dir": self.output_dir.text(),
            "instance_mask": self.instance_mask.isChecked(),
            "binary_mask": self.binary_mask.isChecked(),
            "file_format": self.file_format.currentText(),
            "scope": self.scope.currentText(),
            "classification": self.classification.currentText()
            if self.scope.currentIndex() == 1
            else "",
        }

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "出力先フォルダを選択")
        if directory:
            self.output_dir.setText(directory)
