"""候補操作用ダイアログ（比較・推論設計 16.2）。"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ....services.backend import MaskExportParams
from ....services.comparison_service import (
    INFERENCE_PARAM_SPECS,
    DuplicateCandidateError,
    is_recovered_release,
)
from ....services.models import Candidate, EvaluationRecord, Experiment
from ...context import AppContext
from ...error_messages import value_error_message
from ...labels import (
    OOF_NOTE,
    config_key_label,
    contamination_label,
    format_score,
    model_type_label,
)
from ...theme import numeric_font, set_style
from ...widgets.table import fit_table_columns, mark_primary, setup_table

logger = logging.getLogger(__name__)

# 完了した試行の状態（TrainingService は completed、モックの初期データは 完走）
_COMPLETED_RESULTS = {"completed", "完走"}
AP_NOTE = "AP は Cellpose 方式（TP / (TP + FP + FN)、IoU 0.50–0.95 の平均、画像平均）"
BASE_MISMATCH_NOTE = "⚠ 基準とは別の検証用データセットで評価しています"


def format_evaluated_at(value: str | None) -> str:
    """評価の完了時刻（ISO 形式）を「年-月-日 時:分」で返す。読めなければダッシュ。"""
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return "—"
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.strftime("%Y-%m-%d %H:%M")


def user_message(error: BaseException, fallback: str) -> str:
    """Backend の ValueError から、画面に出してよい部分（最初の「: 」より前）を返す。

    パスや内部の詳細は「: 」の後ろに入るので出さない。ValueError 以外は fallback。
    """
    return value_error_message(error, fallback)


def completed_attempts(experiment: Experiment) -> list[int]:
    """実験の完了した試行番号を返す。"""
    return [run.attempt for run in experiment.runs if run.result in _COMPLETED_RESULTS]


FINAL_PRUNED_PATH = "checkpoints/final.pt"
FINAL_PRUNED_REASON = "最終学習モデルは成果物の整理で削除されています"


def usable_attempts(backend, experiment: Experiment) -> list[int]:
    """候補にできる試行番号を返す。final.pt が成果物の整理で削除された試行は除く。"""
    result = []
    for attempt in completed_attempts(experiment):
        try:
            pruned = backend.pruned_paths(experiment.experiment_id, attempt)
        except (KeyError, ValueError, OSError):
            pruned = set()
        if FINAL_PRUNED_PATH not in pruned:
            result.append(attempt)
    return result


def oof_note(candidate: Candidate) -> str | None:
    """候補一覧の OOF AP が「対象外」になる理由の注記。対象外でなければ None。"""
    if candidate.oof_applicability == "matching":
        return None
    if candidate.oof_applicability == "different":
        return "推論設定が学習時と異なるため、候補一覧の OOF AP は「対象外」です。"
    return "学習時の評価条件を確認できないため、候補一覧の OOF AP は「対象外」です。"


def base_version_text(version: str, base: str | None) -> str:
    """「検証用データセット: val_v001 （学習用データセットの基準: val_v000）」の文。"""
    return f"検証用データセット: {version} （学習用データセットの基準: {base or '記録なし'}）"


class _WorkerThread(QThread):
    """work(progress, is_cancelled) を別スレッドで実行する。"""

    step = Signal(int, int)

    def __init__(self, work: Callable[..., Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.work = work
        self.cancel_requested = False
        self.result: Any = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = self.work(
                lambda done, total: self.step.emit(int(done), int(total)),
                lambda: self.cancel_requested,
            )
        except Exception as error:  # 画面側で利用者向けの文に置き換える
            self.error = error


class WorkerJob(QObject):
    """時間のかかる処理（抽出結果出力・リリース登録）をワーカースレッドで動かすジョブ。

    JobManager に登録できる形（name・key・progress・finished・start・cancel）にしてある。
    結果は finished の後に result、例外は error に入る。
    """

    progress = Signal(int, int, str)
    finished = Signal(bool, str)

    def __init__(
        self,
        name: str,
        work: Callable[..., Any],
        key: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.name = name
        self.key = key
        self.result: Any = None
        self.error: BaseException | None = None
        self._thread = _WorkerThread(work, self)
        self._thread.step.connect(self._step)
        self._thread.finished.connect(self._done)

    def start(self) -> WorkerJob:
        self._thread.start()
        return self

    def cancel(self) -> None:
        """中止を求める。処理は次の確認点で止まる。"""
        self._thread.cancel_requested = True

    @property
    def cancel_requested(self) -> bool:
        return self._thread.cancel_requested

    def is_running(self) -> bool:
        return self._thread.isRunning()

    def _step(self, done: int, total: int) -> None:
        self.progress.emit(done, total, f"{self.name}: {done}/{total}")

    def _done(self) -> None:
        self._thread.wait()
        self.result, self.error = self._thread.result, self._thread.error
        if self.error is not None:
            logger.error("%sに失敗しました", self.name, exc_info=self.error)
        ok = self.error is None
        self.finished.emit(ok, f"{self.name}が完了しました" if ok else f"{self.name}に失敗しました")


def _note(text: str = "", state: str | None = None) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    if state:
        set_style(label, state=state)
    return label


class CandidateDialog(QDialog):
    """完了した試行の final.pt と推論設定を候補に登録する（5.3）。"""

    def __init__(self, ctx: AppContext, parent=None, preset: dict | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.created: Candidate | None = None
        self.duplicate_candidate_id: str | None = None
        self.setWindowTitle("リリース候補を追加")
        self.setMinimumSize(560, 620)
        self.experiment = QComboBox()
        self.attempt = QComboBox()
        self.attempt.setPlaceholderText("試行を選んでください")
        self.model_type = QLineEdit()
        self.model_type.setReadOnly(True)
        self.preprocessing = QPlainTextEdit()
        self.preprocessing.setReadOnly(True)
        self.preprocessing.setMaximumHeight(120)
        self.config = QComboBox()
        self.use_training = QRadioButton("学習時と同じ推論設定")
        self.use_existing = QRadioButton("登録済みの推論設定を使う")
        self.use_new = QRadioButton("推論設定を変更する")
        self.use_training.setChecked(True)
        self.training_params: dict[str, Any] | None = None
        self.config_values = QPlainTextEdit()
        self.config_values.setReadOnly(True)
        self.config_values.setMaximumHeight(80)
        self.comment = QLineEdit()
        form = QFormLayout()
        for label, widget in (
            ("完了した実験", self.experiment),
            ("学習モデル", self.attempt),
            ("モデル種類", self.model_type),
            ("前処理設定", self.preprocessing),
            ("コメント", self.comment),
        ):
            form.addRow(label, widget)
        self.params = QWidget()
        self.params_form = QFormLayout(self.params)
        self.fields: dict[str, QSpinBox | QDoubleSpinBox] = {}
        self.error = _note(state="error")
        self.error.hide()
        self.info = _note()
        self.info.hide()
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("候補を追加")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        mark_primary(self.ok_button)
        self.buttons.accepted.connect(self._submit)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        inference_form = QFormLayout()
        inference_form.addRow(self.use_training)
        inference_form.addRow(self.use_existing, self.config)
        inference_form.addRow("既存設定の値", self.config_values)
        inference_form.addRow(self.use_new)
        layout.addLayout(inference_form)
        layout.addWidget(self.params)
        layout.addStretch(1)
        layout.addWidget(self.info)
        layout.addWidget(self.error)
        layout.addWidget(self.buttons)
        self.experiments = {
            experiment.experiment_id: experiment
            for experiment in ctx.backend.list_experiments()
            if usable_attempts(ctx.backend, experiment)
        }
        self.experiment.addItems(list(self.experiments))
        self.experiment.currentTextChanged.connect(self._experiment_changed)
        self.attempt.currentIndexChanged.connect(self._attempt_changed)
        self.use_training.toggled.connect(self._toggle_config_mode)
        self.use_existing.toggled.connect(self._toggle_config_mode)
        self.config.currentTextChanged.connect(self._show_config_values)
        self.config.currentTextChanged.connect(self._update_ok)
        preset = preset or {}
        index = self.experiment.findText(str(preset.get("experiment_id", "")))
        if index >= 0:
            self.experiment.setCurrentIndex(index)
        self._experiment_changed(self.experiment.currentText())
        attempt = preset.get("attempt")
        if isinstance(attempt, int):
            found = self.attempt.findData(attempt)
            if found >= 0:
                self.attempt.setCurrentIndex(found)
        self.comment.setText(str(preset.get("comment", "")))
        params = preset.get("inference_params")
        if isinstance(params, dict):
            for key, value in params.items():
                widget = self.fields.get(key)
                if widget is not None:
                    widget.setValue(value)
        if preset.get("prefer_new_config"):
            self.use_new.setChecked(True)
            self.info.setText("元候補と学習モデルは保持し、新しい候補を未評価で追加します。")
            self.info.show()
        self._toggle_config_mode()

    def _experiment_changed(self, experiment_id: str) -> None:
        experiment = self.experiments.get(experiment_id)
        self.attempt.clear()
        if not experiment:
            self._update_ok()
            return
        attempts = usable_attempts(self.ctx.backend, experiment)
        for number in attempts:
            self.attempt.addItem(f"試行 {number} / final.pt", number)
        # 試行が 1 つだけなら迷わないので選んでおく。複数なら利用者が選ぶ（5.3）
        self.attempt.setCurrentIndex(0 if len(attempts) == 1 else -1)
        self.model_type.setText(model_type_label(experiment.model_type))
        self.model_type_value = experiment.model_type
        self.preprocessing.setPlainText(
            self._format_mapping(experiment.config.values.get("model", {}))
        )
        self._build_params(experiment.model_type)
        self._refresh_configs()
        self._attempt_changed()
        self._update_ok()

    def _build_params(self, model_type: str) -> None:
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self.fields.clear()
        specs = INFERENCE_PARAM_SPECS.get(model_type, {})
        try:
            defaults = self.ctx.backend.default_inference_params(model_type)
        except ValueError:
            defaults = {}
        for key, (minimum, maximum, integer, label) in specs.items():
            if integer:
                widget = QSpinBox()
                widget.setRange(int(minimum), int(maximum))
                widget.setValue(int(defaults.get(key, minimum)))
            else:
                widget = QDoubleSpinBox()
                widget.setDecimals(2)
                widget.setRange(float(minimum), float(maximum))
                widget.setSingleStep(0.05)
                widget.setValue(float(defaults.get(key, minimum)))
            widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            widget.setToolTip(f"{minimum}〜{maximum}")
            self.fields[key] = widget
            self.params_form.addRow(label, widget)
        self._toggle_config_mode()

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
        if self.config.count() == 0:
            # 既存の設定がないときは、新しく作る以外に選べない
            self.use_existing.setEnabled(False)
            self.use_existing.setToolTip("このモデル種類の推論設定はまだありません")
            self.use_new.setChecked(True)
        elif not self.use_existing.isEnabled():
            self.use_existing.setEnabled(True)
            self.use_existing.setToolTip("")
            self.use_existing.setChecked(True)

    def _show_config_values(self, config_id: str) -> None:
        config = next(
            (c for c in self.ctx.backend.list_inference_configs() if c.config_id == config_id), None
        )
        if config is None:
            self.config_values.setPlainText("設定なし")
            return
        specs = INFERENCE_PARAM_SPECS.get(config.model_type, {})
        self.config_values.setPlainText(
            "\n".join(
                f"{specs[key][3] if key in specs else key}: {value}"
                for key, value in config.params.items()
            )
        )

    def _toggle_config_mode(self, _checked: bool = False) -> None:
        existing = self.use_existing.isChecked()
        training = self.use_training.isChecked()
        self.config.setEnabled(existing)
        self.config_values.setEnabled(existing)
        for widget in self.fields.values():
            widget.setEnabled(not existing and not training)
        self._update_ok()

    def _update_ok(self, *_args) -> None:
        reason = ""
        if not self.experiments:
            reason = "完了した実験がありません"
        elif self.attempt.currentIndex() < 0:
            reason = "試行を選んでください"
        elif self.use_existing.isChecked() and not self.config.currentText():
            reason = "推論設定を選んでください"
        elif self.use_training.isChecked() and self.training_params is None:
            reason = "学習時の推論設定を確認できないため、この選択肢は使えません"
        self.ok_button.setEnabled(not reason)
        self.ok_button.setToolTip(reason)

    def apply(self) -> Candidate:
        """入力値から候補を登録する。重複などは ValueError。"""
        attempt = self.attempt.currentData()
        if not isinstance(attempt, int):
            raise ValueError("試行を選んでください")
        config_id = self.config.currentText()
        experiment = self.experiments.get(self.experiment.currentText())
        # 新しい推論設定を作る前に、この試行が使えるか確かめる（使えないのに設定だけ残さない）
        if experiment is None or attempt not in usable_attempts(self.ctx.backend, experiment):
            raise ValueError(FINAL_PRUNED_REASON)
        if self.use_training.isChecked():
            matching = next(
                (
                    item
                    for item in self.ctx.backend.list_inference_configs(self.model_type_value)
                    if item.params == self.training_params
                ),
                None,
            )
            config_id = matching.config_id if matching is not None else ""
            if not config_id:
                config_id = self.ctx.backend.create_inference_config(
                    self.model_type_value, self.training_params
                ).config_id
        elif self.use_new.isChecked():
            values = {key: widget.value() for key, widget in self.fields.items()}
            config = self.ctx.backend.create_inference_config(self.model_type_value, values)
            config_id = config.config_id
        if not config_id:
            raise ValueError("推論設定を選択してください")
        self.created = self.ctx.backend.add_candidate(
            self.experiment.currentText(), attempt, config_id, self.comment.text()
        )
        return self.created

    def _attempt_changed(self, *_args) -> None:
        """選択試行の保存済み評価条件を既定値へ反映する。"""
        attempt = self.attempt.currentData()
        experiment = self.experiments.get(self.experiment.currentText())
        self.training_params = None
        if isinstance(attempt, int) and experiment is not None:
            try:
                snapshot = self.ctx.backend.create_candidate_snapshot(
                    experiment.experiment_id, attempt=attempt
                )
            except (KeyError, ValueError, OSError):
                snapshot = None
            if snapshot is not None and snapshot.training_eval_params:
                self.training_params = dict(snapshot.training_eval_params)
                for key, value in self.training_params.items():
                    widget = self.fields.get(key)
                    if widget is not None:
                        widget.setValue(value)
        self.use_training.setEnabled(self.training_params is not None)
        self.use_training.setToolTip(
            "" if self.training_params is not None else "学習時の評価条件を確認できません"
        )
        if self.training_params is None and self.use_training.isChecked():
            self.use_new.setChecked(True)
        self._toggle_config_mode()

    def _submit(self) -> None:
        try:
            self.apply()
        except DuplicateCandidateError as error:
            self.duplicate_candidate_id = error.candidate_id
            self.accept()
            return
        except (ValueError, KeyError) as error:
            logger.warning("候補を追加できません: %s", error)
            self.error.setText(user_message(error, "候補を追加できませんでした"))
            self.error.show()
            return
        self.accept()


class EvaluationDialog(QDialog):
    """採用した評価 1 件の詳細と、その評価に対する外部解析結果（9.4）。

    評価 ID はダイアログを開いた時点で固定する（7.7）。
    """

    def __init__(
        self,
        ctx: AppContext,
        candidate: Candidate,
        record: EvaluationRecord,
        parent=None,
        *,
        read_only: bool = False,
    ) -> None:
        super().__init__(parent)
        self.ctx, self.candidate, self.record = ctx, candidate, record
        self.evaluation_id = record.evaluation_id
        editable = getattr(ctx.backend, "external_analysis_editable", None)
        self.read_only = read_only or not bool(
            editable and editable(candidate.candidate_id, record.evaluation_id)
        )
        self.setWindowTitle(f"評価詳細 - {candidate.candidate_id}")
        self.setMinimumSize(800, 640)
        self.resize(820, 680)
        evaluation = record.evaluation
        oof = candidate.oof_evaluation
        classes = list(evaluation.per_class) if evaluation else []
        classes += [name for name in (oof.per_class if oof else {}) if name not in classes]
        self.metrics = QTableWidget(4, 2 + len(classes))
        self.metrics.setHorizontalHeaderLabels(["評価項目", "全体", *classes])
        setup_table(self.metrics, stretch_column=self.metrics.columnCount() - 1)
        self.metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        def total(value) -> int:
            if value is None:
                return 0
            return value.n_images or sum(count for _ap, count in value.per_class.values())

        rows = [
            ("検証 AP", evaluation, False),
            ("  対象件数", evaluation, True),
            ("学習時 OOF AP（参考）", oof, False),
            ("  対象件数", oof, True),
        ]
        for row, (label, value, is_count) in enumerate(rows):
            per_class = value.per_class if value else {}
            cells = [label, str(total(value)) if is_count else format_score(_ap(value))]
            cells.extend(
                str(per_class.get(name, (None, 0))[1])
                if is_count
                else format_score(per_class.get(name, (None, 0))[0])
                for name in classes
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column:
                    item.setFont(numeric_font())
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.metrics.setItem(row, column, item)
        fit_table_columns(self.metrics)
        self.metrics.setFixedHeight(
            self.metrics.horizontalHeader().height()
            + sum(self.metrics.rowHeight(row) for row in range(self.metrics.rowCount()))
            + self.metrics.frameWidth() * 2
            + 2
        )

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"検証用データセット: {record.validation_version}"))
        if read_only and record.validation_version != candidate.validation_version:
            history_note = _note("候補に固定した検証版と異なる過去評価のため、閲覧のみです。")
            set_style(history_note, role="note")
            layout.addWidget(history_note)
        layout.addWidget(QLabel(f"評価日時: {format_evaluated_at(record.completed_at)}"))
        if record.schema == 1:
            layout.addWidget(
                QLabel(
                    "旧評価の同一画像混入記録: "
                    f"{contamination_label(record.contamination.get('status'))}"
                )
            )
        epoch = candidate.oof_epoch
        layout.addWidget(
            QLabel(
                f"学習: {candidate.experiment_id} ・ 試行 {candidate.source_attempt_number} ・ "
                f"選択エポック {epoch if epoch is not None else '—'} ・ final.pt"
            )
        )
        layout.addWidget(self.metrics)
        for text in (AP_NOTE, OOF_NOTE):
            label = _note(text)
            set_style(label, role="note")
            layout.addWidget(label)
        self.oof_note = _note(oof_note(candidate) or "")
        set_style(self.oof_note, role="note")
        self.oof_note.setVisible(bool(oof_note(candidate)))
        layout.addWidget(self.oof_note)

        # ---- 画像別の外部解析値 ----
        saved = self._saved_record()
        layout.addSpacing(8)
        layout.addWidget(
            QLabel("画像ごとの円相当径中央値を入力します。集計は入力済み画像の平均です。")
        )
        self.unit = QComboBox()
        self.unit.addItems(["µm", "px"])
        if saved and saved.get("unit") in {"µm", "px"}:
            self.unit.setCurrentText(saved["unit"])
        self.software = QLineEdit(str(saved.get("software", "")) if saved else "")
        self.software_version = QLineEdit(str(saved.get("software_version", "")) if saved else "")
        self.software_version.setMaximumWidth(90)
        self.date = QLineEdit(str(saved.get("analyzed_on", "")) if saved else "")
        self.date.setPlaceholderText("2026-09-29")
        self.date.setMaximumWidth(110)
        meta = QHBoxLayout()
        for label, widget in (
            ("単位", self.unit),
            ("解析ソフト名", self.software),
            ("版", self.software_version),
            ("解析日", self.date),
        ):
            meta.addWidget(QLabel(label))
            meta.addWidget(widget)
        layout.addLayout(meta)
        self.results = QTableWidget(0, 3)
        self.results.setHorizontalHeaderLabels(["画像 ID", "元ファイル名", "円相当径中央値"])
        setup_table(self.results, stretch_column=1)
        values = (saved or {}).get("values", {})
        self.external_items = ctx.backend.list_validation_items(record.validation_version, None)
        for data_item in self.external_items:
            row_index = self.results.rowCount()
            self.results.insertRow(row_index)
            for column, text in enumerate(
                (data_item.item_id, data_item.source_filename, values.get(data_item.item_id, ""))
            ):
                cell = QTableWidgetItem("" if text is None else str(text))
                if column < 2:
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.results.setItem(row_index, column, cell)
        self.paste = QPlainTextEdit()
        self.paste.setPlaceholderText("画像 ID または元ファイル名、タブ、数値の順で貼り付け")
        self.paste.setMaximumHeight(70)
        self.apply_paste = QPushButton("貼り付け値を反映")
        result_actions = QHBoxLayout()
        result_actions.addStretch(1)
        result_actions.addWidget(self.apply_paste)
        layout.addWidget(self.paste)
        layout.addLayout(result_actions)
        layout.addWidget(self.results, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        legacy_records = self._legacy_external_records()
        if legacy_records:
            legacy_lines = []
            for legacy in legacy_records:
                rows = legacy.get("results") or []
                legacy_lines.extend(
                    f"{row.get('name', '項目')}: {row.get('value', '')} {row.get('unit', '')}"
                    for row in rows
                    if isinstance(row, dict)
                )
            legacy_label = QLabel(
                "旧形式の外部解析記録（読み取り専用）\n" + "\n".join(legacy_lines)
            )
            legacy_label.setWordWrap(True)
            layout.addWidget(legacy_label)
        fit_table_columns(self.results)
        self.comment = QLineEdit(candidate.comment)
        comment_row = QHBoxLayout()
        comment_row.addWidget(QLabel("コメント"))
        comment_row.addWidget(self.comment, 1)
        layout.addLayout(comment_row)
        self.error = _note(state="error")
        self.error.hide()
        layout.addWidget(self.error)
        self.controls = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        self.save_button = self.controls.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText("保存")
        self.controls.button(QDialogButtonBox.StandardButton.Close).setText("閉じる")
        mark_primary(self.save_button)
        self.save_button.clicked.connect(self._save)
        self.controls.rejected.connect(self.reject)
        layout.addWidget(self.controls)
        self.apply_paste.clicked.connect(self._apply_external_paste)
        self.results.itemChanged.connect(lambda _item: self._refresh_external_summary())
        self.unit.currentTextChanged.connect(self._refresh_external_summary)
        self._refresh_external_summary()
        if self.read_only:
            self.unit.setEnabled(False)
            self.software.setReadOnly(True)
            self.software_version.setReadOnly(True)
            self.date.setReadOnly(True)
            self.paste.setReadOnly(True)
            self.apply_paste.setEnabled(False)
            self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.comment.setReadOnly(True)
            self.save_button.setEnabled(False)
        if self.read_only:
            self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            for widget in (
                self.apply_paste,
                self.paste,
                self.unit,
                self.software,
                self.software_version,
                self.date,
                self.comment,
                self.save_button,
            ):
                widget.setEnabled(False)
            self.save_button.setToolTip("リリース済みの候補は変更できません")

    def _saved_record(self) -> dict[str, Any] | None:
        """この評価に対して最後に保存した外部解析結果。"""
        try:
            records = self.ctx.backend.list_external_results(self.candidate.candidate_id)
        except (KeyError, ValueError, OSError) as error:
            logger.warning("外部解析結果を読めません: %s", error)
            return None
        matching = [item for item in records if item.get("evaluation_id") == self.evaluation_id]
        return matching[-1] if matching else None

    def _legacy_external_records(self) -> list[dict[str, Any]]:
        try:
            records = self.ctx.backend.list_external_results(self.candidate.candidate_id)
        except (KeyError, ValueError, OSError):
            return []
        return [
            record
            for record in records
            if record.get("evaluation_id") == self.evaluation_id
            and "format" not in record
            and isinstance(record.get("results"), list)
        ]

    def _refresh_external_summary(self, *_args) -> None:
        """入力済み画像数と全体・分類別の平均を即時表示する。"""
        values: dict[str, float] = {}
        invalid = False
        for row, data_item in enumerate(self.external_items):
            raw = self.results.item(row, 2).text().strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                invalid = True
                continue
            if not math.isfinite(value) or value < 0:
                invalid = True
                continue
            values[data_item.item_id] = value
        if invalid:
            self.summary.setText("数値・0以上・有限の値を入力してください")
            return
        entered = len(values)
        mean = sum(values.values()) / entered if entered else None
        groups: dict[str, list[float]] = {
            item.classification or "分類なし": [] for item in self.external_items
        }
        for data_item in self.external_items:
            if data_item.item_id in values:
                groups.setdefault(data_item.classification or "分類なし", []).append(
                    values[data_item.item_id]
                )
        pieces = [
            f"全体 {mean:.4g} {self.unit.currentText()} ({entered}/{len(self.external_items)} 枚)"
            if mean is not None
            else f"全体 — ({entered}/{len(self.external_items)} 枚)"
        ]
        totals: dict[str, int] = {}
        for item in self.external_items:
            name = item.classification or "分類なし"
            totals[name] = totals.get(name, 0) + 1
        for name, group in sorted(groups.items()):
            average = f"{sum(group) / len(group):.4g}" if group else "—"
            pieces.append(f"{name} {average} ({len(group)}/{totals[name]} 枚)")
        self.summary.setText("画像ごとの円相当径中央値の平均: " + " / ".join(pieces))

    def _apply_external_paste(self) -> None:
        """ID または一意な元ファイル名で貼り付けた画像別値を反映する。"""
        by_id = {item.item_id: row for row, item in enumerate(self.external_items)}
        names: dict[str, list[int]] = {}
        for row, item in enumerate(self.external_items):
            names.setdefault(item.source_filename, []).append(row)
            names.setdefault(Path(item.source_filename).stem, []).append(row)
        seen: set[int] = set()
        updates = []
        for line_number, line in enumerate(self.paste.toPlainText().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split("\t")
            if line_number == 1 and parts == ["item_id", "median_diameter"]:
                continue
            if len(parts) != 2:
                self.error.setText(f"{line_number} 行目は画像名と数値をタブで区切ってください")
                self.error.show()
                return
            key, raw = (part.strip() for part in parts)
            try:
                parsed_value = float(raw)
            except ValueError:
                self.error.setText(f"{line_number} 行目の値が数値ではありません")
                self.error.show()
                return
            if not math.isfinite(parsed_value) or parsed_value < 0:
                self.error.setText(f"{line_number} 行目の値は有限の0以上にしてください")
                self.error.show()
                return
            row = by_id.get(key)
            if row is None:
                matches = set(names.get(key, []))
                if len(matches) != 1:
                    self.error.setText(
                        f"{line_number} 行目の画像名は見つからないか、一意に決まりません"
                    )
                    self.error.show()
                    return
                row = next(iter(matches))
            if row in seen:
                self.error.setText(f"{line_number} 行目で同じ画像が重複しています")
                self.error.show()
                return
            seen.add(row)
            updates.append((row, raw))
        for row, raw in updates:
            self.results.item(row, 2).setText(raw)
        self._refresh_external_summary()
        self.error.hide()

    def apply(self) -> Candidate:
        """表の外部解析結果を、開いた時点の評価 ID に結び付けて保存する。"""
        values = {}
        for row in range(self.results.rowCount()):
            item_id = self.results.item(row, 0).text()
            raw = self.results.item(row, 2).text().strip()
            if not raw:
                values[item_id] = None
                continue
            try:
                values[item_id] = float(raw)
            except ValueError as error:
                raise ValueError(f"数値を確認してください（{item_id}）") from error
        return self.ctx.backend.save_external_analysis(
            self.candidate.candidate_id,
            self.evaluation_id,
            values,
            unit=self.unit.currentText(),
            software=self.software.text().strip(),
            software_version=self.software_version.text().strip(),
            analyzed_on=self.date.text().strip(),
            comment=self.comment.text(),
        )

    def _save(self) -> None:
        try:
            self.apply()
        except (ValueError, KeyError, OSError) as error:
            logger.warning("外部解析結果を保存できません: %s", error)
            self.error.setText(user_message(error, "外部解析結果を保存できませんでした"))
            self.error.show()
            return
        self.accept()


def _ap(value) -> float | None:
    return value.overall_map if value is not None else None


class ReleaseDialog(QDialog):
    """開いた時点の評価でリリース登録する（13 章）。コピーはワーカースレッドで行う。"""

    def __init__(
        self,
        ctx: AppContext,
        candidate: Candidate,
        record: EvaluationRecord,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.ctx, self.candidate, self.record = ctx, candidate, record
        self.evaluation_id = record.evaluation_id
        self.model = None
        self.job: WorkerJob | None = None
        self.setWindowTitle("候補を採用")
        self.setMinimumSize(560, 360)
        legacy_evaluation = record.schema == 1
        contamination = str(record.contamination.get("status") or "")
        evaluation = record.evaluation
        count = (
            (evaluation.n_images or sum(count for _ap, count in evaluation.per_class.values()))
            if evaluation
            else 0
        )
        self.comment = QLineEdit(candidate.comment)
        form = QFormLayout()
        form.addRow("候補", QLabel(candidate.candidate_id))
        form.addRow(
            "学習",
            QLabel(
                f"{candidate.experiment_id} ・ 試行 {candidate.source_attempt_number} ・ final.pt"
            ),
        )
        form.addRow(
            "検証用データセット",
            QLabel(record.validation_version),
        )
        form.addRow(
            "評価",
            QLabel(
                f"評価日時 {format_evaluated_at(record.completed_at)} ・ "
                f"検証 AP {format_score(_ap(evaluation))}（対象 {count} 枚）"
            ),
        )
        self.contamination_note = _note(state="warning")
        if legacy_evaluation and contamination == "unknown":
            self.contamination_note.setText(
                "学習データに同じ画像が含まれていないかを確認できませんでした。"
                "このことはリリース記録に残ります。"
            )
        elif legacy_evaluation and contamination == "found":
            self.contamination_note.setText(
                "学習データと同じ画像が検証用データセットにあるため、採用できません。"
            )
            set_style(self.contamination_note, state="error")
        self.contamination_note.setVisible(bool(self.contamination_note.text()))
        form.addRow("", self.contamination_note)
        form.addRow("コメント", self.comment)
        self.progress_label = QLabel("モデルをコピーしています…（数十秒かかることがあります）")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress_label.hide()
        self.progress.hide()
        self.error = _note(state="error")
        self.error.hide()
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.cancel_button = self.buttons.button(QDialogButtonBox.StandardButton.Cancel)
        self.ok_button.setText("採用")
        self.cancel_button.setText("キャンセル")
        mark_primary(self.ok_button)
        self.buttons.accepted.connect(self._submit)
        self.buttons.rejected.connect(self.reject)
        if legacy_evaluation and contamination == "found":
            self.ok_button.setEnabled(False)
            self.ok_button.setToolTip(self.contamination_note.text())
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch(1)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.error)
        layout.addWidget(self.buttons)

    def _submit(self) -> None:
        if self.job is not None:
            return
        candidate_id, evaluation_id, comment = (
            self.candidate.candidate_id,
            self.evaluation_id,
            self.comment.text(),
        )
        backend = self.ctx.backend
        self.job = WorkerJob(
            "採用",
            lambda _progress, _cancelled: backend.release_candidate(
                candidate_id, evaluation_id, comment
            ),
            key="release",
        )
        self.job.finished.connect(self._finished)
        self.error.hide()
        self.progress_label.show()
        self.progress.show()
        self.ok_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.comment.setEnabled(False)
        self.ctx.jobs.start(self.job)

    def _finished(self, ok: bool, _message: str) -> None:
        job, self.job = self.job, None
        self.progress_label.hide()
        self.progress.hide()
        self.cancel_button.setEnabled(True)
        self.comment.setEnabled(True)
        if ok:
            self.model = job.result
            if is_recovered_release(self.model):
                QMessageBox.information(
                    self,
                    "採用",
                    f"前回の登録で {self.model.model_id} として公開済みでした。"
                    "候補の状態を修復しました。"
                    "コメントは前回の登録内容のままです。",
                )
            self.accept()
            return
        self.ok_button.setEnabled(True)
        self.error.setText(user_message(job.error, "候補を採用できませんでした"))
        self.error.show()

    def reject(self) -> None:
        # コピー中は閉じない（途中で閉じても処理は止まらないため）
        if self.job is None:
            super().reject()


class MaskExportDialog(QDialog):
    """評価済みの予測から粒子解析用抽出結果を出力する（12 章）。

    候補ごとの評価 ID と対象画像は、出力開始の時点で固定する。
    """

    def __init__(
        self,
        ctx: AppContext,
        selections: list[tuple[str, str]],
        validation_version: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.selections = list(selections)
        self.validation_version = validation_version
        self.job: WorkerJob | None = None
        self.summary: dict[str, Any] | None = None
        self.setWindowTitle("粒子解析用抽出結果出力")
        self.setMinimumSize(560, 400)
        self.items = ctx.backend.list_validation_items(validation_version)
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("出力先フォルダを選んでください")
        self.browse = QPushButton("参照")
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_dir, 1)
        output_layout.addWidget(self.browse)
        self.instance_mask = QCheckBox("気泡ごとのラベル画像")
        self.instance_mask.setChecked(True)
        self.binary_mask = QCheckBox("粒子解析用の白黒画像（隣接気泡を分離済み）")
        self.binary_mask.setChecked(True)
        self.file_format = QComboBox()
        self.file_format.addItems(["PNG", "TIFF"])
        self.scope = QComboBox()
        self.scope.addItems(["検証用データセット全体", "画像分類で絞り込み"])
        self.classification = QComboBox()
        self.classification.addItems(
            sorted({item.classification or "未分類" for item in self.items})
        )
        self.classification.setEnabled(False)
        self.scope.currentIndexChanged.connect(
            lambda index: self.classification.setEnabled(index == 1)
        )
        self.browse.clicked.connect(self._browse)
        ids = "、".join(candidate_id for candidate_id, _evaluation in self.selections)
        form = QFormLayout()
        form.addRow("対象候補", QLabel(f"{ids}（{validation_version} の評価結果）"))
        form.addRow("出力先フォルダ", output_row)
        form.addRow("出力内容", self.instance_mask)
        form.addRow("", self.binary_mask)
        form.addRow("ファイル形式", self.file_format)
        form.addRow("対象", self.scope)
        form.addRow("画像分類", self.classification)
        note = _note("後処理は隣接気泡の分離のみ（8近傍で接する境界を背景にします）")
        set_style(note, role="note")
        form.addRow("", note)
        self.progress_label = QLabel("出力しています…")
        self.progress = QProgressBar()
        self.progress_label.hide()
        self.progress.hide()
        self.error = _note(state="error")
        self.error.hide()
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.start_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.cancel_button = self.buttons.button(QDialogButtonBox.StandardButton.Cancel)
        self.start_button.setText("出力開始")
        self.cancel_button.setText("キャンセル")
        mark_primary(self.start_button)
        self.buttons.accepted.connect(self._start)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch(1)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.error)
        layout.addWidget(self.buttons)
        for signal in (
            self.output_dir.textChanged,
            self.instance_mask.toggled,
            self.binary_mask.toggled,
        ):
            signal.connect(self._update_start)
        self._update_start()

    def values(self) -> dict[str, Any]:
        """選択された出力条件を返す。"""
        return {
            "output_dir": self.output_dir.text().strip(),
            "instance_mask": self.instance_mask.isChecked(),
            "binary_mask": self.binary_mask.isChecked(),
            "file_format": self.file_format.currentText(),
            "scope": self.scope.currentText(),
            "classification": self.classification.currentText()
            if self.scope.currentIndex() == 1
            else "",
        }

    def _update_start(self, *_args) -> None:
        if self.job is not None:
            return
        reason = ""
        if not self.output_dir.text().strip():
            reason = "出力先フォルダを選ぶと出力できます"
        elif not (self.instance_mask.isChecked() or self.binary_mask.isChecked()):
            reason = "出力内容を 1 つ以上選ぶと出力できます"
        self.start_button.setEnabled(not reason)
        self.start_button.setToolTip(reason)

    def params(self) -> MaskExportParams:
        """出力開始の時点で、評価 ID と対象画像を固定した条件を作る。"""
        values = self.values()
        contents = set()
        if values["instance_mask"]:
            contents.add("label")
        if values["binary_mask"]:
            contents.add("binary")
        item_ids = None
        if values["classification"]:
            item_ids = [
                item.item_id
                for item in self.items
                if (item.classification or "未分類") == values["classification"]
            ]
        return MaskExportParams(
            output_parent=values["output_dir"],
            selections=list(self.selections),
            contents=contents,
            file_format=values["file_format"].lower(),
            item_ids=item_ids,
        )

    def _start(self) -> None:
        if self.job is not None:
            return
        params = self.params()
        if params.item_ids is not None and not params.item_ids:
            self._show_error("選んだ画像分類の画像がありません")
            return
        backend = self.ctx.backend
        self._params = params
        self.job = WorkerJob(
            "粒子解析用抽出結果出力",
            lambda progress, cancelled: backend.export_particle_masks(params, progress, cancelled),
            key="mask-export",
        )
        self.job.progress.connect(self._progress)
        self.job.finished.connect(self._finished)
        self.error.hide()
        self.progress.setRange(0, 0)
        self.progress_label.setText("出力しています…")
        self.progress_label.show()
        self.progress.show()
        self.start_button.setEnabled(False)
        self.cancel_button.setText("出力を中止")
        for widget in (
            self.output_dir,
            self.browse,
            self.instance_mask,
            self.binary_mask,
            self.file_format,
            self.scope,
            self.classification,
        ):
            widget.setEnabled(False)
        self.ctx.jobs.start(self.job)

    def _progress(self, done: int, total: int, _message: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.progress_label.setText(f"出力しています… {done} / {total} 枚")

    def _finished(self, ok: bool, _message: str) -> None:
        job, self.job = self.job, None
        self.progress_label.hide()
        self.progress.hide()
        self.cancel_button.setText("キャンセル")
        for widget in (
            self.output_dir,
            self.browse,
            self.instance_mask,
            self.binary_mask,
            self.file_format,
            self.scope,
        ):
            widget.setEnabled(True)
        self.classification.setEnabled(self.scope.currentIndex() == 1)
        self._update_start()
        result = job.result
        if ok and result is not None and not getattr(result, "cancelled", False):
            folder = getattr(result, "folder", None)
            n_items = (
                len(self._params.item_ids) if self._params.item_ids is not None else len(self.items)
            )
            self.summary = {
                "folder": str(folder) if folder else self._params.output_parent,
                "created": folder is not None,
                "n_candidates": len(self.selections),
                "n_images": n_items,
                "vanished_count": result.vanished_count,
                "vanished_images": result.vanished_images,
                "split_count": result.split_count,
                "split_images": result.split_images,
            }
            self.accept()
            return
        partial = getattr(result, "partial_folder", None) or getattr(
            job.error, "partial_folder", None
        )
        if partial is not None:
            self._show_error(
                "出力は完了していません。"
                f"途中までのファイルは {Path(partial).name} に残っています。"
            )
        elif ok:
            self._show_error("出力を中止しました。")
        else:
            self._show_error(user_message(job.error, "出力を開始できませんでした"))

    def _show_error(self, text: str) -> None:
        self.error.setText(text)
        self.error.show()

    def reject(self) -> None:
        if self.job is not None:
            # 出力中は「出力を中止」。1 枚ごとに中止を確認して止まる（12 章）
            self.job.cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("中止しています…")
            return
        super().reject()

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "出力先フォルダを選択")
        if directory:
            self.output_dir.setText(directory)


class MaskExportDoneDialog(QDialog):
    """粒子解析用抽出結果の出力完了の案内（承認済みモック）。"""

    FOLDER_TEXT_WIDTH = 520

    def __init__(self, summary: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.summary = summary
        self.setWindowTitle("粒子解析用抽出結果を出力しました")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        # 長いパスでダイアログが横に伸びないよう、中央を省略して全文はツールチップに出す
        full_text = f"出力先: {summary['folder']}"
        folder = QLabel(
            QFontMetrics(self.font()).elidedText(
                full_text, Qt.TextElideMode.ElideMiddle, self.FOLDER_TEXT_WIDTH
            )
        )
        folder.setToolTip(str(summary["folder"]))
        folder.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.folder_label = folder
        layout.addWidget(folder)
        layout.addWidget(
            QLabel(f"候補 {summary['n_candidates']} 件 ・ 画像 {summary['n_images']} 枚")
        )
        layout.addWidget(
            QLabel(
                f"粒子分離で消えた気泡: {summary['vanished_count']} 個"
                f"（{summary['vanished_images']} 枚）"
            )
        )
        layout.addWidget(
            QLabel(
                f"粒子分離で分断された気泡: {summary['split_count']} 個"
                f"（{summary['split_images']} 枚）"
            )
        )
        layout.addWidget(QLabel("詳細は出力フォルダの export_report.csv を見てください。"))
        buttons = QDialogButtonBox()
        self.open_button = buttons.addButton(
            "フォルダを開く", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.close_button = buttons.addButton("閉じる", QDialogButtonBox.ButtonRole.RejectRole)
        self.open_button.clicked.connect(self._open_folder)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.summary["folder"]))
