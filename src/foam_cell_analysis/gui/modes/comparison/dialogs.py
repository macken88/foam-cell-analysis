"""候補操作用ダイアログ（比較・推論設計 16.2）。"""

from __future__ import annotations

import logging
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
from ....services.comparison_service import INFERENCE_PARAM_SPECS
from ....services.models import Candidate, EvaluationRecord, Experiment, ExternalResult
from ...context import AppContext
from ...labels import config_key_label, contamination_label, format_score, model_type_label
from ...theme import numeric_font, set_style
from ...widgets.table import fit_table_columns, mark_primary, setup_table

logger = logging.getLogger(__name__)

# 完了した試行の状態（TrainingService は completed、モックの初期データは 完走）
_COMPLETED_RESULTS = {"completed", "完走"}
AP_NOTE = "AP は Cellpose 方式（TP / (TP + FP + FN)、IoU 0.50–0.95 の平均、画像平均）"
OOF_NOTE = "OOF AP は交差検証の各 fold モデルによる評価で、final.pt 自身の評価ではありません。"
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
    if not isinstance(error, ValueError):
        return fallback
    text = str(error).split(": ", 1)[0].strip()
    return text or fallback


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
    """時間のかかる処理（マスク出力・リリース登録）をワーカースレッドで動かすジョブ。

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
        inference_form.addRow(self.use_existing, self.config)
        inference_form.addRow("既存設定の値", self.config_values)
        inference_form.addRow(self.use_new)
        layout.addLayout(inference_form)
        layout.addWidget(self.params)
        layout.addStretch(1)
        layout.addWidget(self.error)
        layout.addWidget(self.buttons)
        self.experiments = {
            experiment.experiment_id: experiment
            for experiment in ctx.backend.list_experiments()
            if usable_attempts(ctx.backend, experiment)
        }
        self.experiment.addItems(list(self.experiments))
        self.experiment.currentTextChanged.connect(self._experiment_changed)
        self.attempt.currentIndexChanged.connect(self._update_ok)
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
        self.config.setEnabled(existing)
        self.config_values.setEnabled(existing)
        for widget in self.fields.values():
            widget.setEnabled(not existing)
        self._update_ok()

    def _update_ok(self, *_args) -> None:
        reason = ""
        if not self.experiments:
            reason = "完了した実験がありません"
        elif self.attempt.currentIndex() < 0:
            reason = "試行を選んでください"
        elif self.use_existing.isChecked() and not self.config.currentText():
            reason = "推論設定を選んでください"
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
        if self.use_new.isChecked():
            values = {key: widget.value() for key, widget in self.fields.items()}
            config = self.ctx.backend.create_inference_config(self.model_type_value, values)
            config_id = config.config_id
        if not config_id:
            raise ValueError("推論設定を選択してください")
        self.created = self.ctx.backend.add_candidate(
            self.experiment.currentText(), attempt, config_id, self.comment.text()
        )
        return self.created

    def _submit(self) -> None:
        try:
            self.apply()
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
    ) -> None:
        super().__init__(parent)
        self.ctx, self.candidate, self.record = ctx, candidate, record
        self.evaluation_id = record.evaluation_id
        self.read_only = candidate.status == "released"
        self.setWindowTitle(f"詳細評価 - {candidate.candidate_id}")
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

        base = ctx.backend.base_validation_version_for(candidate.candidate_id)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(base_version_text(record.validation_version, base)))
        self.base_warning = _note(BASE_MISMATCH_NOTE, "warning")
        self.base_warning.setVisible(bool(base) and base != record.validation_version)
        layout.addWidget(self.base_warning)
        layout.addWidget(
            QLabel(
                f"評価日時: {format_evaluated_at(record.completed_at)} ・ "
                f"同一画像の混入: {contamination_label(record.contamination.get('status'))}"
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

        # ---- 外部粒子解析結果（この評価に対する手入力） ----
        saved = self._saved_record()
        layout.addSpacing(8)
        layout.addWidget(QLabel("外部粒子解析結果（この評価に対する手入力。補助評価です）"))
        self.scope = QComboBox()
        self.scope.addItems(["全体", *(evaluation.per_class if evaluation else {})])
        if saved and self.scope.findText(str(saved.get("scope", ""))) >= 0:
            self.scope.setCurrentText(str(saved.get("scope")))
        self.software = QLineEdit(str(saved.get("software", "")) if saved else "")
        self.software_version = QLineEdit(str(saved.get("software_version", "")) if saved else "")
        self.software_version.setMaximumWidth(90)
        self.date = QLineEdit(str(saved.get("analyzed_on", "")) if saved else "")
        self.date.setPlaceholderText("2026-09-29")
        self.date.setMaximumWidth(110)
        meta = QHBoxLayout()
        for label, widget in (
            ("対象範囲", self.scope),
            ("解析ソフト名", self.software),
            ("版", self.software_version),
            ("解析日", self.date),
        ):
            meta.addWidget(QLabel(label))
            meta.addWidget(widget)
        layout.addLayout(meta)
        self.results = QTableWidget(0, 3)
        self.results.setHorizontalHeaderLabels(["項目名", "値", "単位"])
        setup_table(self.results, stretch_column=0)
        self.add_row = QPushButton("行を追加")
        self.remove_row = QPushButton("選択行を削除")
        result_actions = QHBoxLayout()
        result_actions.addStretch(1)
        result_actions.addWidget(self.add_row)
        result_actions.addWidget(self.remove_row)
        layout.addLayout(result_actions)
        layout.addWidget(self.results, 1)
        for item in (saved or {}).get("results", []):
            self._append_result(item.get("name", ""), item.get("value", ""), item.get("unit", ""))
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
        self.add_row.clicked.connect(lambda: self._append_result("", "", ""))
        self.remove_row.clicked.connect(
            lambda: (
                self.results.removeRow(self.results.currentRow())
                if self.results.currentRow() >= 0
                else None
            )
        )
        if self.read_only:
            self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            for widget in (
                self.add_row,
                self.remove_row,
                self.scope,
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

    def _append_result(self, name: str, value: Any, unit: str) -> None:
        row = self.results.rowCount()
        self.results.insertRow(row)
        for col, cell_value in enumerate((name, value, unit)):
            item = QTableWidgetItem(str(cell_value))
            if col == 1:
                item.setFont(numeric_font())
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results.setItem(row, col, item)
        fit_table_columns(self.results)

    def _cell(self, row: int, column: int) -> str:
        item = self.results.item(row, column)
        return item.text().strip() if item else ""

    def apply(self) -> Candidate:
        """表の外部解析結果を、開いた時点の評価 ID に結び付けて保存する。"""
        results = []
        for row in range(self.results.rowCount()):
            name, raw, unit = (self._cell(row, column) for column in range(3))
            if not (name or raw or unit):
                continue
            try:
                value: float | str = float(raw)
            except ValueError:
                value = raw
            results.append(ExternalResult(name, value, unit))
        return self.ctx.backend.save_external_results(
            self.candidate.candidate_id,
            self.evaluation_id,
            results,
            software=self.software.text().strip(),
            software_version=self.software_version.text().strip(),
            analyzed_on=self.date.text().strip(),
            scope=self.scope.currentText(),
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
        self.setWindowTitle("リリース済みモデル登録")
        self.setMinimumSize(560, 360)
        base = ctx.backend.base_validation_version_for(candidate.candidate_id)
        contamination = str(record.contamination.get("status") or "unknown")
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
            QLabel(
                f"{record.validation_version} （学習用データセットの基準: {base or '記録なし'}）"
            ),
        )
        self.base_warning = _note(BASE_MISMATCH_NOTE, "warning")
        self.base_warning.setVisible(bool(base) and base != record.validation_version)
        form.addRow("", self.base_warning)
        form.addRow(
            "評価",
            QLabel(
                f"評価日時 {format_evaluated_at(record.completed_at)} ・ "
                f"検証 AP {format_score(_ap(evaluation))}（対象 {count} 枚）"
            ),
        )
        form.addRow("同一画像の混入", QLabel(contamination_label(contamination)))
        self.contamination_note = _note(state="warning")
        if contamination == "unknown":
            self.contamination_note.setText(
                "学習データに同じ画像が含まれていないかを確認できませんでした。"
                "このことはリリース記録に残ります。"
            )
        elif contamination == "found":
            self.contamination_note.setText(
                "学習データと同じ画像が検証用データセットにあるため、リリースできません。"
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
        self.ok_button.setText("リリース済みモデルとして登録")
        self.cancel_button.setText("キャンセル")
        mark_primary(self.ok_button)
        self.buttons.accepted.connect(self._submit)
        self.buttons.rejected.connect(self.reject)
        if contamination == "found":
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
            "リリース登録",
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
            self.accept()
            return
        self.ok_button.setEnabled(True)
        self.error.setText(user_message(job.error, "リリース登録に失敗しました"))
        self.error.show()

    def reject(self) -> None:
        # コピー中は閉じない（途中で閉じても処理は止まらないため）
        if self.job is None:
            super().reject()


class MaskExportDialog(QDialog):
    """評価済みの予測から粒子解析用マスクを出力する（12 章）。

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
        self.setWindowTitle("粒子解析用マスク出力")
        self.setMinimumSize(560, 400)
        self.items = ctx.backend.list_validation_items(validation_version)
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("出力先フォルダを選んでください")
        self.browse = QPushButton("参照…")
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_dir, 1)
        output_layout.addWidget(self.browse)
        self.instance_mask = QCheckBox("インスタンスラベルマスク（モデル出力のまま）")
        self.instance_mask.setChecked(True)
        self.binary_mask = QCheckBox("粒子解析用二値マスク")
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
            "粒子解析用マスク出力",
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
    """粒子解析用マスクの出力完了の案内（承認済みモック）。"""

    FOLDER_TEXT_WIDTH = 520

    def __init__(self, summary: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.summary = summary
        self.setWindowTitle("粒子解析用マスクを出力しました")
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
