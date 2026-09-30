"""モデル比較・リリース候補一覧（比較・推論設計 16.2）。"""

import logging
import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
)

from ....services.models import Candidate, EvaluationRecord
from ...evaluation_runner import PREPARE_FAILED_TEXT
from ...labels import candidate_status_label, config_key_label, format_score, model_type_label
from ...navigation import PageId
from ...theme import Color, numeric_font, set_style
from ...widgets.marks import STATUS_MARKS, TagDelegate
from ...widgets.page_base import BasePage
from ...widgets.table import (
    add_row_context_menu,
    bind_button_action,
    fit_table_columns,
    mark_primary,
    setup_table,
)
from .dialogs import (
    CandidateDialog,
    EvaluationDialog,
    MaskExportDialog,
    MaskExportDoneDialog,
    ReleaseDialog,
)

logger = logging.getLogger(__name__)

BROKEN_TEXT = "評価結果が壊れています（再評価してください）"
BROKEN_REASON = "評価結果のファイルが壊れています。再評価してください"
NOT_APPLICABLE = "対象外"
STATE_FILTERS = ("すべて", "候補", "評価中", "リリース済み", "非採用")
COLUMN_AP, COLUMN_OOF, COLUMN_STATUS = 7, 8, 9


class _StatusMarks(dict):
    """「評価中 3 / 30」のような進捗付きの文字も、先頭の語で色を決める。"""

    def get(self, key, default=None):
        return super().get(str(key).split(" ")[0], default)


_MARKS = _StatusMarks(STATUS_MARKS)
_MARKS.setdefault("評価待ち", STATUS_MARKS["待機"])


class CandidatesPage(BasePage):
    """候補の追加・評価・比較・リリースを行う。"""

    def __init__(self, ctx, parent=None, *, show_heading: bool = True) -> None:
        super().__init__(
            ctx,
            "モデル比較・リリース",
            "候補を評価し、抽出結果を比較してリリースします。",
            parent,
            show_heading=show_heading,
        )
        self._records: dict[str, EvaluationRecord | None] = {}
        # 状態列の再描画に使う候補情報（一覧を読み込んだときの値。進捗通知では読み直さない）
        self._row_candidates: dict[str, Candidate] = {}
        self._history_available: dict[str, bool] = {}
        # 占有の変化は runner の状態が確定してから（イベントループに戻ってから）状態列へ反映する
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.setInterval(0)
        self._status_timer.timeout.connect(lambda: self._update_status_cells())
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self.refresh)
        self.validation = QComboBox()
        versions = [v.version for v in ctx.backend.list_validation_versions()]
        self.validation.addItem("すべて", "")
        for version in versions:
            self.validation.addItem(version, version)
        self.state_filter = QComboBox()
        self.state_filter.addItems(list(STATE_FILTERS))
        self.validation_menu = QMenu("検証用データセット", self)
        self.validation_actions = {}
        self.state_menu = QMenu("候補の状態で絞り込み", self)
        self.state_actions = {}
        for version in versions:
            action = self.validation_menu.addAction(version)
            action.triggered.connect(
                lambda _checked=False, value=version: self.validation.setCurrentText(value)
            )
            self.validation_actions[version] = action
        for state in STATE_FILTERS:
            action = self.state_menu.addAction(state)
            action.triggered.connect(
                lambda _checked=False, value=state: self.state_filter.setCurrentText(value)
            )
            self.state_actions[state] = action
        row = QHBoxLayout()
        row.addWidget(QLabel("検証用データセット:"))
        self.validation.setMaximumWidth(150)
        row.addWidget(self.validation)
        row.addWidget(QLabel("状態:"))
        self.state_filter.setMaximumWidth(150)
        row.addWidget(self.state_filter)
        row.addStretch(1)
        self.release_reason = QLabel()
        self.table = QTableWidget(0, 12)
        self.table.setHorizontalHeaderLabels(
            [
                "選択",
                "候補ID",
                "モデル",
                "実験",
                "学習モデル",
                "推論設定",
                "検証版",
                "検証 AP",
                "OOF AP",
                "状態",
                "外部解析",
                "コメント",
            ]
        )
        setup_table(
            self.table, stretch_column=11, selection_mode=QTableWidget.SelectionMode.SingleSelection
        )
        self.table.setItemDelegateForColumn(COLUMN_STATUS, TagDelegate(_MARKS, self.table))
        for column, width in enumerate((58, 72, 105, 82, 115, 92, 88, 66, 66, 78, 78)):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Interactive
            )
            self.table.setColumnWidth(column, width)
        self.table.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
        self.buttons: dict[str, QPushButton] = {}
        self.candidate_actions = {}
        for key, label in (
            ("add", "候補追加"),
            ("add_config", "設定を変えて候補追加"),
            ("evaluate", "評価実行"),
            ("detail", "評価詳細"),
            ("compare", "抽出結果比較"),
            ("export", "抽出結果出力"),
            ("release", "選択候補リリース"),
        ):
            button = QPushButton(label)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.buttons[key] = button
            action = QAction(label, self)
            self.candidate_actions[key] = action
            bind_button_action(button, action)
        mark_primary(self.buttons["release"])
        self.candidate_actions["stop"] = QAction("評価中止", self)
        self.candidate_actions["reject"] = QAction("非採用にする", self)
        self.candidate_actions["stop"].triggered.connect(self._stop_evaluation)
        self.candidate_actions["export"].triggered.connect(self._export_masks)
        self.candidate_actions["reject"].triggered.connect(self._reject)
        top_actions = QHBoxLayout()
        for key in ("add", "add_config", "evaluate"):
            top_actions.addWidget(self.buttons[key])
        top_actions.addLayout(row)
        self.content_layout.addLayout(top_actions)
        self.failure_note = QLabel()
        self.failure_note.setWordWrap(True)
        set_style(self.failure_note, state="error")
        self.failure_note.hide()
        self.content_layout.addWidget(self.failure_note)
        self.content_layout.addWidget(self.table, 1)
        bottom_actions = QHBoxLayout()
        for key in ("detail", "compare", "export"):
            bottom_actions.addWidget(self.buttons[key])
        bottom_actions.addStretch(1)
        bottom_actions.addWidget(self.release_reason)
        bottom_actions.addWidget(self.buttons["release"])
        self.content_layout.addLayout(bottom_actions)
        self.validation.currentTextChanged.connect(self.refresh)
        self.state_filter.currentTextChanged.connect(self.refresh)
        self.table.itemChanged.connect(lambda _item: self._update_buttons())
        self.table.itemClicked.connect(self._remember_clicked_candidate)
        self.candidate_actions["add"].triggered.connect(self._add_candidate)
        self.candidate_actions["add_config"].triggered.connect(self._add_config_candidate)
        self.candidate_actions["evaluate"].triggered.connect(self._evaluate)
        self.candidate_actions["detail"].triggered.connect(self._detail)
        self.candidate_actions["history_detail"] = QAction("別の検証版の評価を見る", self)
        self.candidate_actions["history_detail"].triggered.connect(self._history_detail)
        self.candidate_actions["compare"].triggered.connect(self._compare)
        self.candidate_actions["release"].triggered.connect(self._release)
        self.context_menu = QMenu(self)
        for key in (
            "add",
            "add_config",
            "evaluate",
            "stop",
            "detail",
            "compare",
            "export",
            "reject",
            "release",
        ):
            self.context_menu.addAction(self.candidate_actions[key])

        def select_candidate_for_context(row_index):
            clicked = self.table.item(row_index, 0)
            if clicked.checkState() == Qt.CheckState.Checked:
                return
            for row_index_existing in range(self.table.rowCount()):
                item = self.table.item(row_index_existing, 0)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if row_index_existing == row_index
                    else Qt.CheckState.Unchecked
                )

        add_row_context_menu(self.table, self.context_menu, select_candidate_for_context)
        self._pending_action = None
        runner = self.runner
        if runner is not None:
            runner.progressed.connect(self._on_progressed)
            runner.ended.connect(self._evaluation_ended)
            runner.busy_changed.connect(lambda _busy: self._refresh_timer.start())
        self.ctx.compute.changed.connect(self._on_compute_changed)
        self.refresh()

    @property
    def runner(self):
        return self.ctx.evaluation_runner

    def menu_action_groups(self):
        return {
            "file": [self.candidate_actions["export"]],
            "view": [self.validation_menu.menuAction(), self.state_menu.menuAction(), None],
            "candidate": [
                self.candidate_actions[key]
                for key in (
                    "add",
                    "add_config",
                    "evaluate",
                    "stop",
                    "detail",
                    "history_detail",
                    "compare",
                    "export",
                )
            ]
            + [None, self.candidate_actions["reject"], self.candidate_actions["release"]],
        }

    def on_enter(self, params: dict) -> None:
        """他画面から候補追加を受け付ける。"""
        self._refresh_validation_versions()
        if params.get("action") == "add_candidate":
            preset = {
                key: params[key] for key in ("experiment_id", "attempt", "comment") if key in params
            }
            QTimer.singleShot(0, lambda: self._show_add_dialog(preset))

    def refresh_on_activate(self) -> None:
        """検証版と候補状態を読み直し、画面の選択を保つ。"""
        current_row = self.table.currentRow()
        current_id = (
            self.table.item(current_row, 1).text()
            if current_row >= 0 and self.table.item(current_row, 1)
            else None
        )
        current_column = max(0, self.table.currentColumn())
        scroll_value = self.table.verticalScrollBar().value()
        self._activation_selection = self._checked_ids()
        self._activation_current_id = current_id
        self._activation_current_column = current_column
        self._activation_scroll_value = scroll_value
        try:
            self._refresh_validation_versions()
        finally:
            del self._activation_selection
            del self._activation_current_id
            del self._activation_current_column
            del self._activation_scroll_value

    def _refresh_validation_versions(self) -> None:
        """検証用版の候補を更新し、選択可能な版を維持する。"""
        selected_version = self.validation.currentText()
        available_versions = [
            version.version for version in self.ctx.backend.list_validation_versions()
        ]
        self.validation.blockSignals(True)
        self.validation.clear()
        self.validation.addItem("すべて", "")
        for value in available_versions:
            self.validation.addItem(value, value)
        if selected_version in available_versions:
            target_version = selected_version
        else:
            target_version = "すべて"
        self.validation.setCurrentText(target_version)
        self.validation.blockSignals(False)
        for value in available_versions:
            if value not in self.validation_actions:
                action = self.validation_menu.addAction(value)
                action.triggered.connect(
                    lambda _checked=False, selected=value: self.validation.setCurrentText(selected)
                )
                self.validation_actions[value] = action
        self.refresh()

    # ---- 評価の状態（runner から導出する。5.5） ----

    def _is_active(self, candidate_id: str) -> bool:
        runner = self.runner
        return bool(runner is not None and runner.is_evaluation_active(candidate_id))

    def _status_text(self, candidate) -> str:
        """状態列の文字。評価中は進捗（n / 全体）を付ける。"""
        runner = self.runner
        candidate_id = candidate.candidate_id
        if runner is not None and runner.is_running(candidate_id):
            progress = self.ctx.backend.get_evaluation_progress(candidate_id)
            if progress is not None and progress.total:
                return f"評価中 {progress.completed} / {progress.total}"
            return "評価中"
        if self._is_active(candidate_id):
            return "評価待ち"
        return candidate_status_label(candidate.status)

    def _status_tooltip(self, text: str) -> str:
        if text.startswith("評価待ち"):
            return self.ctx.compute.wait_message("evaluation") or "前の候補の評価を待っています"
        return ""

    def _on_progressed(self, candidate_id: str) -> None:
        """進捗イベントでは通知された候補の行だけを書き換える。"""
        self._update_status_cells({candidate_id})

    def _on_compute_changed(self) -> None:
        """占有の開始・待機・返却では、評価待ち⇔評価中が複数行で変わりうる。"""
        self._status_timer.start()
        self._update_buttons()

    def _update_status_cells(self, candidate_ids: set[str] | None = None) -> None:
        """状態列だけを書き換える（None なら全行）。

        候補・データセット・評価成果物のファイルは読み直さない。候補の状態は一覧読込み時の
        値を使い、評価中・評価待ちと進捗は runner とメモリ上の進捗から作る。
        """
        changed = False
        self.table.blockSignals(True)
        try:
            for row in range(self.table.rowCount()):
                id_item = self.table.item(row, 1)
                status_item = self.table.item(row, COLUMN_STATUS)
                if id_item is None or status_item is None:
                    continue
                candidate_id = id_item.text()
                if candidate_ids is not None and candidate_id not in candidate_ids:
                    continue
                candidate = self._row_candidates.get(candidate_id)
                if candidate is None:
                    continue
                text = self._status_text(candidate)
                # 待機の理由（ツールチップ）は文字が同じでも変わりうる
                status_item.setToolTip(self._status_tooltip(text))
                if status_item.text() != text:
                    status_item.setText(text)
                    changed = True
        finally:
            self.table.blockSignals(False)
        if changed:
            fit_table_columns(self.table)

    @staticmethod
    def _failure_reason(outcome) -> str:
        """失敗の理由を、画面に出してよい日本語にする。生のパスや英語の文は出さない。"""
        text = str(outcome.message or "").strip()
        readable = (
            re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text) is not None
            and re.search(r"[A-Za-z]:[\\/]|\\|/[\w.-]+/|Traceback", text) is None
        )
        if readable:
            return text.rstrip("。") + "。"
        if outcome.reason in {"prepare_failed", "start_failed"}:
            return PREPARE_FAILED_TEXT
        return "詳しくはログを確認してください。"

    def _evaluation_ended(self, outcome) -> None:
        candidate_id = outcome.candidate_id
        if outcome.status == "completed":
            message = f"{candidate_id} の評価が完了しました"
        elif outcome.status == "stopped":
            message = f"{candidate_id} の評価を中止しました"
        else:
            logger.warning("%s の評価に失敗しました: %s", candidate_id, outcome.message)
            message = f"{candidate_id} の評価に失敗しました"
            self.failure_note.setText(f"{message}。{self._failure_reason(outcome)}")
            self.failure_note.show()
        self.ctx.status.show_message(message)
        self._refresh_timer.start()

    # ---- 表示 ----

    def _checked_ids(self) -> set[str]:
        return {
            self.table.item(row, 1).text()
            for row in range(self.table.rowCount())
            if self.table.item(row, 0)
            and self.table.item(row, 0).checkState() == Qt.CheckState.Checked
            and self.table.item(row, 1)
        }

    def _load_record(self, candidate_id: str, version: str) -> EvaluationRecord | None:
        if not version:
            return None
        try:
            return self.ctx.backend.get_candidate_evaluation(candidate_id, version)
        except (KeyError, ValueError, OSError) as error:
            logger.warning("%s の評価を読めません: %s", candidate_id, error)
            return None

    def refresh(self, _value: str = "") -> None:
        """選択中の検証版で候補評価を再表示する。"""
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
        version, state = self.validation.currentData() or "", self.state_filter.currentText()
        selected_ids = getattr(self, "_activation_selection", None)
        if selected_ids is None:
            selected_ids = self._checked_ids()
        current_id = getattr(self, "_activation_current_id", None)
        if current_id is None and self.table.currentRow() >= 0:
            current_item = self.table.item(self.table.currentRow(), 1)
            current_id = current_item.text() if current_item else None
        current_column = getattr(
            self, "_activation_current_column", max(0, self.table.currentColumn())
        )
        scroll_value = getattr(
            self, "_activation_scroll_value", self.table.verticalScrollBar().value()
        )
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        candidates = self.ctx.backend.list_candidates()
        if version:
            candidates = [
                candidate for candidate in candidates if candidate.validation_version == version
            ]
        self._row_candidates = {candidate.candidate_id: candidate for candidate in candidates}
        self._records = {
            candidate.candidate_id: self._load_record(
                candidate.candidate_id, version or (candidate.validation_version or "")
            )
            for candidate in candidates
        }
        self._history_available = {}
        for candidate in candidates:
            try:
                self._history_available[candidate.candidate_id] = any(
                    record.validation_version != candidate.validation_version
                    and record.status == "completed"
                    and not record.broken
                    and record.evaluation is not None
                    for record in self.ctx.backend.list_candidate_evaluations(
                        candidate.candidate_id
                    )
                )
            except (KeyError, ValueError, OSError):
                self._history_available[candidate.candidate_id] = False
        usable = {
            candidate_id: record.evaluation.overall_map
            for candidate_id, record in self._records.items()
            if record is not None and not record.broken and record.evaluation is not None
        }
        configs = {item.config_id: item for item in self.ctx.backend.list_inference_configs()}
        best_score = max((value for value in usable.values() if value is not None), default=None)
        for candidate in candidates:
            status_text = self._status_text(candidate)
            if state != "すべて" and not status_text.startswith(
                "評価" if state == "評価中" else state
            ):
                continue
            model_type = (
                candidate.snapshot.experiment_config.get("model", {}).get("type")
                if candidate.snapshot is not None
                else self.ctx.backend.get_experiment(candidate.experiment_id).model_type
            )
            record = self._records.get(candidate.candidate_id)
            if record is not None and record.broken:
                ap_text = BROKEN_TEXT
            elif candidate.candidate_id in usable:
                ap_text = format_score(usable[candidate.candidate_id])
            else:
                ap_text = "未評価"
            oof = candidate.oof_evaluation
            oof_matching = candidate.oof_applicability == "matching"
            if oof is None:
                oof_text = "—"
            elif oof_matching:
                oof_text = format_score(oof.overall_map)
            else:
                oof_text = NOT_APPLICABLE
            external = candidate.external_summary
            inference = configs.get(candidate.inference_config_id)
            trained_params = (
                candidate.snapshot.training_eval_params if candidate.snapshot is not None else None
            )
            if inference is None or not trained_params or candidate.oof_applicability == "unknown":
                inference_text = "確認できません"
                inference_tip = "学習時または候補の推論設定を確認できません"
            elif candidate.effective_params == trained_params:
                inference_text = "学習時と同じ"
                inference_tip = "学習時の推論設定と一致しています"
            else:
                inference_text = "変更あり"
                keys = sorted(set(candidate.effective_params) | set(trained_params))
                inference_tip = "\n".join(
                    f"{config_key_label(key)}: 学習時 {trained_params.get(key, '—')} → "
                    f"現在 {candidate.effective_params.get(key, '—')}"
                    for key in keys
                    if candidate.effective_params.get(key) != trained_params.get(key)
                )
            if external:
                external_text = (
                    f"{external.get('mean', '—')} {external.get('unit', '')} "
                    f"({external.get('n_images', 0)}/{external.get('n_total', 0)})"
                )
            else:
                external_text = "記録なし" if not candidate.external_results else "旧記録あり"
            values = [
                candidate.candidate_id,
                model_type_label(model_type),
                candidate.experiment_id,
                f"試行 {candidate.source_attempt_number} / final.pt",
                inference_text,
                candidate.validation_version or "確認できません",
                ap_text,
                oof_text,
                status_text,
                external_text,
                candidate.comment,
            ]
            row = self.table.rowCount()
            self.table.insertRow(row)
            check = QTableWidgetItem()
            check.setFlags(
                (check.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                & ~Qt.ItemFlag.ItemIsEditable
            )
            check.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, check)
            if candidate.candidate_id in selected_ids:
                check.setCheckState(Qt.CheckState.Checked)
            for col, value in enumerate(values, 1):
                item = QTableWidgetItem(str(value))
                if col == 5:
                    item.setToolTip(inference_tip)
                if col in (1, 3, 5, 6, 7) and value not in (BROKEN_TEXT, NOT_APPLICABLE):
                    item.setFont(numeric_font())
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if value in ("未評価", NOT_APPLICABLE):
                    item.setForeground(QColor(Color.SLATE))
                if col == COLUMN_AP and value == BROKEN_TEXT:
                    item.setForeground(QColor(Color.ERROR))
                    item.setToolTip(BROKEN_REASON)
                if col == COLUMN_OOF and value == NOT_APPLICABLE:
                    item.setToolTip(candidate.oof_reason or "学習時の評価条件と一致しません")
                if col == COLUMN_STATUS:
                    item.setToolTip(self._status_tooltip(value))
                comparable_filter = bool(version)
                bold = (
                    comparable_filter
                    and col == COLUMN_AP
                    and best_score is not None
                    and usable.get(candidate.candidate_id) == best_score
                )
                if bold:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        if current_id:
            for row in range(self.table.rowCount()):
                if self.table.item(row, 1).text() == current_id:
                    self.table.setCurrentCell(
                        row,
                        min(
                            current_column,
                            self.table.columnCount() - 1,
                        ),
                    )
                    break
        self.table.verticalScrollBar().setValue(scroll_value)
        fit_table_columns(self.table)
        self._update_buttons()

    def _selected(self):
        selected = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                selected.append(self.ctx.backend.get_candidate(self.table.item(row, 1).text()))
        return selected

    def _usable_record(self, candidate_id: str) -> EvaluationRecord | None:
        record = self._records.get(candidate_id)
        if record is None or record.broken or record.evaluation is None:
            return None
        return record

    def _remember_clicked_candidate(self, item: QTableWidgetItem) -> None:
        """チェック欄をクリックした候補をキーボード操作対象にする。"""
        if item.column() == 0:
            self.table.setCurrentCell(item.row(), 0)

    def _set_action(self, key: str, reason: str) -> None:
        """reason が空なら有効、あれば無効にして理由をツールチップに出す。"""
        action = self.candidate_actions[key]
        action.setToolTip(reason)
        self.set_menu_action_enabled(action, not reason)

    def _update_buttons(self) -> None:
        selected = self._selected()
        one = len(selected) == 1
        version = self.validation.currentText()
        active = [c for c in selected if self._is_active(c.candidate_id)]
        broken = [
            c
            for c in selected
            if self._records.get(c.candidate_id) is not None
            and self._records[c.candidate_id].broken
        ]
        unevaluated = [c for c in selected if self._usable_record(c.candidate_id) is None]

        evaluate_reason = ""
        if not version:
            evaluate_reason = "検証用データセットがありません"
        elif self.ctx.compute.external_block:
            evaluate_reason = self.ctx.compute.external_block
        elif not selected:
            evaluate_reason = "候補を 1 つ以上選ぶと使えます"
        elif any(c.status != "candidate" for c in selected):
            evaluate_reason = "候補状態のモデルを選ぶと使えます"
        elif active:
            evaluate_reason = "評価中・評価待ちの候補が含まれています"
        runner = self.runner
        if runner is not None and runner.is_busy:
            self.buttons["evaluate"].setText("評価中止")
            self.candidate_actions["evaluate"].setText("評価中止")
            self._set_action("evaluate", "")
        else:
            self.buttons["evaluate"].setText("評価実行")
            self.candidate_actions["evaluate"].setText("評価実行")
            self._set_action("evaluate", evaluate_reason)
        self._set_action(
            "stop",
            "" if runner is not None and runner.is_busy else "評価を実行していないときは使えません",
        )

        self._set_action(
            "add_config",
            "設定変更の元にする候補を 1 件選んでください" if not one else "",
        )

        detail_reason = ""
        if not one:
            detail_reason = "評価済みの候補を 1 つ選ぶと使えます"
        elif broken:
            detail_reason = BROKEN_REASON
        elif unevaluated:
            detail_reason = "評価済みの候補を 1 つ選ぶと使えます"
        self._set_action("detail", detail_reason)

        historical_reason = ""
        if not one:
            historical_reason = "候補を 1 件選ぶと過去評価を表示できます"
        elif not self._history_available.get(selected[0].candidate_id, False):
            historical_reason = "別の検証版で完了した評価はありません"
        self._set_action("history_detail", historical_reason)

        compare_reason = ""
        if not 2 <= len(selected) <= 4:
            compare_reason = "同じ検証用データセットで評価済みの候補を 2〜4 件選んでください"
        elif len({candidate.validation_version for candidate in selected}) != 1:
            compare_reason = "同じ検証用データセットの候補を選んでください"
        elif any(self._usable_record(candidate.candidate_id) is None for candidate in selected):
            compare_reason = "選択候補を評価してから比較できます"
        self._set_action("compare", compare_reason)

        release_reason = ""
        if not one:
            release_reason = "評価済みの候補を 1 つ選ぶとリリースできます"
        elif selected[0].status != "candidate":
            release_reason = "候補状態のモデルを選ぶとリリースできます"
        elif active:
            release_reason = "評価中・評価待ちの候補はリリースできません"
        elif broken:
            release_reason = BROKEN_REASON
        elif unevaluated:
            release_reason = "評価済みの候補を 1 つ選ぶとリリースできます"
        self._set_action("release", release_reason)
        self.release_reason.setText(release_reason)
        self.release_reason.setVisible(bool(release_reason))

        export_reason = ""
        if not selected:
            export_reason = "候補を選ぶと使えます"
        elif broken:
            export_reason = BROKEN_REASON
        elif unevaluated:
            export_reason = "選んだ候補を、この検証用データセットで評価してから使えます"
        self._set_action("export", export_reason)

        rejectable = [
            c for c in selected if c.status == "candidate" and not self._is_active(c.candidate_id)
        ]
        self._set_action("reject", "" if rejectable else "候補状態の行を 1 つ以上選ぶと使えます")

    # ---- 操作 ----

    def _show_add_dialog(self, preset=None) -> None:
        dialog = CandidateDialog(self.ctx, self, preset)
        if dialog.exec():
            if dialog.duplicate_candidate_id:
                self.validation.setCurrentIndex(0)
                self.state_filter.setCurrentText("すべて")
                self.refresh()
                for row in range(self.table.rowCount()):
                    if self.table.item(row, 1).text() == dialog.duplicate_candidate_id:
                        self.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
                        self.table.setCurrentCell(row, 0)
                        self.ctx.status.show_message(
                            "同じ推論設定の候補が登録済みのため、その候補を選択しました"
                        )
                        break
                return
            if dialog.created is None:
                try:
                    dialog.apply()
                except ValueError as error:
                    QMessageBox.warning(self, "候補を追加できません", str(error))
            self.refresh()

    def _add_candidate(self) -> None:
        self._show_add_dialog()

    def _add_config_candidate(self) -> None:
        selected = self._selected()
        if len(selected) != 1:
            return
        candidate = selected[0]
        preset = {
            "experiment_id": candidate.experiment_id,
            "attempt": candidate.source_attempt_number,
            "comment": candidate.comment,
            "inference_params": candidate.effective_params,
            "prefer_new_config": True,
        }
        self._show_add_dialog(preset)

    def _evaluate(self) -> None:
        runner = self.runner
        if runner is not None and runner.is_busy:
            self._stop_evaluation()
            return
        ids = [
            item.candidate_id
            for item in self._selected()
            if item.status == "candidate" and not self._is_active(item.candidate_id)
        ]
        if runner is None or not ids:
            return
        self.failure_note.hide()
        try:
            runner.start(ids)
        except ValueError as error:
            QMessageBox.warning(self, "評価を開始できません", str(error))
        self.refresh()

    def _stop_evaluation(self) -> None:
        runner = self.runner
        if runner is not None and runner.is_busy:
            runner.request_stop("user_stop")
        self.refresh()

    def _selected_record(self):
        selected = self._selected()
        if len(selected) != 1:
            return None, None
        candidate = selected[0]
        # 開いた時点の評価 ID を固定する（7.7）
        record = self._load_record(
            candidate.candidate_id,
            self.validation.currentData() or (candidate.validation_version or ""),
        )
        if record is None or record.broken or record.evaluation is None:
            QMessageBox.warning(self, "評価結果を開けません", BROKEN_REASON)
            self.refresh()
            return None, None
        return candidate, record

    def _detail(self) -> None:
        candidate, record = self._selected_record()
        if candidate is None:
            return
        EvaluationDialog(self.ctx, candidate, record, self).exec()
        self.refresh()

    def _history_detail(self) -> None:
        selected = self._selected()
        if len(selected) != 1:
            return
        candidate = selected[0]
        records = [
            record
            for record in self.ctx.backend.list_candidate_evaluations(candidate.candidate_id)
            if record.validation_version != candidate.validation_version
            and record.status == "completed"
            and not record.broken
            and record.evaluation is not None
        ]
        if not records:
            return
        records.sort(key=lambda record: (record.validation_version, record.evaluation_id))
        labels = [f"{record.validation_version} / {record.evaluation_id}" for record in records]
        label, accepted = QInputDialog.getItem(
            self, "過去評価を表示", "検証版 / 評価 ID", labels, 0, False
        )
        if not accepted:
            return
        record = records[labels.index(label)]
        EvaluationDialog(self.ctx, candidate, record, self, read_only=True).exec()

    def _compare(self) -> None:
        selected = self._selected()
        if not 2 <= len(selected) <= 4:
            self.ctx.status.show_message(
                "抽出結果比較は、同じ検証用データセットで評価済みの候補 2〜4 件を選んでください"
            )
            return
        versions = {candidate.validation_version for candidate in selected}
        if len(versions) != 1:
            self.ctx.status.show_message("同じ検証用データセットの候補を選んでください")
            return
        validation_version = next(iter(versions))
        self.ctx.navigator.navigate(
            PageId.MASK_COMPARISON,
            validation_version=validation_version,
            candidate_ids=[c.candidate_id for c in selected],
        )

    def _export_masks(self) -> None:
        selected = self._selected()
        versions = {candidate.validation_version for candidate in selected}
        if len(versions) != 1 or not next(iter(versions), None):
            self.ctx.status.show_message("同じ検証用データセットの候補を選んでください")
            return
        selections = []
        for candidate in selected:
            record = self._usable_record(candidate.candidate_id)
            if record is None:
                return
            selections.append((candidate.candidate_id, record.evaluation_id))
        if not selections:
            return
        dialog = MaskExportDialog(self.ctx, selections, next(iter(versions)), self)
        if dialog.exec() and dialog.summary is not None:
            MaskExportDoneDialog(dialog.summary, self).exec()

    def _reject(self) -> None:
        candidates = [
            c
            for c in self._selected()
            if c.status == "candidate" and not self._is_active(c.candidate_id)
        ]
        if (
            candidates
            and QMessageBox.question(self, "確認", "選択候補を非採用にしますか？")
            == QMessageBox.StandardButton.Yes
        ):
            for candidate in candidates:
                try:
                    self.ctx.backend.reject_candidate(candidate.candidate_id)
                except ValueError as error:
                    QMessageBox.warning(self, "非採用にできません", str(error))
            self.refresh()

    def _release(self) -> None:
        candidate, record = self._selected_record()
        if candidate is None:
            return
        dialog = ReleaseDialog(self.ctx, candidate, record, self)
        if not dialog.exec() or dialog.model is None:
            self.refresh()
            return
        model = dialog.model
        self.refresh()
        answer = QMessageBox.question(self, "登録完了", "モデル振り分け画面を開きますか？")
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.navigator.navigate(PageId.RELEASED_MODELS, select=model.model_id)
