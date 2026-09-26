"""モデル比較・リリース候補一覧。"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
)

from ...jobs import FakeJob
from ...labels import candidate_status_label, format_score, model_type_label
from ...navigation import PageId
from ...theme import Color, numeric_font
from ...widgets.marks import STATUS_MARKS, TagDelegate
from ...widgets.page_base import BasePage
from ...widgets.table import fit_table_columns, mark_primary, setup_table
from .dialogs import CandidateDialog, EvaluationDialog, MaskExportDialog, ReleaseDialog


class CandidatesPage(BasePage):
    """候補の追加・評価・比較・リリースを行う。"""

    def __init__(self, ctx, parent=None, *, show_heading: bool = True) -> None:
        super().__init__(
            ctx,
            "モデル比較・リリース",
            "検証用データセットを切り替えて候補を比較します。",
            parent,
            show_heading=show_heading,
        )
        self.validation = QComboBox()
        self.validation.addItems([v.version for v in ctx.backend.list_validation_versions()])
        latest = self.validation.findText("val_v003")
        if latest >= 0:
            self.validation.setCurrentIndex(latest)
        self.state_filter = QComboBox()
        self.state_filter.addItems(["すべて", "候補", "評価中", "リリース済み", "非採用"])
        row = QHBoxLayout()
        row.addWidget(QLabel("検証用データセット:"))
        self.validation.setMaximumWidth(150)
        row.addWidget(self.validation)
        row.addWidget(QLabel("状態:"))
        self.state_filter.setMaximumWidth(150)
        row.addWidget(self.state_filter)
        row.addStretch(1)
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "選択",
                "候補ID",
                "モデル",
                "実験",
                "最終学習モデル",
                "推論設定",
                "検証 mAP",
                "OOF mAP",
                "状態",
                "外部解析",
                "コメント",
            ]
        )
        setup_table(
            self.table, stretch_column=10, selection_mode=QTableWidget.SelectionMode.SingleSelection
        )
        self.table.setItemDelegateForColumn(
            8,
            TagDelegate({label: colors for label, colors in STATUS_MARKS.items()}, self.table),
        )
        for column, width in enumerate((58, 72, 105, 82, 115, 92, 66, 66, 78, 78)):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Interactive
            )
            self.table.setColumnWidth(column, width)
        self.table.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
        self.buttons: dict[str, QPushButton] = {}
        button_row = QHBoxLayout()
        for key, label in (
            ("add", "候補を追加…"),
            ("evaluate", "評価を実行"),
            ("detail", "詳細評価を見る…"),
            ("compare", "マスク比較"),
            ("release", "選択候補をリリース…"),
        ):
            button = QPushButton(label)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.buttons[key] = button
            button_row.addWidget(button)
        for key in ("release",):
            mark_primary(self.buttons[key])
        self.more_button = QPushButton("その他 ▾")
        self.more_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.more_menu = QMenu(self.more_button)
        self.menu_actions = {}
        for key, label, callback in (
            ("export", "粒子解析用マスク出力…", self._export_masks),
            ("reject", "非採用にする", self._reject),
        ):
            action = self.more_menu.addAction(label)
            action.triggered.connect(callback)
            self.menu_actions[key] = action
        self.more_button.setMenu(self.more_menu)
        self.content_layout.addLayout(row)
        self.content_layout.addWidget(self.table, 1)
        self.content_layout.addLayout(button_row)
        button_row.insertWidget(4, self.more_button)
        button_row.addStretch(1)
        self.validation.currentTextChanged.connect(self.refresh)
        self.state_filter.currentTextChanged.connect(self.refresh)
        self.table.itemChanged.connect(lambda _item: self._update_buttons())
        self.table.itemClicked.connect(self._remember_clicked_candidate)
        self.buttons["add"].clicked.connect(self._add_candidate)
        self.buttons["evaluate"].clicked.connect(self._evaluate)
        self.buttons["detail"].clicked.connect(self._detail)
        self.buttons["compare"].clicked.connect(self._compare)
        self.buttons["release"].clicked.connect(self._release)
        self._pending_action = None
        self.refresh()

    def on_enter(self, params: dict) -> None:
        """他画面から候補追加を受け付ける。"""
        self._refresh_validation_versions()
        if params.get("action") == "add_candidate":
            preset = {key: params[key] for key in ("experiment_id", "checkpoint") if key in params}
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
        self._activation_selection = {
            self.table.item(row, 1).text()
            for row in range(self.table.rowCount())
            if self.table.item(row, 0)
            and self.table.item(row, 0).checkState() == Qt.CheckState.Checked
            and self.table.item(row, 1)
        }
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
        self.validation.addItems(available_versions)
        target_version = (
            selected_version
            if selected_version in available_versions
            else available_versions[-1]
            if available_versions
            else ""
        )
        self.validation.setCurrentText(target_version)
        self.validation.blockSignals(False)
        self.refresh()

    def refresh(self, _value: str = "") -> None:
        """選択中の検証版で候補評価を再表示する。"""
        version, state = self.validation.currentText(), self.state_filter.currentText()
        selected_ids = getattr(self, "_activation_selection", set())
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        candidates = self.ctx.backend.list_candidates()
        scores = [
            candidate.evaluations[version].overall_map
            for candidate in candidates
            if version in candidate.evaluations
        ]
        best_score = max(scores, default=None)
        oof_scores = [c.oof_evaluation.overall_map for c in candidates if c.oof_evaluation]
        best_oof = max(oof_scores, default=None)
        for candidate in candidates:
            if (
                state != "すべて"
                and candidate.status
                != {
                    "候補": "candidate",
                    "評価中": "evaluating",
                    "リリース済み": "released",
                    "非採用": "rejected",
                }[state]
            ):
                continue
            experiment = self.ctx.backend.get_experiment(candidate.experiment_id)
            evaluation = candidate.evaluations.get(version)
            values = [
                candidate.candidate_id,
                model_type_label(experiment.model_type),
                candidate.experiment_id,
                candidate.checkpoint,
                candidate.inference_config_id,
                format_score(evaluation.overall_map) if evaluation else "未評価",
                format_score(candidate.oof_evaluation.overall_map)
                if candidate.oof_evaluation
                else "—",
                candidate_status_label(candidate.status),
                "あり" if candidate.external_results else "—",
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
                if col in (1, 3, 4, 5, 6, 7):
                    item.setFont(numeric_font())
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if col == 6 and value == "未評価":
                    item.setForeground(QColor(Color.SLATE))
                if col == 6 and evaluation and evaluation.overall_map == best_score:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                if (
                    col == 7
                    and candidate.oof_evaluation
                    and candidate.oof_evaluation.overall_map == best_oof
                ):
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        current_id = getattr(self, "_activation_current_id", None)
        if current_id:
            for row in range(self.table.rowCount()):
                if self.table.item(row, 1).text() == current_id:
                    self.table.setCurrentCell(
                        row,
                        min(
                            getattr(self, "_activation_current_column", 0),
                            self.table.columnCount() - 1,
                        ),
                    )
                    break
        if hasattr(self, "_activation_scroll_value"):
            self.table.verticalScrollBar().setValue(self._activation_scroll_value)
        fit_table_columns(self.table)
        self._update_buttons()

    def _selected(self):
        selected = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                selected.append(self.ctx.backend.get_candidate(self.table.item(row, 1).text()))
        return selected

    def _remember_clicked_candidate(self, item: QTableWidgetItem) -> None:
        """チェック欄をクリックした候補をキーボード操作対象にする。"""
        if item.column() == 0:
            self.table.setCurrentCell(item.row(), 0)

    def _update_buttons(self) -> None:
        selected = self._selected()
        one = len(selected) == 1
        self.buttons["evaluate"].setEnabled(
            bool(selected) and all(c.status == "candidate" for c in selected)
        )
        self.buttons["detail"].setEnabled(
            one and self.validation.currentText() in selected[0].evaluations
        )
        self.buttons["compare"].setEnabled(len(selected) >= 2)
        release_enabled = (
            one
            and selected[0].status == "candidate"
            and self.validation.currentText() in selected[0].evaluations
        )
        self.buttons["release"].setEnabled(release_enabled)
        reason = ""
        if not one:
            reason = "リリース候補を1件選択してください。"
        elif selected[0].status != "candidate":
            reason = "候補状態のモデルのみリリースできます。"
        elif self.validation.currentText() not in selected[0].evaluations:
            reason = "選択中の検証用データセットで評価を完了してください。"
        self.buttons["release"].setToolTip(reason)
        self.menu_actions["export"].setEnabled(bool(selected))
        self.menu_actions["reject"].setEnabled(any(c.status == "candidate" for c in selected))

    def _show_add_dialog(self, preset=None) -> None:
        dialog = CandidateDialog(self.ctx, self, preset)
        if dialog.exec():
            try:
                dialog.apply()
            except ValueError as error:
                QMessageBox.warning(self, "候補を追加できません", str(error))
            self.refresh()

    def _add_candidate(self) -> None:
        self._show_add_dialog()

    def _evaluate(self) -> None:
        selected = self._selected()
        ids = [item.candidate_id for item in selected]
        version = self.validation.currentText()
        self.ctx.backend.start_evaluation(ids, version)
        self.refresh()
        job = FakeJob(
            "候補評価",
            total_steps=4,
            on_step=lambda step: self._finish_evaluation(ids, version) if step == 4 else None,
        )
        job.finished.connect(lambda _ok, _msg: self.refresh())
        self.ctx.jobs.start(job)

    def _finish_evaluation(self, ids, version) -> None:
        for candidate_id in ids:
            self.ctx.backend.evaluate_candidate(candidate_id, version)

    def _detail(self) -> None:
        selected = self._selected()
        if selected:
            EvaluationDialog(self.ctx, selected[0], self.validation.currentText(), self).exec()
            self.refresh()

    def _compare(self) -> None:
        self.ctx.navigator.navigate(
            PageId.MASK_COMPARISON,
            validation_version=self.validation.currentText(),
            candidate_ids=[c.candidate_id for c in self._selected()[:4]],
        )

    def _export_masks(self) -> None:
        ids = [c.candidate_id for c in self._selected()]
        dialog = MaskExportDialog(self)
        if not dialog.exec():
            return
        job = FakeJob("粒子解析用マスク出力", total_steps=3, key="mask-export")
        self.ctx.jobs.start(job)
        QMessageBox.information(
            self,
            "出力を開始しました",
            f"{len(ids)}候補のマスク出力を開始しました。実ファイルは作成しません。",
        )

    def _reject(self) -> None:
        candidates = [c for c in self._selected() if c.status == "candidate"]
        if (
            candidates
            and QMessageBox.question(self, "確認", "選択候補を非採用にしますか？")
            == QMessageBox.StandardButton.Yes
        ):
            for candidate in candidates:
                self.ctx.backend.reject_candidate(candidate.candidate_id)
            self.refresh()

    def _release(self) -> None:
        candidate = self._selected()[0]
        dialog = ReleaseDialog(candidate, self)
        if dialog.exec():
            model = self.ctx.backend.release_candidate(
                candidate.candidate_id, dialog.comment.text(), self.validation.currentText()
            )
            self.refresh()
            answer = QMessageBox.question(self, "登録完了", "モデル振り分け画面を開きますか？")
            if answer == QMessageBox.StandardButton.Yes:
                self.ctx.navigator.navigate(PageId.RELEASED_MODELS, select=model.model_id)
