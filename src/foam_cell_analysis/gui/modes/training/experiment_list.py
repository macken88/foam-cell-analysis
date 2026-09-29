"""実験検索、実行状態、学習結果の一覧画面。"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ....services.models import Experiment
from ...context import AppContext
from ...labels import (
    classification_label,
    config_key_label,
    experiment_status_label,
    format_bytes,
    format_datetime,
    format_score,
    model_type_label,
    quality_filter_label,
)
from ...navigation import PageId
from ...theme import Color, numeric_font, set_style
from ...widgets.chart import LineChart
from ...widgets.marks import STATUS_MARKS, TagDelegate
from ...widgets.page_base import BasePage
from ...widgets.table import (
    add_row_context_menu,
    bind_button_action,
    fit_table_columns,
    mark_primary,
    setup_table,
)
from ..comparison.dialogs import FINAL_PRUNED_REASON, usable_attempts
from .cleanup_dialog import ArtifactCleanupDialog, cleanup_status_message
from .dialogs import ExperimentCompareDialog, SendToCandidatesDialog, flatten_config

RETRY_LABEL = "同じ設定でやり直す"
CLEANUP_LABEL = "成果物を整理…"
PRUNED_TEXT = "削除済み"
RETRY_TIP = (
    "同じ実験の新しい試行として、同じ設定で最初から学習し直します。"
    "キューに追加して実行します（学習中のときはキューの末尾で順番を待ちます）。"
)
STOP_TIP = (
    "今の学習をすぐに止め、キューも止めます。今の学習は「中断」になり、"
    "途中までの結果だけが残ります。止める前に確認します。"
)
DELETABLE_STATUSES = {"draft", "stopped", "failed", "completed"}


class ExperimentListPage(BasePage):
    """実験一覧と詳細を表示し、実験操作へ誘導する。"""

    def __init__(
        self, ctx: AppContext, parent: QWidget | None = None, *, show_heading: bool = True
    ) -> None:
        super().__init__(
            ctx,
            "実験一覧",
            "実験の状態、設定、学習結果を確認します。",
            parent,
            show_heading=show_heading,
        )
        self.study_filter = QComboBox()
        self.study_filter.addItem("すべて")
        self.study_filter.setMaximumWidth(150)
        self.model_filter = QComboBox()
        self.model_filter.addItems(["すべて", "Mask R-CNN", "Cellpose"])
        self.model_filter.setMaximumWidth(150)
        self.state_filter = QComboBox()
        self.state_filter.addItems(["すべて", "下書き", "待機", "実行中", "完了", "失敗", "中断"])
        self.state_filter.setMaximumWidth(120)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("実験群"))
        filter_row.addWidget(self.study_filter)
        filter_row.addWidget(QLabel("モデル"))
        filter_row.addWidget(self.model_filter)
        filter_row.addWidget(QLabel("状態"))
        filter_row.addWidget(self.state_filter)
        filter_row.addStretch(1)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "選択",
                "ID",
                "実験群",
                "モデル",
                "データセット",
                "データ拡張",
                "状態",
                "進捗",
                "OOF AP",
                "途中保存モデル",
            ]
        )
        self.column_button = QPushButton("表示する列 ▾")
        self.column_menu = QMenu(self.column_button)
        self.column_menu.setTitle("表示する列（実験一覧）")
        self.column_button.setMenu(self.column_menu)
        self.column_actions = []
        column_labels = (
            "選択",
            "ID",
            "実験群",
            "モデル",
            "データセット",
            "データ拡張",
            "状態",
            "進捗",
            "OOF AP",
            "途中保存モデル",
        )
        for column, label in enumerate(column_labels):
            action = self.column_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(
                lambda visible, index=column: self.table.setColumnHidden(index, not visible)
            )
            self.column_actions.append(action)
        filter_row.addWidget(self.column_button)
        self.experiment_filter_menu = QMenu("実験の絞り込み", self)
        self.experiment_filter_actions = []
        self.experiment_filter_submenus = []
        self._experiment_filter_specs = (
            ("実験群", self.study_filter),
            ("モデル", self.model_filter),
            ("状態", self.state_filter),
        )
        for label, combo in (*self._experiment_filter_specs,):
            submenu = self.experiment_filter_menu.addMenu(label)
            self.experiment_filter_submenus.append(submenu)
            combo.currentIndexChanged.connect(
                lambda selected, target=combo: self._update_filter_menu_checks(target, selected)
            )
        self._refresh_experiment_filter_menu()
        setup_table(self.table, stretch_column=9)
        self.table.setItemDelegateForColumn(
            6,
            TagDelegate({label: colors for label, colors in STATUS_MARKS.items()}, self.table),
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.table)
        self.details = QTabWidget()
        overview_page = QWidget()
        overview_layout = QVBoxLayout(overview_page)
        self.overview_placeholder = QLabel("実験を選ぶと詳細が表示されます")
        self.overview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_style(self.overview_placeholder, role="note")
        overview_layout.addWidget(self.overview_placeholder, 1)
        self.overview_table = QTableWidget(0, 2)
        self.overview_table.setHorizontalHeaderLabels(["設定項目", "値"])
        self.overview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.overview_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        setup_table(self.overview_table, stretch_column=1)
        overview_layout.addWidget(self.overview_table, 1)
        self.yaml_button = QPushButton("設定 YAML を表示")
        self.yaml_button.clicked.connect(self.show_config_yaml)
        overview_layout.addWidget(self.yaml_button, 0, Qt.AlignmentFlag.AlignRight)
        self.chart = QWidget()
        chart_layout = QVBoxLayout(self.chart)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addWidget(QLabel("交差検証（OOF）AP"))
        self.chart_map = LineChart()
        chart_layout.addWidget(self.chart_map, 1)
        chart_layout.addWidget(QLabel("学習 loss"))
        self.chart_loss = LineChart()
        chart_layout.addWidget(self.chart_loss, 1)
        self.cv_table = QTableWidget(0, 4)
        self.cv_table.setHorizontalHeaderLabels(
            ["フォールド", "学習件数", "検証件数", "選択エポック AP"]
        )
        setup_table(self.cv_table, stretch_column=0)
        self.cv_page = QWidget()
        cv_layout = QVBoxLayout(self.cv_page)
        self.selected_epoch_label = QLabel("選択エポック: —")
        self.selected_epoch_label.setFont(numeric_font())
        cv_layout.addWidget(self.selected_epoch_label)
        cv_layout.addWidget(QLabel("フォールド別評価"))
        cv_layout.addWidget(self.cv_table)
        self.oof_table = QTableWidget(2, 5)
        self.oof_table.setHorizontalHeaderLabels(["OOF評価", "全体", "分類A", "分類B", "分類C"])
        setup_table(self.oof_table, stretch_column=0)
        cv_layout.addWidget(QLabel("OOF評価"))
        cv_layout.addWidget(self.oof_table)
        self.checkpoint_table = QTableWidget(0, 5)
        self.checkpoint_table.setHorizontalHeaderLabels(
            ["フォールド（1〜K / 最終）", "ファイル名", "エポック", "AP", "保存日時"]
        )
        setup_table(self.checkpoint_table, stretch_column=0)
        self.run_table = QTableWidget(0, 7)
        self.run_table.setHorizontalHeaderLabels(
            ["試行", "開始", "終了", "結果", "OOF AP", "選択エポック", "実行環境"]
        )
        self.run_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.run_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        setup_table(self.run_table, stretch_column=4)
        self.run_page = QWidget()
        run_layout = QVBoxLayout(self.run_page)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.addWidget(self.run_table, 1)
        self.retry_button = QPushButton(RETRY_LABEL)
        self.retry_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        run_layout.addWidget(self.retry_button, 0, Qt.AlignmentFlag.AlignRight)
        self.used_data = QTextEdit()
        self.used_data.setReadOnly(True)
        self.details.addTab(overview_page, "概要")
        self.details.addTab(self.chart, "学習曲線")
        self.details.addTab(self.cv_page, "交差検証")
        self.details.addTab(self.checkpoint_table, "途中保存モデル")
        self.details.addTab(self.run_page, "実行試行")
        self.details.addTab(self.used_data, "実使用データ")
        splitter.addWidget(self.details)
        splitter.setSizes([400, 330])
        self.button_map: dict[str, QPushButton] = {}
        for key, label in (
            ("compare", "選択した実験を比較"),
            ("copy", "設定を複製して新規実験"),
            ("send", "モデル比較へ送る…"),
        ):
            button = QPushButton(label)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.button_map[key] = button
        mark_primary(self.button_map["send"])
        self.more_button = QPushButton("その他 ▾")
        self.more_menu = QMenu(self.more_button)
        self.action_map = {}
        for key, label, callback in (
            ("compare", "選択した実験を比較", self.compare_selected),
            ("copy", "設定を複製して新規実験", self.copy_selected),
            ("send", "モデル比較へ送る…", self.send_selected),
        ):
            self.action_map[key] = QAction(label, self)
            self.action_map[key].triggered.connect(callback)
        for key, label, callback in (
            ("stop", "■ 今すぐ停止", self.stop_selected),
            ("retry", RETRY_LABEL, self.retry_selected),
            ("edit", "下書きを編集", self.edit_selected),
            ("queue_copy", "複製してキューに追加", self.copy_to_queue),
            ("result", "結果を開く", self.open_result),
        ):
            action = self.more_menu.addAction(label)
            action.triggered.connect(callback)
            self.action_map[key] = action
        self.action_map["delete"] = QAction("実験を削除…", self)
        self.action_map["delete"].triggered.connect(self.delete_selected)
        self.action_map["cleanup"] = QAction(CLEANUP_LABEL, self)
        self.action_map["cleanup"].triggered.connect(self.cleanup_selected)
        bind_button_action(self.retry_button, self.action_map["retry"])
        self.more_button.setMenu(self.more_menu)
        self.more_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        for key in ("compare", "copy"):
            self.button_map[key].hide()
            bind_button_action(self.button_map[key], self.action_map[key])
        self.more_button.hide()
        bind_button_action(self.button_map["send"], self.action_map["send"])
        filter_row.addWidget(self.button_map["send"])
        self.content_layout.addLayout(filter_row)
        self.content_layout.addWidget(splitter, 1)
        self.study_filter.currentTextChanged.connect(self.refresh)
        self.model_filter.currentTextChanged.connect(self.refresh)
        self.state_filter.currentTextChanged.connect(self.refresh)
        self.table.itemChanged.connect(self._selection_changed)
        self.table.itemSelectionChanged.connect(self._current_changed)
        self.context_menu = QMenu(self)
        self.context_menu.setToolTipsVisible(True)
        self.context_menu.aboutToShow.connect(self._build_context_menu)
        self._build_context_menu()

        def select_experiment_for_context(row_index):
            item = self.table.item(row_index, 0)
            if item and item.checkState() != Qt.CheckState.Checked:
                for current_row in range(self.table.rowCount()):
                    current_item = self.table.item(current_row, 0)
                    if current_item:
                        current_item.setCheckState(
                            Qt.CheckState.Checked
                            if current_row == row_index
                            else Qt.CheckState.Unchecked
                        )

        add_row_context_menu(self.table, self.context_menu, select_experiment_for_context)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.ctx.training_runner.progressed.connect(lambda _experiment_id: self.refresh())
        self.refresh()

    def _build_context_menu(self) -> None:
        """選んだ実験の状態に合わせて、よく使う操作を先頭に並べる。"""
        self._update_buttons()
        current = self._current_experiment()
        status = current.status if current is not None else ""
        menu = self.context_menu
        # QMenu.clear() は共有している QAction まで破棄するため、1 件ずつ外す
        for action in menu.actions():
            menu.removeAction(action)
            if action.isSeparator():
                action.deleteLater()
        if status in {"failed", "stopped"}:
            menu.addAction(self.action_map["retry"])
            menu.addSeparator()
        elif status == "running":
            menu.addAction(self.action_map["stop"])
            menu.addSeparator()
        for key in ("send", "compare", "copy", "queue_copy", "result", "edit"):
            menu.addAction(self.action_map[key])
        menu.addSeparator()
        menu.addAction(self.action_map["cleanup"])
        menu.addAction(self.action_map["delete"])

    def menu_actions(self):
        if not hasattr(self, "yaml_menu_action"):
            self.yaml_menu_action = QAction("設定 YAML を表示…", self)
            self.yaml_menu_action.triggered.connect(self.show_config_yaml)
            self._update_buttons()
        return {
            "file": [None, self.yaml_menu_action],
            "edit": [self.action_map["edit"]],
            "training": [
                self.action_map["retry"],
                None,
                self.action_map["copy"],
                self.action_map["queue_copy"],
                None,
                self.action_map["compare"],
                self.action_map["result"],
                self.action_map["send"],
                None,
                self.action_map["cleanup"],
                self.action_map["delete"],
            ],
            "view": [self.column_menu.menuAction(), self.experiment_filter_menu.menuAction()],
        }

    def _refresh_experiment_filter_menu(self) -> None:
        self.experiment_filter_actions = []
        for submenu, (_label, combo) in zip(
            self.experiment_filter_submenus, self._experiment_filter_specs, strict=True
        ):
            submenu.clear()
            actions = []
            for index in range(combo.count()):
                action = submenu.addAction(combo.itemText(index))
                action.setCheckable(True)
                action.triggered.connect(
                    lambda _checked=False, target=combo, item=index: target.setCurrentIndex(item)
                )
                actions.append(action)
            self.experiment_filter_actions.append(actions)
            self._update_filter_menu_checks(combo, combo.currentIndex())

    def _update_filter_menu_checks(self, combo: QComboBox, selected: int) -> None:
        filter_index = next(
            index
            for index, (_label, target) in enumerate(self._experiment_filter_specs)
            if target is combo
        )
        for index, action in enumerate(self.experiment_filter_actions[filter_index]):
            action.setChecked(index == selected)

    def on_enter(self, params: dict[str, object]) -> None:
        """遷移パラメータの実験を選択して再読込する。"""
        self.refresh()
        selected = params.get("select")
        if selected:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 1)
                if item and item.text() == selected:
                    self.table.setCurrentCell(row, 1)
                    self.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
                    break

    def refresh(self, *_args: object) -> None:
        """一覧を再読込してフィルターと実行状況を反映する。"""
        selected_ids = {item.experiment_id for item in self._checked_experiments()}
        current = self._current_experiment_id()
        scroll_value = self.table.verticalScrollBar().value()
        experiments = self.ctx.backend.list_experiments()
        studies = sorted({item.study_id for item in experiments})
        old_study = self.study_filter.currentText()
        self.study_filter.blockSignals(True)
        self.study_filter.clear()
        self.study_filter.addItems(["すべて", *studies])
        if old_study in ["すべて", *studies]:
            self.study_filter.setCurrentText(old_study)
        self.study_filter.blockSignals(False)
        self._refresh_experiment_filter_menu()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for experiment in experiments:
            if (
                self.study_filter.currentText() != "すべて"
                and experiment.study_id != self.study_filter.currentText()
            ):
                continue
            if (
                self.model_filter.currentText() != "すべて"
                and model_type_label(experiment.model_type) != self.model_filter.currentText()
            ):
                continue
            if (
                self.state_filter.currentText() != "すべて"
                and experiment_status_label(experiment.status) != self.state_filter.currentText()
            ):
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            select = QTableWidgetItem()
            select.setFlags(
                select.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            )
            select.setCheckState(
                Qt.CheckState.Checked
                if experiment.experiment_id in selected_ids
                else Qt.CheckState.Unchecked
            )
            self.table.setItem(row, 0, select)
            config = copy.deepcopy(experiment.config.values)
            # 保存日時が同じなら後に追加したもの（最終モデル）を最新とする
            latest = max(reversed(experiment.checkpoints), key=lambda cp: cp.saved_at, default=None)
            selected_metric = next(
                (
                    point
                    for point in experiment.oof_history
                    if point.epoch == experiment.selected_epoch
                ),
                None,
            )
            cv_folds = config.get("data", {}).get("cv", {}).get("n_folds", 5)
            if experiment.status != "running":
                progress = "—"
            elif experiment.phase == "final_training":
                progress = (
                    "最終学習・エポック "
                    f"{experiment.current_epoch}/"
                    f"{experiment.selected_epoch or experiment.total_epochs}"
                )
            else:
                progress = (
                    f"分割 {max(experiment.fold_histories, default=1)}/{cv_folds}・エポック "
                    f"{experiment.current_epoch}/{experiment.total_epochs}"
                )
            values = [
                experiment.experiment_id,
                experiment.study_id,
                model_type_label(experiment.model_type),
                str(config.get("data", {}).get("dataset_version", "—")),
                str(config.get("augmentation", {}).get("profile", "—")),
                experiment_status_label(experiment.status),
                progress,
                format_score(selected_metric.map) if selected_metric else "—",
                latest.name if latest else "—",
            ]
            for col, value in enumerate(values, start=1):
                cell = QTableWidgetItem(value)
                if col in (1, 4, 7, 8):
                    cell.setFont(numeric_font())
                    cell.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if col == 8 and selected_metric:
                    font = cell.font()
                    font.setBold(True)
                    cell.setFont(font)
                cell.setData(Qt.ItemDataRole.UserRole, experiment.experiment_id)
                self.table.setItem(row, col, cell)
        self.table.blockSignals(False)
        if current:
            for row in range(self.table.rowCount()):
                if self.table.item(row, 1).text() == current:
                    self.table.setCurrentCell(row, 1)
                    break
        self.table.verticalScrollBar().setValue(scroll_value)
        self._current_changed()
        fit_table_columns(self.table)
        self._update_buttons()

    def _checked_experiments(self) -> list[Experiment]:
        ids = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                ids.append(self.table.item(row, 1).text())
        # 削除直後は表に残った行が実験を指さないことがあるため、存在するものだけ返す
        known = {item.experiment_id: item for item in self.ctx.backend.list_experiments()}
        return [known[experiment_id] for experiment_id in ids if experiment_id in known]

    def _current_experiment_id(self) -> str | None:
        row = self.table.currentRow()
        return self.table.item(row, 1).text() if row >= 0 and self.table.item(row, 1) else None

    def _current_experiment(self) -> Experiment | None:
        experiment_id = self._current_experiment_id()
        if not experiment_id:
            return None
        try:
            return self.ctx.backend.get_experiment(experiment_id)
        except KeyError:
            return None

    def _selection_changed(self, _item: QTableWidgetItem | None = None) -> None:
        self._update_buttons()

    def _current_changed(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            self.overview_placeholder.show()
            self.overview_table.hide()
            self.overview_table.setRowCount(0)
            self.yaml_button.hide()
            self.chart_map.set_series([])
            self.chart_loss.set_series([])
            self.checkpoint_table.setRowCount(0)
            self.run_table.setRowCount(0)
            self.used_data.clear()
            fit_table_columns(self.table)
            self._update_buttons()
            return
        config = experiment.config.values
        flattened = flatten_config(config)
        overview_rows = [
            (config_key_label(key), self._display_value(key, value))
            for key, value in flattened.items()
            if key not in {"data.used_item_ids", "experiment.id"}
        ]
        overview_rows.extend(
            [
                ("モデル", model_type_label(experiment.model_type)),
                ("実使用データ数", f"{len(experiment.used_item_ids)} 件"),
            ]
        )
        self.overview_placeholder.hide()
        self.overview_table.show()
        self.yaml_button.show()
        self.overview_table.setRowCount(len(overview_rows))
        for row, (label, value) in enumerate(overview_rows):
            self.overview_table.setItem(row, 0, QTableWidgetItem(label))
            value_item = QTableWidgetItem(value)
            value_item.setFont(numeric_font())
            self.overview_table.setItem(row, 1, value_item)
        map_series = [
            (
                "OOF AP",
                QColor(Color.GRAPHITE),
                [float(p.epoch) for p in experiment.oof_history if p.map is not None],
                [float(p.map) for p in experiment.oof_history if p.map is not None],
            )
        ]
        for fold, points in sorted(experiment.fold_histories.items()):
            map_series.append(
                (
                    f"分割 {fold}",
                    QColor(Color.SLATE),
                    [float(p.epoch) for p in points if p.map is not None],
                    [float(p.map) for p in points if p.map is not None],
                )
            )
        self.chart_map.set_series(map_series)
        loss_series = [
            (
                "交差検証 loss",
                QColor(Color.SLATE),
                [float(p.epoch) for p in experiment.oof_history],
                [float(p.loss) for p in experiment.oof_history],
            )
        ]
        if experiment.final_history:
            loss_series.append(
                (
                    "最終学習 loss",
                    QColor(Color.TRAIN),
                    [float(p.epoch) for p in experiment.final_history],
                    [float(p.loss) for p in experiment.final_history],
                )
            )
        self.chart_loss.set_series(loss_series)
        self.chart_map.set_best(None, None)
        self.chart_loss.set_best(None, None)
        best = next(
            (p for p in experiment.oof_history if p.epoch == experiment.selected_epoch), None
        )
        if best and best.map is not None:
            self.chart_map.set_best(float(best.epoch), float(best.map))
        item_folds = experiment.fold_assignments
        n_folds = int(config.get("data", {}).get("cv", {}).get("n_folds", 5))
        self.cv_table.setRowCount(n_folds)
        for fold in range(1, n_folds + 1):
            validation_count = sum(value == fold for value in item_folds.values())
            train_count = len(item_folds) - validation_count
            fold_metric = next(
                (
                    p.map
                    for p in experiment.fold_histories.get(fold, [])
                    if p.epoch == experiment.selected_epoch
                ),
                None,
            )
            for col, value in enumerate(
                (str(fold), str(train_count), str(validation_count), format_score(fold_metric))
            ):
                item = QTableWidgetItem(value)
                item.setFont(numeric_font())
                if col:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.cv_table.setItem(fold - 1, col, item)
        self.selected_epoch_label.setText(
            f"選択エポック: {experiment.selected_epoch if experiment.selected_epoch else '—'}"
        )
        oof_eval = experiment.oof_evaluation
        classifications = ["分類A", "分類B", "分類C"]
        if oof_eval and "未分類" in oof_eval.per_class:
            classifications.append("未分類")
        self.oof_table.setColumnCount(2 + len(classifications))
        self.oof_table.setHorizontalHeaderLabels(["OOF評価", "全体", *classifications])
        self.oof_table.setItem(0, 0, QTableWidgetItem("OOF AP（Cellpose 方式、IoU 0.50–0.95）"))
        self.oof_table.setItem(1, 0, QTableWidgetItem("対象件数"))
        self.oof_table.setItem(
            0, 1, QTableWidgetItem(format_score(oof_eval.overall_map) if oof_eval else "—")
        )
        self.oof_table.setItem(
            1,
            1,
            QTableWidgetItem(
                str(sum(count for _score, count in oof_eval.per_class.values()))
                if oof_eval
                else "—"
            ),
        )
        for column, classification in enumerate(classifications, start=2):
            score, count = (
                oof_eval.per_class.get(classification, (None, 0)) if oof_eval else (None, 0)
            )
            self.oof_table.setItem(0, column, QTableWidgetItem(format_score(score)))
            self.oof_table.setItem(1, column, QTableWidgetItem(str(count) if oof_eval else "—"))
        for row in range(self.oof_table.rowCount()):
            for column in range(1, self.oof_table.columnCount()):
                item = self.oof_table.item(row, column)
                item.setFont(numeric_font())
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.checkpoint_table.setRowCount(len(experiment.checkpoints))
        checkpoints = sorted(
            experiment.checkpoints,
            key=lambda item: (
                item.fold is None,
                item.fold or 99,
                item.epoch,
            ),
        )
        pruned = self._pruned_paths(experiment)
        for row, checkpoint in enumerate(checkpoints):
            fold_text = "最終" if checkpoint.name == "final.pt" else str(checkpoint.fold or "—")
            is_pruned = self._checkpoint_path(checkpoint) in pruned
            values = [
                fold_text,
                f"{checkpoint.name}（{PRUNED_TEXT}）" if is_pruned else checkpoint.name,
                str(checkpoint.epoch),
                format_score(checkpoint.map, 4),
                format_datetime(checkpoint.saved_at),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 2, 3, 4):
                    item.setFont(numeric_font())
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if is_pruned:
                    item.setForeground(QColor(Color.IDLE))
                    item.setToolTip("成果物の整理で削除したファイルです。記録は残っています。")
                self.checkpoint_table.setItem(row, col, item)
        self.run_table.setRowCount(len(experiment.runs))
        for row, run in enumerate(experiment.runs):
            values = [
                str(run.attempt),
                format_datetime(run.started_at),
                format_datetime(run.finished_at) if run.finished_at else "実行中",
                experiment_status_label(run.result),
                format_score(run.oof_evaluation.overall_map) if run.oof_evaluation else "—",
                str(run.selected_epoch) if run.selected_epoch is not None else "—",
                self._format_environment(run.environment),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 1, 2, 4, 5):
                    item.setFont(numeric_font())
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.run_table.setItem(row, col, item)
        self.used_data.setPlainText(
            f"実使用データ数: {len(experiment.used_item_ids)} 件\n"
            + "\n".join(
                f"{item_id}\t分割 {experiment.fold_assignments.get(item_id, '—')}"
                for item_id in experiment.used_item_ids
            )
        )
        for table in (
            self.table,
            self.overview_table,
            self.cv_table,
            self.oof_table,
            self.checkpoint_table,
            self.run_table,
        ):
            fit_table_columns(table)
        self._update_buttons()

    def _update_buttons(self) -> None:
        selected = self._checked_experiments()
        current = self._current_experiment()
        self.set_menu_action_enabled(self.action_map["compare"], len(selected) >= 2)
        self.set_menu_action_enabled(self.action_map["copy"], current is not None)
        send_reason = self._send_block_reason(current)
        self.set_menu_action_enabled(self.action_map["send"], not send_reason)
        self.action_map["compare"].setToolTip(
            "実験を 2 つ以上選ぶと使えます" if len(selected) < 2 else ""
        )
        self.action_map["copy"].setToolTip(
            "複製する実験を選ぶと使えます" if current is None else ""
        )
        self.action_map["send"].setToolTip(send_reason)
        self.set_menu_action_enabled(
            self.action_map["result"], current is not None and current.status == "completed"
        )
        self.set_menu_action_enabled(
            self.action_map["stop"], current is not None and current.status == "running"
        )
        self.set_menu_action_enabled(
            self.action_map["retry"],
            current is not None and current.status in {"failed", "stopped"},
        )
        self.set_menu_action_enabled(
            self.action_map["edit"], current is not None and current.status == "draft"
        )
        self.action_map["result"].setToolTip(
            "完了した実験を 1 つ選ぶと結果を開けます"
            if current is None or current.status != "completed"
            else ""
        )
        stop_tip = (
            "実行中の実験を 1 つ選ぶと今すぐ停止できます。"
            if current is None or current.status != "running"
            else STOP_TIP
        )
        self.action_map["stop"].setToolTip(stop_tip)
        self.action_map["retry"].setToolTip(
            "失敗または中断した実験を 1 つ選ぶと使えます"
            if current is None or current.status not in {"failed", "stopped"}
            else RETRY_TIP
        )
        self._update_delete_action(current)
        targets = bool(selected) or current is not None
        self.set_menu_action_enabled(self.action_map["cleanup"], targets)
        self.action_map["cleanup"].setToolTip(
            "記録は残したまま、選んだ実験の途中保存モデルなどの大きなファイルを削除します。"
            if targets
            else "整理する実験を選ぶかチェックしてください"
        )
        self.action_map["edit"].setToolTip(
            "下書きの実験を 1 つ選ぶと編集できます"
            if current is None or current.status != "draft"
            else ""
        )
        self.action_map["compare"].setToolTip(
            "比較する実験を 2 つ以上選んでください" if len(selected) < 2 else ""
        )
        self.action_map["copy"].setToolTip(
            "複製する実験を 1 つ選んでください" if current is None else ""
        )
        self.action_map["send"].setToolTip(send_reason)
        if hasattr(self, "yaml_menu_action"):
            self.set_menu_action_enabled(self.yaml_menu_action, current is not None)
            self.yaml_menu_action.setToolTip(
                "設定 YAML を表示する実験を選んでください" if current is None else ""
            )
        self.set_menu_action_enabled(self.action_map["queue_copy"], bool(selected))
        self.action_map["queue_copy"].setToolTip(
            "複製する実験をチェックしてください" if not selected else ""
        )

    def _send_block_reason(self, current: Experiment | None) -> str:
        """「モデル比較へ送る」を押せない理由。押せるなら空文字。"""
        if current is None or current.status != "completed" or not current.checkpoints:
            return "完了した実験と途中保存モデルを選ぶと使えます"
        if not usable_attempts(self.ctx.backend, current):
            return FINAL_PRUNED_REASON
        return ""

    def _pruned_paths(self, experiment: Experiment) -> set[str]:
        """最新の試行で、成果物の整理により削除済みのファイル（run_dir 相対）を返す。"""
        if not experiment.runs:
            return set()
        try:
            return set(
                self.ctx.backend.pruned_paths(experiment.experiment_id, experiment.runs[-1].attempt)
            )
        except (KeyError, ValueError, OSError):
            return set()

    @staticmethod
    def _checkpoint_path(checkpoint) -> str:
        """途中保存モデルの run_dir 相対パス（pruned.json の path と同じ形）。"""
        if checkpoint.fold is None:
            return f"checkpoints/{checkpoint.name}"
        return f"checkpoints/fold_{checkpoint.fold}/{checkpoint.name}"

    def _cleanup_targets(self) -> list[str]:
        """整理の対象: チェックした実験。なければ選択中の実験。"""
        checked = [item.experiment_id for item in self._checked_experiments()]
        if checked:
            return checked
        current = self._current_experiment_id()
        return [current] if current else []

    def cleanup_selected(self) -> ArtifactCleanupDialog | None:
        """選んだ実験の成果物を整理するダイアログを開く。"""
        targets = self._cleanup_targets()
        if not targets:
            return None
        try:
            dialog = ArtifactCleanupDialog(self.ctx.backend, targets, self)
        except (KeyError, ValueError, OSError):
            QMessageBox.warning(
                self,
                "成果物を整理できません",
                "対象の実験を読み込めませんでした。一覧を更新してから、もう一度選んでください。",
            )
            self.refresh()
            return None
        self._cleanup_dialog = dialog
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_value is not None:
            self.ctx.status.show_message(cleanup_status_message(dialog.result_value))
            self.refresh()
            self._current_changed()
        return dialog

    def _update_delete_action(self, current: Experiment | None) -> None:
        action = self.action_map["delete"]
        if current is None:
            enabled, tip = False, "削除する実験を 1 つ選んでください"
        else:
            try:
                info = self.ctx.backend.experiment_deletion_info(
                    current.experiment_id, measure_size=False
                )
                enabled, tip = info.allowed, info.reason
            except (KeyError, ValueError) as error:
                enabled, tip = False, str(error)
            if enabled:
                tip = "実験の設定・全試行・途中保存モデル・ログを削除します。取り消せません。"
        self.set_menu_action_enabled(action, enabled)
        action.setToolTip(tip)

    @staticmethod
    def _display_value(key: str, value: object) -> str:
        """設定値を日本語ラベルと読みやすい文字列へ変換する。"""
        if value is None:
            return "未設定"
        if isinstance(value, bool):
            return "有効" if value else "無効"
        if key == "model.type":
            return model_type_label(str(value))
        if key == "data.classification":
            return classification_label(str(value))
        if key == "data.quality_filter":
            return quality_filter_label(str(value))
        if key == "checkpoint.best_metric" and value == "oof_instance_map":
            return "OOF AP（Cellpose 方式）・最大"
        if key == "model.pretrained_weights":
            return {"coco": "COCO", "imagenet": "ImageNet"}.get(str(value), str(value))
        if key == "model.backbone":
            return {"resnet50_fpn_v2": "ResNet-50 FPN v2", "resnet101_fpn": "ResNet-101 FPN"}.get(
                str(value), str(value)
            )
        if key == "data.used_item_ids":
            return f"{len(value)} 件" if isinstance(value, list) else str(value)
        if isinstance(value, list):
            return "、".join(str(item) for item in value)
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    @staticmethod
    def _format_environment(environment: dict[str, object]) -> str:
        """実行環境の辞書を項目名付きで表示する。"""
        if not environment:
            return "モック環境"
        labels = {"gpu": "GPU", "num_workers": "ワーカー数"}
        return "、".join(f"{labels.get(key, key)}: {value}" for key, value in environment.items())

    def compare_selected(self) -> ExperimentCompareDialog | None:
        selected = self._checked_experiments()
        if len(selected) < 2:
            return None
        dialog = ExperimentCompareDialog(selected, self)
        dialog.exec()
        return dialog

    def open_result(self) -> None:
        if self._current_experiment():
            self.details.setCurrentIndex(1)

    def show_config_yaml(self) -> QDialog | None:
        """現在の実験設定 YAML を読み取り専用ダイアログで表示する。"""
        experiment = self._current_experiment()
        if experiment is None:
            return None
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{experiment.experiment_id} の設定 YAML")
        dialog.resize(720, 600)
        dialog.setMinimumSize(600, 450)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(experiment.config.to_yaml())
        layout.addWidget(text)
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, 0, Qt.AlignmentFlag.AlignRight)
        dialog.exec()
        return dialog

    def copy_selected(self) -> None:
        experiment = self._current_experiment()
        if experiment:
            self.ctx.navigator.navigate(PageId.TRAINING, copy_from=experiment.experiment_id)

    def copy_to_queue(self) -> None:
        """選択した実験設定を新しい ID でキューへ複製する。"""
        selected = self._checked_experiments()
        added = []
        for experiment in selected:
            config = copy.deepcopy(experiment.config.values)
            config["experiment"]["id"] = self.ctx.backend.next_experiment_id()
            added.append(self.ctx.backend.add_training_queue_item(config))
        if added:
            self.ctx.queue_controller.sync_training_identifier()
            self.ctx.status.show_message(f"{len(added)} 件を学習キューに追加しました")
            self.ctx.navigator.navigate(PageId.TRAINING_QUEUE)

    def edit_selected(self) -> None:
        experiment = self._current_experiment()
        if experiment and experiment.status == "draft":
            self.ctx.navigator.navigate(PageId.TRAINING, edit=experiment.experiment_id)

    def stop_selected(self, confirm: bool = True) -> None:
        experiment = self._current_experiment()
        if not experiment or experiment.status != "running":
            return
        if (
            confirm
            and QMessageBox.question(
                self,
                "今すぐ停止",
                "今の学習をすぐに止め、キューも止めますか？\n"
                "今の学習は「中断」になり、途中までの結果だけが残ります。",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        controller = self.ctx.queue_controller
        if controller.executing or controller.waiting_for_training:
            controller.stop_now()
        elif self.ctx.training_runner.is_busy:
            self.ctx.training_runner.request_stop("user_stop")
        self.refresh()

    def retry_selected(self) -> None:
        experiment = self._current_experiment()
        if not experiment or experiment.status not in {"failed", "stopped"}:
            return
        controller = self.ctx.queue_controller
        busy = self.ctx.training_runner.is_busy or controller.executing
        if busy:
            if self.ctx.training_runner.is_busy:
                active_id = self.ctx.training_runner.experiment_id
                prompt = f"学習を実行中です（{active_id}）。この学習をキューの末尾に追加しますか？"
            else:
                prompt = "キューを実行中です。やり直しをキューの末尾に追加しますか？"
            if QMessageBox.question(self, "学習中", prompt) != QMessageBox.StandardButton.Yes:
                return
        # 学習の開始はすべてキューを通す（空いていればそのままキューを実行する）
        try:
            queued = self.ctx.backend.add_training_retry_reservation(experiment.experiment_id)
        except ValueError as error:
            QMessageBox.warning(self, "やり直せません", str(error))
            return
        controller.sync_training_identifier()
        if not controller.executing:
            controller.start()
        label = f"{queued.experiment_id}（再試行 {queued.queue_retry_attempt}）"
        self.ctx.status.show_message(
            f"{label}をキューに予約しました"
            if busy
            else f"{label}をキューに追加して実行を始めました"
        )
        self.refresh()

    @staticmethod
    def delete_prompt(info) -> str:
        """削除確認ダイアログの本文を返す。"""
        lines = [f"{info.experiment_id} を削除しますか？", ""]
        if info.attempts:
            lines.append(f"設定と実行試行 {info.attempts} 回分の記録を削除します。")
            lines.append("（途中保存モデル・OOF 予測・ログを含む）")
        else:
            lines.append("実験の設定を削除します。")
        if info.size_bytes is not None:
            lines.append(f"{format_bytes(info.size_bytes)} を削除します。")
        lines.append("学習キューに残っているこの実験の行も削除します。")
        lines.extend(["", "この操作は取り消せません。"])
        return "\n".join(lines)

    def delete_selected(self) -> bool:
        """選んだ実験を、確認のうえ記録ごと削除する。"""
        experiment = self._current_experiment()
        if experiment is None:
            return False
        try:
            info = self.ctx.backend.experiment_deletion_info(experiment.experiment_id)
        except (KeyError, ValueError) as error:
            QMessageBox.warning(self, "実験を削除できません", str(error))
            return False
        if not info.allowed:
            QMessageBox.warning(self, "実験を削除できません", info.reason)
            return False
        if (
            QMessageBox.question(
                self,
                "実験を削除",
                self.delete_prompt(info),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return False
        try:
            self.ctx.backend.delete_experiment(experiment.experiment_id)
        except (KeyError, ValueError, OSError) as error:
            QMessageBox.warning(self, "実験を削除できません", str(error))
            self.refresh()
            return False
        self.ctx.status.show_message(f"{experiment.experiment_id} を削除しました")
        self.ctx.queue_controller.changed.emit()
        self.ctx.queue_controller.sync_training_identifier()
        self.refresh()
        return True

    def send_selected(self) -> SendToCandidatesDialog | None:
        experiment = self._current_experiment()
        if not experiment or self._send_block_reason(experiment):
            return None
        dialog = SendToCandidatesDialog(
            experiment, self, usable_attempts(self.ctx.backend, experiment)
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.ctx.navigator.navigate(PageId.CANDIDATES, **dialog.transition_params())
        return dialog
