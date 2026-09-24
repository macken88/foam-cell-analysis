"""実験検索、実行状態、学習結果の一覧画面。"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
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
from ...jobs import FakeJob
from ...labels import (
    classification_label,
    config_key_label,
    experiment_status_label,
    format_datetime,
    format_score,
    model_type_label,
    quality_filter_label,
)
from ...navigation import PageId
from ...widgets.chart import LineChart
from ...widgets.page_base import BasePage
from ...widgets.table import mark_primary, setup_table
from .dialogs import ExperimentCompareDialog, SendToCandidatesDialog, flatten_config


class ExperimentListPage(BasePage):
    """実験一覧と詳細を表示し、実験操作へ誘導する。"""

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(ctx, "実験一覧", "実験の状態、設定、学習結果を確認します。", parent)
        self.study_filter = QComboBox()
        self.study_filter.addItem("すべて")
        self.model_filter = QComboBox()
        self.model_filter.addItems(["すべて", "Mask R-CNN", "Cellpose"])
        self.state_filter = QComboBox()
        self.state_filter.addItems(["すべて", "下書き", "実行中", "完了", "失敗", "中断"])
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("実験群"))
        filter_row.addWidget(self.study_filter)
        filter_row.addWidget(QLabel("モデル"))
        filter_row.addWidget(self.model_filter)
        filter_row.addWidget(QLabel("状態"))
        filter_row.addWidget(self.state_filter)
        filter_row.addStretch(1)
        self.table = QTableWidget(0, 9)
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
                "最良 mAP",
                "途中保存モデル",
            ]
        )
        setup_table(self.table, stretch_column=8)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.table)
        self.details = QTabWidget()
        self.overview = QTextEdit()
        self.overview.setReadOnly(True)
        overview_page = QWidget()
        overview_layout = QVBoxLayout(overview_page)
        self.yaml_button = QPushButton("設定 YAML を表示")
        self.yaml_button.clicked.connect(self.show_config_yaml)
        overview_layout.addWidget(self.overview, 1)
        overview_layout.addWidget(self.yaml_button, 0, Qt.AlignmentFlag.AlignRight)
        self.chart = LineChart()
        self.checkpoint_table = QTableWidget(0, 4)
        self.checkpoint_table.setHorizontalHeaderLabels(
            ["ファイル名", "エポック", "mAP", "保存日時"]
        )
        setup_table(self.checkpoint_table, stretch_column=0)
        self.run_table = QTableWidget(0, 5)
        self.run_table.setHorizontalHeaderLabels(["試行", "開始", "終了", "結果", "実行環境"])
        setup_table(self.run_table, stretch_column=4)
        self.used_data = QTextEdit()
        self.used_data.setReadOnly(True)
        self.details.addTab(overview_page, "概要")
        self.details.addTab(self.chart, "学習曲線")
        self.details.addTab(self.checkpoint_table, "途中保存モデル")
        self.details.addTab(self.run_table, "実行試行")
        self.details.addTab(self.used_data, "実使用データ")
        splitter.addWidget(self.details)
        splitter.setSizes([400, 330])
        self.button_map: dict[str, QPushButton] = {}
        buttons = QHBoxLayout()
        for key, label in (
            ("compare", "選択した実験を比較"),
            ("result", "結果を開く"),
            ("copy", "設定を複製して新規実験"),
            ("stop", "学習を中断"),
            ("retry", "再実行"),
            ("send", "モデル比較へ送る…"),
            ("edit", "下書きを編集"),
        ):
            button = QPushButton(label)
            self.button_map[key] = button
            buttons.addWidget(button)
        mark_primary(self.button_map["send"])
        mark_primary(self.button_map["compare"])
        self.content_layout.addLayout(filter_row)
        self.content_layout.addWidget(splitter, 1)
        self.content_layout.addLayout(buttons)
        self.study_filter.currentTextChanged.connect(self.refresh)
        self.model_filter.currentTextChanged.connect(self.refresh)
        self.state_filter.currentTextChanged.connect(self.refresh)
        self.table.itemChanged.connect(self._selection_changed)
        self.table.itemSelectionChanged.connect(self._current_changed)
        self.button_map["compare"].clicked.connect(self.compare_selected)
        self.button_map["result"].clicked.connect(self.open_result)
        self.button_map["copy"].clicked.connect(self.copy_selected)
        self.button_map["stop"].clicked.connect(self.stop_selected)
        self.button_map["retry"].clicked.connect(self.retry_selected)
        self.button_map["send"].clicked.connect(self.send_selected)
        self.button_map["edit"].clicked.connect(self.edit_selected)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

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
        experiments = self.ctx.backend.list_experiments()
        studies = sorted({item.study_id for item in experiments})
        old_study = self.study_filter.currentText()
        self.study_filter.blockSignals(True)
        self.study_filter.clear()
        self.study_filter.addItems(["すべて", *studies])
        if old_study in ["すべて", *studies]:
            self.study_filter.setCurrentText(old_study)
        self.study_filter.blockSignals(False)
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
            config = experiment.config.values
            latest = max(experiment.checkpoints, key=lambda cp: cp.epoch, default=None)
            best = next(
                (cp for cp in experiment.checkpoints if cp.name in {"best", "best.pt"}), None
            )
            progress = (
                f"epoch {experiment.current_epoch}/{experiment.total_epochs}"
                if experiment.status == "running"
                else "—"
            )
            values = [
                experiment.experiment_id,
                experiment.study_id,
                model_type_label(experiment.model_type),
                str(config.get("data", {}).get("dataset_version", "—")),
                str(config.get("augmentation", {}).get("profile", "—")),
                experiment_status_label(experiment.status),
                progress,
                format_score(best.map) if best else "—",
                (best or latest).name if best or latest else "—",
            ]
            for col, value in enumerate(values, start=1):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, experiment.experiment_id)
                self.table.setItem(row, col, cell)
        self.table.blockSignals(False)
        if current:
            for row in range(self.table.rowCount()):
                if self.table.item(row, 1).text() == current:
                    self.table.setCurrentCell(row, 1)
                    break
        self._current_changed()
        self._update_buttons()

    def _checked_experiments(self) -> list[Experiment]:
        ids = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                ids.append(self.table.item(row, 1).text())
        return [self.ctx.backend.get_experiment(experiment_id) for experiment_id in ids]

    def _current_experiment_id(self) -> str | None:
        row = self.table.currentRow()
        return self.table.item(row, 1).text() if row >= 0 and self.table.item(row, 1) else None

    def _current_experiment(self) -> Experiment | None:
        experiment_id = self._current_experiment_id()
        return self.ctx.backend.get_experiment(experiment_id) if experiment_id else None

    def _selection_changed(self, _item: QTableWidgetItem | None = None) -> None:
        self._update_buttons()

    def _current_changed(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            self.overview.clear()
            self.chart.set_series([])
            self.checkpoint_table.setRowCount(0)
            self.run_table.setRowCount(0)
            self.used_data.clear()
            self._update_buttons()
            return
        config = experiment.config.values
        data = config.get("data", {})
        overview_lines = [
            f"実験群: {experiment.study_id}",
            f"説明: {experiment.description or '—'}",
            f"データセット版: {data.get('dataset_version', '—')}",
            f"交差検証分割: {data.get('split_id', '—')}",
            f"画像分類: {classification_label(data.get('classification', '未設定'))}",
            f"品質条件: {quality_filter_label(data.get('quality_filter', '未設定'))}",
            f"モデル: {model_type_label(experiment.model_type)}",
            "",
            "主要設定:",
            *[
                f"{config_key_label(key)}: {self._display_value(key, value)}"
                for key, value in flatten_config(config).items()
                if key != "data.used_item_ids"
            ],
            f"実使用データ一覧: {len(experiment.used_item_ids)} 件",
        ]
        self.overview.setPlainText("\n".join(overview_lines))
        points_loss = experiment.history
        self.chart.set_series(
            [
                (
                    "loss",
                    QColor("#2563eb"),
                    [float(p.epoch) for p in points_loss],
                    [float(p.loss) for p in points_loss],
                ),
                (
                    "mAP",
                    QColor("#dc2626"),
                    [float(p.epoch) for p in points_loss if p.map is not None],
                    [float(p.map) for p in points_loss if p.map is not None],
                ),
            ]
        )
        best = next((cp for cp in experiment.checkpoints if cp.name in {"best", "best.pt"}), None)
        if best and best.map is not None:
            self.chart.set_highlight("mAP", float(best.epoch), float(best.map))
        self.checkpoint_table.setRowCount(len(experiment.checkpoints))
        for row, checkpoint in enumerate(experiment.checkpoints):
            values = [
                checkpoint.name,
                str(checkpoint.epoch),
                format_score(checkpoint.map, 4),
                format_datetime(checkpoint.saved_at),
            ]
            for col, value in enumerate(values):
                self.checkpoint_table.setItem(row, col, QTableWidgetItem(value))
        self.run_table.setRowCount(len(experiment.runs))
        for row, run in enumerate(experiment.runs):
            values = [
                str(run.attempt),
                format_datetime(run.started_at),
                format_datetime(run.finished_at) if run.finished_at else "実行中",
                experiment_status_label(run.result),
                self._format_environment(run.environment),
            ]
            for col, value in enumerate(values):
                self.run_table.setItem(row, col, QTableWidgetItem(value))
        self.used_data.setPlainText(
            f"実使用データ数: {len(experiment.used_item_ids)} 件\n"
            + "\n".join(experiment.used_item_ids)
        )
        self._update_buttons()

    def _update_buttons(self) -> None:
        selected = self._checked_experiments()
        current = self._current_experiment()
        self.button_map["compare"].setEnabled(len(selected) >= 2)
        self.button_map["result"].setEnabled(current is not None and current.status == "completed")
        self.button_map["copy"].setEnabled(current is not None)
        self.button_map["stop"].setEnabled(current is not None and current.status == "running")
        self.button_map["retry"].setEnabled(
            current is not None and current.status in {"failed", "stopped"}
        )
        self.button_map["send"].setEnabled(
            current is not None and current.status == "completed" and bool(current.checkpoints)
        )
        self.button_map["edit"].setEnabled(current is not None and current.status == "draft")

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
        if key == "checkpoint.best_metric" and value == "instance_map":
            return "インスタンス平均適合率（mAP）"
        if key == "checkpoint.best_mode":
            return "最大化" if value == "max" else "最小化" if value == "min" else str(value)
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
                self, "学習を中断", f"{experiment.experiment_id} を中断しますか？"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        job = self.ctx.jobs.find(f"training:{experiment.experiment_id}")
        if job:
            job.cancel()
        self.ctx.backend.finish_training(experiment.experiment_id, "stopped")
        self.refresh()

    def retry_selected(self) -> None:
        experiment = self._current_experiment()
        if not experiment or experiment.status not in {"failed", "stopped"}:
            return
        experiment = self.ctx.backend.retry_experiment(experiment.experiment_id)
        self._start_job(experiment.experiment_id)
        self.refresh()

    def _start_job(self, experiment_id: str) -> None:
        experiment = self.ctx.backend.get_experiment(experiment_id)
        total = max(1, experiment.total_epochs)
        interval = max(1, int(experiment.config.values["checkpoint"]["validation_interval"]))

        def record(epoch: int) -> None:
            progress = epoch / total
            loss = max(0.01, 1.2 * math.exp(-3 * progress) + 0.01 * (epoch % 3))
            map_value = (
                min(0.99, 0.35 + 0.6 * progress)
                if epoch % interval == 0 or epoch == total
                else None
            )
            self.ctx.backend.record_epoch(experiment_id, epoch, loss, map_value)

        job = FakeJob(f"学習 {experiment_id}", total, 200, record, key=f"training:{experiment_id}")
        job.finished.connect(
            lambda ok, _message: self.ctx.backend.finish_training(
                experiment_id, "completed" if ok else "stopped"
            )
        )
        self.ctx.jobs.start(job)

    def send_selected(self) -> SendToCandidatesDialog | None:
        experiment = self._current_experiment()
        if not experiment or experiment.status != "completed" or not experiment.checkpoints:
            return None
        dialog = SendToCandidatesDialog(experiment, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.ctx.navigator.navigate(PageId.CANDIDATES, **dialog.transition_params())
        return dialog
