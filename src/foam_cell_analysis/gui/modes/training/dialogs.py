"""学習モードで使う確認・比較ダイアログ。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ....services.models import Experiment
from ...labels import (
    classification_label,
    config_key_label,
    model_type_label,
    quality_filter_label,
)
from ...theme import SERIES, Color
from ...widgets.chart import LineChart
from ...widgets.table import fit_table_columns, mark_primary, setup_table


def flatten_config(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """入れ子の設定を比較表用のキー・値にする。"""
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten_config(item, path))
        else:
            result[path] = item
    return result


class ExperimentCompareDialog(QDialog):
    """複数実験の設定差分と AP 曲線を表示する。"""

    def __init__(self, experiments: list[Experiment], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.experiments = experiments
        self.setWindowTitle("実験を比較")
        self.resize(900, 700)
        self.setMinimumSize(780, 580)
        layout = QVBoxLayout(self)
        self.differences_only = QCheckBox("差分のみ表示")
        self.table = QTableWidget()
        self.table.setColumnCount(len(experiments) + 1)
        setup_table(self.table, stretch_column=0)
        self.chart = LineChart()
        layout.addWidget(self.differences_only)
        split = QSplitter(Qt.Orientation.Horizontal)
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(QLabel("OOF AP の推移"))
        plot_layout.addWidget(self.chart)
        split.addWidget(self.table)
        split.addWidget(plot_panel)
        split.setSizes([430, 470])
        layout.addWidget(split, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("閉じる")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.differences_only.toggled.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """設定差分表と系列を更新する。"""
        configs = [flatten_config(item.config.values) for item in self.experiments]
        keys = sorted(set().union(*(config.keys() for config in configs)))
        differing = [
            key
            for key in keys
            if len({self._compare_value(key, config.get(key)) for config in configs}) > 1
        ]
        visible = differing if self.differences_only.isChecked() else keys
        self.table.setRowCount(len(visible))
        self.table.setColumnCount(len(self.experiments) + 1)
        self.table.setHorizontalHeaderLabels(
            ["設定項目", *[e.experiment_id for e in self.experiments]]
        )
        for row, key in enumerate(visible):
            label = QTableWidgetItem(config_key_label(key))
            label.setToolTip(key)
            self.table.setItem(row, 0, label)
            for col, config in enumerate(configs, start=1):
                item = QTableWidgetItem(self._display_value(key, config.get(key)))
                if key in differing:
                    item.setBackground(QColor(Color.CHANGED))
                self.table.setItem(row, col, item)
        fit_table_columns(self.table)
        colors = [QColor(color) for color in SERIES]
        series = []
        for index, experiment in enumerate(self.experiments):
            points = [entry for entry in experiment.history if entry.map is not None]
            series.append(
                (
                    experiment.experiment_id,
                    colors[index % len(colors)],
                    [float(item.epoch) for item in points],
                    [float(item.map) for item in points],
                )
            )
        self.chart.set_series(series[:4])
        values = [value for item in series[:4] for value in item[3]]
        lower = 0.5 if values and min(values) >= 0.5 else None
        self.chart.set_y_range(lower, 1.0 if values and max(values) <= 1.0 else None)

    @staticmethod
    def _compare_value(key: str, value: Any) -> str:
        """差分比較用の値を正規化する。"""
        if key == "data.used_item_ids" and isinstance(value, list):
            return str(len(value))
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        return str(value)

    @classmethod
    def _display_value(cls, key: str, value: Any) -> str:
        """比較表の値を画面表示用に整える。"""
        if key == "data.used_item_ids" and isinstance(value, list):
            return f"{len(value)} 件"
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
            return "OOF 平均適合率（AP）・最大"
        if key == "model.pretrained_weights":
            return {"coco": "COCO", "imagenet": "ImageNet"}.get(str(value), str(value))
        if key == "model.backbone":
            return {"resnet50_fpn_v2": "ResNet-50 FPN v2", "resnet101_fpn": "ResNet-101 FPN"}.get(
                str(value), str(value)
            )
        if isinstance(value, list):
            return "、".join(str(item) for item in value)
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)


class SendToCandidatesDialog(QDialog):
    """最終学習モデルを比較候補へ渡す。"""

    def __init__(self, experiment: Experiment, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.experiment = experiment
        self.setWindowTitle("モデル比較へ送る")
        self.resize(520, 260)
        self.setMinimumSize(480, 250)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.checkpoint = QComboBox()
        final_model = "final.pt"
        if any(entry.name == final_model for entry in experiment.checkpoints):
            self.checkpoint.addItem(final_model)
        self.comment = QLineEdit()
        form.addRow("実験", QLabel(experiment.experiment_id))
        form.addRow("最終学習モデル", self.checkpoint)
        form.addRow("説明", self.comment)
        layout.addLayout(form)
        layout.addWidget(QLabel("推論設定や正式リリースはモデル比較・リリースで行います。"))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("モデル比較へ送る")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        mark_primary(buttons.button(QDialogButtonBox.StandardButton.Ok))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def transition_params(self) -> dict[str, Any]:
        """設計書の候補追加遷移パラメータを返す。"""
        return {
            "action": "add_candidate",
            "experiment_id": self.experiment.experiment_id,
            "checkpoint": self.checkpoint.currentText(),
            "comment": self.comment.text().strip(),
        }
