"""モデル学習設定を編集し、モック学習を開始する画面。"""

from __future__ import annotations

import copy
import math
from typing import Any

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...context import DEFAULT_CHANNEL, AppContext
from ...jobs import FakeJob
from ...labels import classification_label, config_key_label, model_type_label, quality_filter_label
from ...navigation import PageId
from ...theme import Color, mono_font, numeric_font, set_style
from ...widgets.form import CollapsibleSection, FormSection
from ...widgets.page_base import BasePage
from ...widgets.table import mark_primary
from .augmentation_dialog import AugmentationDialog

LABELS = {
    "data": "データセット",
    "model": "モデル固有設定",
    "training": "共通学習設定",
    "checkpoint": "途中保存モデル / 評価",
    "augmentation": "データ拡張",
    "dataset_version": "データセット版",
    "split_id": "交差検証分割",
    "seed": "乱数シード",
    "classification": "画像分類",
    "quality_filter": "品質条件",
    "epochs": "エポック数",
    "batch_size": "バッチサイズ",
    "learning_rate": "学習率",
    "weight_decay": "重み減衰",
    "early_stopping": "早期終了",
    "enabled": "有効",
    "patience": "待機エポック数",
    "profile": "プロファイル",
    "save_every": "保存間隔（エポック）",
    "save_last": "最終モデルを保存",
    "save_best": "最良モデルを保存",
    "best_metric": "最良モデル判定指標",
    "best_mode": "判定方向",
    "validation_interval": "検証間隔",
    "type": "モデル種類",
    "input": "入力画像 / 前処理",
    "anchors": "アンカー設定",
    "rpn": "RPN 設定",
    "roi": "ROI 設定",
    "num_classes": "クラス数",
    "pretrained_weights": "事前学習済み重み",
    "backbone": "バックボーン",
    "trainable_backbone_layers": "学習するバックボーン層数",
    "min_size": "入力画像の短辺サイズ",
    "max_size": "入力画像の最大辺サイズ",
    "image_mean": "画像平均",
    "image_std": "画像標準偏差",
    "sizes": "アンカーサイズ",
    "aspect_ratios": "アンカー縦横比",
    "fg_iou_thresh": "前景 IoU 閾値",
    "bg_iou_thresh": "背景 IoU 閾値",
    "batch_size_per_image": "バッチ数 / 画像",
    "positive_fraction": "正例割合",
    "pre_nms_top_n": "NMS 前候補数",
    "post_nms_top_n": "NMS 後候補数",
    "nms_thresh": "NMS 閾値",
    "pretrained_model": "事前学習済みモデル",
    "optimizer": "最適化手法",
    "normalize": "正規化",
    "scale_range": "スケール変動幅",
    "rescale": "リスケール",
    "bsize": "学習パッチサイズ",
    "nimg_per_epoch": "1 エポック当たり画像数",
    "min_train_masks": "最小マスク数",
    "class_weights": "クラス重み",
}

CHOICES = {
    "data.dataset_version": ("datasets", "確定済み学習版"),
    "data.split_id": ("splits", ""),
    "data.classification": ("classifications", "全分類"),
    "data.quality_filter": (["all", "good_only", "good_and_acceptable"], ""),
    "model.pretrained_weights": ("pretrained", ""),
    "model.backbone": ("backbones", ""),
    "model.pretrained_model": ("cellpose_models", ""),
    "model.optimizer": ("optimizers", ""),
    "checkpoint.best_metric": (["instance_map"], ""),
    "checkpoint.best_mode": (["max", "min"], ""),
    "augmentation.profile": ("profiles", ""),
}


class TrainingPage(BasePage):
    """実験設定・YAML プレビュー・学習開始を提供する。"""

    def __init__(
        self, ctx: AppContext, parent: QWidget | None = None, *, show_heading: bool = True
    ) -> None:
        super().__init__(
            ctx,
            "モデル学習",
            "学習設定を作成し、再現可能な実験として記録します。",
            parent,
            show_heading=show_heading,
        )
        self.options = ctx.backend.list_training_options()
        self.config = ctx.backend.default_experiment_config("mask_rcnn")
        self._configs_by_model: dict[str, dict[str, Any]] = {}
        self._model_widgets: dict[str, dict[str, QWidget]] = {}
        self.fields: dict[str, QWidget] = {}
        self._model_fields: dict[str, QWidget] = {}
        self._edit_id: str | None = None
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_layout.addWidget(splitter, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        self.form_layout = QVBoxLayout(form_host)
        self.form_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(form_host)
        self.form_host = form_host
        splitter.addWidget(scroll)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.addWidget(QLabel("設定プレビュー（YAML）"))
        self.yaml_preview = QTextEdit()
        self.yaml_preview.setReadOnly(True)
        self.yaml_preview.setFont(mono_font())
        set_style(self.yaml_preview, role="panel")
        preview_layout.addWidget(self.yaml_preview, 1)
        splitter.addWidget(preview_panel)
        splitter.setSizes([720, 310])
        self._build_form()
        buttons = QHBoxLayout()
        self.validate_button = QPushButton("設定を検証")
        self.save_button = QPushButton("下書き保存")
        self.start_button = QPushButton("学習開始")
        mark_primary(self.start_button)
        self.validate_button.clicked.connect(self.validate_config)
        self.save_button.clicked.connect(self.save_draft)
        self.start_button.clicked.connect(lambda: self.start_training())
        buttons.addStretch(1)
        buttons.addWidget(self.validate_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.start_button)
        self.content_layout.addLayout(buttons)
        self._refresh_yaml()

    def _build_form(self) -> None:
        """共通フォームとモデル別スタックを組み立てる。"""
        self.fields.clear()
        self._model_widgets.clear()
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        experiment = FormSection("実験")
        self.study = QComboBox()
        self.study.setEditable(True)
        self.study.addItems(sorted({item.study_id for item in self.ctx.backend.list_experiments()}))
        self.study.setCurrentText(self.config["experiment"].get("study_id", "foam_study"))
        self.study.setToolTip("experiment.study_id")
        self.experiment_id = QLineEdit()
        self.experiment_id.setReadOnly(True)
        set_style(self.experiment_id, role="readonly")
        self.experiment_id.setText(
            self.config["experiment"].get("id") or self.ctx.backend.next_experiment_id()
        )
        self.experiment_id.setToolTip("experiment.id")
        self.description_edit = QLineEdit(self.config["experiment"].get("description", ""))
        self.description_edit.setToolTip("experiment.description")
        experiment.add_row("実験群", self.study, "experiment.study_id")
        experiment.add_row("実験識別子", self.experiment_id, "experiment.id")
        experiment.add_row("説明", self.description_edit, "experiment.description")
        self.form_layout.addWidget(experiment)

        section, widgets = self._make_section("data", self.config["data"], "data")
        self.fields.update(widgets)
        self.form_layout.addWidget(section)
        self.dataset_note = QLabel("交差検証分割は正式評価用の検証用データセットとは別です。")
        set_style(self.dataset_note, role="note")
        self.form_layout.addWidget(self.dataset_note)
        self.estimate_label = QLabel()
        self.estimate_label.setTextFormat(Qt.TextFormat.RichText)
        set_style(self.estimate_label, state="warning")
        self.form_layout.addWidget(self.estimate_label)
        self.used_items_note = QLabel("実使用データ一覧は学習開始時に確定し、実験に保存されます。")
        set_style(self.used_items_note, role="note")
        self.form_layout.addWidget(self.used_items_note)

        self.model_type = QComboBox()
        self.model_type.addItem("Mask R-CNN", "mask_rcnn")
        self.model_type.addItem("Cellpose", "cellpose")
        self.model_type.setCurrentIndex(0 if self.config["model"]["type"] == "mask_rcnn" else 1)
        self._active_model = self.model_type.currentData()
        self.model_type.setToolTip("model.type")
        model_switch = FormSection("モデル")
        model_switch.add_row("モデル種類", self.model_type, "model.type")
        self.form_layout.addWidget(model_switch)
        self.model_stack = QStackedWidget()
        for model_name in ("mask_rcnn", "cellpose"):
            section, model_widgets = self._make_section(
                "model",
                self.config["model"]
                if self.config["model"]["type"] == model_name
                else self.ctx.backend.default_experiment_config(model_name)["model"],
                "model",
                model_name,
            )
            section.setTitle(
                "Mask R-CNN 固有設定" if model_name == "mask_rcnn" else "Cellpose 固有設定"
            )
            self._model_fields[model_name] = section
            self._model_widgets[model_name] = model_widgets
            self.model_stack.addWidget(section)
        self.model_section = CollapsibleSection("モデル固有設定", self.model_stack)
        self.model_section.button.setText("モデル固有設定 ▶")
        self.form_layout.addWidget(self.model_section)
        self.model_note = QLabel()
        set_style(self.model_note, role="note")
        self.form_layout.addWidget(self.model_note)

        for key in ("training", "augmentation", "checkpoint"):
            section, widgets = self._make_section(key, self.config[key], key)
            self.fields.update(widgets)
            if key in {"augmentation", "checkpoint"}:
                collapsible = CollapsibleSection(section.title(), section)
                collapsible.button.setText(f"{section.title()} ▶")
                self.form_layout.addWidget(collapsible)
            else:
                self.form_layout.addWidget(section)
            if key == "augmentation":
                profile_buttons = QHBoxLayout()
                self.profile_preview_button = QPushButton("プロファイルをプレビュー")
                self.profile_edit_button = QPushButton("表示 / 編集…")
                self.profile_preview_button.clicked.connect(self.open_augmentation_dialog)
                self.profile_edit_button.clicked.connect(self.open_augmentation_dialog)
                profile_buttons.addWidget(self.profile_preview_button)
                profile_buttons.addWidget(self.profile_edit_button)
                section.form.addRow(profile_buttons)
        self._bind_signals()
        self._update_model_stack()
        self._update_estimate()

    def _make_section(
        self, top: str, data: dict[str, Any], prefix: str, model_name: str | None = None
    ) -> tuple[FormSection, dict[str, QWidget]]:
        """設定辞書の値から入力欄を再帰生成する。"""
        heading = config_key_label(prefix)
        if heading == prefix:
            heading = LABELS.get(top, top)
        section = FormSection(heading)
        widgets: dict[str, QWidget] = {}
        self._add_values(section, data, prefix, widgets, top, model_name)
        return section, widgets

    def _add_values(
        self,
        section: FormSection,
        values: dict[str, Any],
        prefix: str,
        widgets: dict[str, QWidget],
        top: str,
        model_name: str | None,
    ) -> None:
        for key, value in values.items():
            path = f"{prefix}.{key}"
            if path in {"data.used_item_ids", "data.input_channels"}:
                continue
            if isinstance(value, dict):
                if key in {"rpn", "roi"}:
                    child = FormSection(LABELS[key])
                    collapsible = CollapsibleSection("詳細設定", child)
                    self._add_values(child, value, path, widgets, top, model_name)
                    section.form.addRow(collapsible)
                else:
                    child_label = LABELS.get(key, config_key_label(path))
                    child = FormSection(child_label)
                    self._add_values(child, value, path, widgets, top, model_name)
                    section.form.addRow(child)
                continue
            if path == "model.type":
                continue
            control = self._control(path, value)
            control.installEventFilter(self)
            widgets[path] = control
            label = config_key_label(path)
            if label == path:
                label = LABELS.get(key, key)
            section.add_row(label, control, path)

    def _control(self, path: str, value: Any) -> QWidget:
        """内部キーに応じた入力部品を作る。"""
        if path in CHOICES:
            source, _special = CHOICES[path]
            items = source if isinstance(source, list) else self.options.get(source, [])
            if path == "data.classification":
                items = ["all", *items]
            combo = QComboBox()
            for item in items:
                display = (
                    model_type_label(item)
                    if path == "model.type"
                    else classification_label(item)
                    if path == "data.classification"
                    else "インスタンス平均適合率（mAP）"
                    if path == "checkpoint.best_metric" and item == "instance_map"
                    else "最大化"
                    if path == "checkpoint.best_mode" and item == "max"
                    else "最小化"
                    if path == "checkpoint.best_mode" and item == "min"
                    else quality_filter_label(item)
                    if path == "data.quality_filter"
                    else "良のみ"
                    if item == "good_only"
                    else "良・可"
                    if item == "good_and_acceptable"
                    else str(item)
                )
                combo.addItem(display, item)
            wanted = "all" if path == "data.classification" and value == "all" else value
            index = combo.findData(wanted)
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.setToolTip(path)
            return combo
        if isinstance(value, bool):
            control = QCheckBox()
            control.setChecked(value)
        elif isinstance(value, int):
            control = QSpinBox()
            control.setRange(0, 1000000)
            control.setValue(value)
        elif isinstance(value, float):
            control = QDoubleSpinBox()
            control.setRange(-1000000.0, 1000000.0)
            control.setDecimals(6)
            control.setSingleStep(0.01)
            control.setValue(value)
        elif isinstance(value, list):
            control = QLineEdit(", ".join(str(item) for item in value))
            control.setPlaceholderText("カンマ区切り")
        else:
            control = QLineEdit("" if value is None else str(value))
        control.setToolTip(path)
        return control

    def _bind_signals(self) -> None:
        """変更時にプレビューと件数を更新する。"""
        self.model_type.currentIndexChanged.connect(self._switch_model)
        self.study.currentTextChanged.connect(self._refresh_yaml)
        self.description_edit.textChanged.connect(self._refresh_yaml)
        for widget in [
            *self.fields.values(),
            *self._model_widgets.get("mask_rcnn", {}).values(),
            *self._model_widgets.get("cellpose", {}).values(),
        ]:
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._on_config_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_config_changed)
            elif isinstance(widget, QSpinBox | QDoubleSpinBox):
                widget.valueChanged.connect(self._on_config_changed)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._on_config_changed)
            elif widget is self.fields.get("data.input_channels"):
                for checkbox in widget.findChildren(QCheckBox):
                    checkbox.toggled.connect(self._on_config_changed)

    def _switch_model(self, _index: int) -> None:
        old_type = getattr(self, "_active_model", self.config["model"]["type"])
        self._configs_by_model[old_type] = self._collect_config(model_override=old_type)
        model_type = self.model_type.currentData()
        base = self._configs_by_model.get(
            model_type, self.ctx.backend.default_experiment_config(model_type)
        )
        self.config = copy.deepcopy(base)
        previous = self._configs_by_model[old_type]
        for section in ("experiment", "data", "training", "augmentation", "checkpoint"):
            self.config[section] = copy.deepcopy(previous[section])
        self._active_model = model_type
        self._build_form()
        self._refresh_yaml()

    def _update_model_stack(self) -> None:
        self.model_stack.setCurrentIndex(0 if self.model_type.currentData() == "mask_rcnn" else 1)
        is_cellpose = self.model_type.currentData() == "cellpose"
        self.model_note.setVisible(is_cellpose)
        self.model_note.setText(
            "セル確率閾値・フロー閾値はモデル比較・リリースの推論設定で管理します。"
        )

    def _on_config_changed(self, *_args) -> None:
        self._update_estimate()
        self._refresh_yaml()

    def eventFilter(self, watched, event) -> bool:
        """フォーカス中の設定に対応する YAML 行を強調する。"""
        if event.type() == QEvent.Type.FocusIn and watched.toolTip():
            self._highlight_yaml_path(watched.toolTip())
        return super().eventFilter(watched, event)

    def _highlight_yaml_path(self, path: str) -> None:
        """設定キーに対応する YAML の行へ CHANGED 色を付ける。"""
        leaf = path.rsplit(".", 1)[-1]
        block = self.yaml_preview.document().firstBlock()
        while block.isValid():
            if block.text().lstrip().startswith(f"{leaf}:"):
                selection = QTextEdit.ExtraSelection()
                selection.cursor = self.yaml_preview.textCursor()
                selection.cursor.setPosition(block.position())
                selection.cursor.movePosition(
                    selection.cursor.MoveOperation.EndOfBlock, selection.cursor.MoveMode.KeepAnchor
                )
                selection.format.setBackground(QColor(Color.CHANGED))
                self.yaml_preview.setExtraSelections([selection])
                return
            block = block.next()
        self.yaml_preview.setExtraSelections([])

    def _collect_config(self, model_override: str | None = None) -> dict[str, Any]:
        """現在の入力値を仕様の入れ子設定へ戻す。"""
        result = copy.deepcopy(self.config)
        result["experiment"]["study_id"] = self.study.currentText().strip() or "foam_study"
        result["experiment"]["description"] = self.description_edit.text()
        model_type = model_override or self.model_type.currentData()
        result["model"]["type"] = model_type
        widgets = dict(self.fields)
        widgets.update(self._model_widgets.get(model_type, {}))
        for path, widget in widgets.items():
            key_path = path.split(".")
            value = self._read_control(path, widget)
            target = result
            for part in key_path[:-1]:
                target = target[part]
            target[key_path[-1]] = value
        result["data"]["input_channels"] = [DEFAULT_CHANNEL]
        result["experiment"]["id"] = self._edit_id or self.experiment_id.text().strip()
        return result

    def _read_control(self, path: str, widget: QWidget) -> Any:
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox | QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QLineEdit):
            text = widget.text().strip()
            if "," in text:
                return [
                    self._parse_number(token.strip()) for token in text.split(",") if token.strip()
                ]
            original = self._lookup_default(path)
            if isinstance(original, list):
                return [
                    self._parse_number(token.strip()) for token in text.split(",") if token.strip()
                ]
            if isinstance(original, float) or original is None:
                return None if not text else float(text)
            return text
        return None

    def _lookup_default(self, path: str) -> Any:
        target: Any = self.ctx.backend.default_experiment_config(self.model_type.currentData())
        for part in path.split("."):
            target = target.get(part) if isinstance(target, dict) else None
        return target

    @staticmethod
    def _parse_number(value: str) -> float | int | str:
        try:
            number = float(value)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value

    def _refresh_yaml(self, *_args) -> None:
        if not hasattr(self, "yaml_preview"):
            return
        try:
            config = self._collect_config()
            preview_config = copy.deepcopy(config)
            preview_config["data"]["used_item_ids"] = "学習開始時に確定"
            self.yaml_preview.setPlainText(self._to_yaml(preview_config))
        except (KeyError, ValueError):
            self.yaml_preview.setPlainText("設定を入力してください")

    @staticmethod
    def _to_yaml(config: dict[str, Any]) -> str:
        import yaml

        return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)

    def _update_estimate(self, *_args) -> None:
        if not hasattr(self, "estimate_label"):
            return
        try:
            config = self._collect_config()
            train, valid = self.ctx.backend.estimate_training_items(
                config["data"].get("dataset_version"),
                config["data"].get("classification", "all"),
                config["data"].get("quality_filter", "all"),
            )
            numeric_family = numeric_font().family()
            self.estimate_label.setText(
                "見込み使用データ数：学習 "
                f"<span style='font-family:{numeric_family};font-weight:bold'>{train}</span> 件 / "
                "学習内検証 "
                f"<span style='font-family:{numeric_family};font-weight:bold'>{valid}</span> 件"
            )
        except (KeyError, ValueError, TypeError):
            self.estimate_label.setText("見込み使用データ数：条件を確認してください")

    def open_augmentation_dialog(self) -> AugmentationDialog:
        """プロファイルを編集し、保存された新版を選択する。"""
        profile = self.fields["augmentation.profile"].currentData()
        dialog = AugmentationDialog(self.ctx.backend, profile, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            saved = self.ctx.backend.save_augmentation_profile(dialog.build_profile())
            combo = self.fields["augmentation.profile"]
            combo.addItem(saved.profile_id, saved.profile_id)
            combo.setCurrentIndex(combo.count() - 1)
            self.options = self.ctx.backend.list_training_options()
            self.ctx.status.show_message(f"{saved.profile_id} を新しい版として保存しました")
        return dialog

    def _refresh_options(self) -> None:
        """選択肢を読み直し、現在の選択を可能な限り保つ。"""
        self.options = self.ctx.backend.list_training_options()
        for path, (source, _special) in CHOICES.items():
            widget = self.fields.get(path)
            if not isinstance(widget, QComboBox):
                continue
            selected = widget.currentData()
            items = source if isinstance(source, list) else self.options.get(source, [])
            if path == "data.classification":
                items = ["all", *items]
            widget.blockSignals(True)
            widget.clear()
            for item in items:
                display = (
                    classification_label(item)
                    if path == "data.classification"
                    else quality_filter_label(item)
                    if path == "data.quality_filter"
                    else str(item)
                )
                widget.addItem(display, item)
            index = widget.findData(selected)
            if index < 0:
                default_value = self._lookup_default(path)
                index = widget.findData(default_value)
            widget.setCurrentIndex(max(0, index))
            widget.blockSignals(False)

    def validate_config(self) -> list[dict[str, str]]:
        """Backend 検証結果をダイアログに表示する。"""
        self.config = self._collect_config()
        results = self.ctx.backend.validate_experiment_config(self.config)
        text = (
            "\n".join(f"{item['level']}: {item['message']}" for item in results)
            or "OK: 設定に問題はありません"
        )
        QMessageBox.information(self, "設定の検証", text)
        return results

    def save_draft(self) -> str:
        """下書きを保存し、実験 ID を保持する。"""
        self.config = self._collect_config()
        if self._has_non_draft_id(self.config["experiment"]["id"]):
            QMessageBox.warning(self, "下書き保存", "下書き以外の実験識別子は使用できません。")
            return ""
        experiment = self.ctx.backend.save_experiment_draft(self.config, self._edit_id)
        self._edit_id = experiment.experiment_id
        self.config["experiment"]["id"] = experiment.experiment_id
        self.experiment_id.setText(experiment.experiment_id)
        self.ctx.status.show_message(f"{experiment.experiment_id} を下書き保存しました")
        return experiment.experiment_id

    def _has_non_draft_id(self, experiment_id: str) -> bool:
        """指定 ID が下書き以外の既存実験か調べる。"""
        for experiment in self.ctx.backend.list_experiments():
            if experiment.experiment_id == experiment_id:
                return experiment.status != "draft"
        return False

    def start_training(self, confirm: bool = True) -> str | None:
        """学習を登録してジョブを開始し、実験一覧へ移る。"""
        self.config = self._collect_config()
        results = self.ctx.backend.validate_experiment_config(self.config)
        errors = [item for item in results if item["level"] == "error"]
        if errors:
            QMessageBox.warning(self, "設定エラー", "\n".join(item["message"] for item in errors))
            return None
        if confirm:
            warnings = [item["message"] for item in results if item["level"] == "warning"]
            prompt = "この設定で学習を開始しますか？"
            if warnings:
                prompt += "\n\n警告:\n" + "\n".join(warnings)
            if QMessageBox.question(self, "学習開始", prompt) != QMessageBox.StandardButton.Yes:
                return None
        if self._has_non_draft_id(self.config["experiment"]["id"]):
            QMessageBox.warning(self, "学習開始", "下書き以外の実験識別子は使用できません。")
            return None
        experiment = self.ctx.backend.start_training(self.config, self._edit_id)
        experiment_id = experiment.experiment_id
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

        job = FakeJob(
            f"学習 {experiment_id}",
            total_steps=total,
            interval_ms=200,
            on_step=record,
            key=f"training:{experiment_id}",
        )
        job.finished.connect(
            lambda _ok, _message: self.ctx.backend.finish_training(experiment_id, "completed")
        )
        self.ctx.jobs.start(job)
        self.ctx.status.show_message(f"{experiment_id} の学習を開始しました")
        self._edit_id = None
        self.config["experiment"]["id"] = self.ctx.backend.next_experiment_id()
        self.experiment_id.setText(self.config["experiment"]["id"])
        self._refresh_yaml()
        self.ctx.navigator.navigate(PageId.EXPERIMENTS, select=experiment_id)
        return experiment_id

    def on_enter(self, params: dict[str, Any]) -> None:
        """複製・下書き編集の設定を読み込む。"""
        self._refresh_options()
        copy_from = params.get("copy_from")
        edit = params.get("edit")
        if not copy_from and not edit:
            return
        source = self.ctx.backend.get_experiment(copy_from or edit)
        self._edit_id = edit
        self.config = copy.deepcopy(source.config.values)
        self.config["experiment"]["id"] = edit if edit else self.ctx.backend.next_experiment_id()
        self._configs_by_model.clear()
        self._build_form()
        self._refresh_yaml()
