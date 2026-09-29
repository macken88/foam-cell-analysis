"""モデル学習設定を編集し、モック学習を開始する画面。"""

from __future__ import annotations

import copy
from typing import Any

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ....services.backend import normalization_for_weights
from ...context import DEFAULT_CHANNEL, AppContext
from ...labels import (
    config_key_label,
    model_type_label,
    training_choice_label,
)
from ...navigation import PageId
from ...settings import app_settings
from ...theme import Color, body_font, mono_font, numeric_font, set_style
from ...widgets.form import CollapsibleSection, FormSection
from ...widgets.page_base import BasePage
from ...widgets.table import bind_button_action, mark_primary
from .augmentation_dialog import AugmentationDialog

LABELS = {
    "data": "データセット",
    "model": "モデル固有設定",
    "training": "共通学習設定",
    "checkpoint": "途中保存モデル / 評価",
    "augmentation": "データ拡張",
    "dataset_version": "データセット版",
    "cv": "交差検証",
    "n_folds": "分割数",
    "stratify_by_classification": "画像分類で層別",
    "group_by_source_folder": "取り込み元フォルダ単位で分割",
    "seed": "乱数シード（分割・学習）",
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
    "save_fold_models": "各フォールドのモデルを保存",
    "validation_interval": "各フォールドの検証間隔",
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
    "data.classification": ("classifications", "全分類"),
    "data.quality_filter": (["all", "good_only", "good_and_acceptable"], ""),
    "model.pretrained_weights": ("pretrained", ""),
    "model.backbone": ("backbones", ""),
    "model.pretrained_model": ("cellpose_models", ""),
    "model.optimizer": ("optimizers", ""),
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
        self._queue_edit_id: str | None = None
        self._queue_edit_unavailable = False
        self._pre_queue_edit: dict[str, Any] | None = None
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview_splitter = splitter
        self.content_layout.addWidget(splitter, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll = scroll
        form_host = QWidget()
        self.form_root_layout = QVBoxLayout(form_host)
        self.form_root_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.form_root_layout.setSpacing(16)
        self.summary_row = QFrame()
        self.summary_row.setObjectName("trainingSummaryCard")
        self.summary_row.setProperty("role", "trainingSummary")
        summary_layout = QHBoxLayout(self.summary_row)
        summary_layout.setContentsMargins(12, 8, 12, 8)
        self.summary_label = QLabel()
        self.summary_label.setObjectName("trainingSummary")
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label, 1)
        self.validation_result_button = QPushButton()
        self.validation_result_button.hide()
        self.validation_result_button.clicked.connect(self._show_validation_results)
        summary_layout.addWidget(self.validation_result_button)
        self.preview_button = QPushButton("設定プレビュー（YAML）")
        self.preview_button.setObjectName("trainingPreviewButton")
        self.preview_button.setCheckable(True)
        summary_layout.addWidget(self.preview_button)
        self.form_root_layout.addWidget(self.summary_row)
        self.form_columns = QWidget()
        self.form_layout = QHBoxLayout(self.form_columns)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.setSpacing(22)
        self.left_column_widget = QWidget(self.form_columns)
        self.left_column = QVBoxLayout(self.left_column_widget)
        self.left_column.setContentsMargins(0, 0, 0, 0)
        self.left_column.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.right_column_widget = QWidget(self.form_columns)
        self.right_column = QVBoxLayout(self.right_column_widget)
        self.right_column.setContentsMargins(0, 0, 0, 0)
        self.right_column.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.form_layout.addWidget(self.left_column_widget, 1)
        self.form_layout.addWidget(self.right_column_widget, 1)
        self.form_root_layout.addWidget(self.form_columns)
        self._form_widgets: list[tuple[QWidget, int]] = []
        self._two_columns: bool | None = None
        scroll.setWidget(form_host)
        self.form_host = form_host
        scroll.installEventFilter(self)
        splitter.addWidget(scroll)
        self.preview_panel = QWidget()
        preview_layout = QVBoxLayout(self.preview_panel)
        preview_layout.addWidget(QLabel("設定プレビュー（YAML）"))
        self.yaml_preview = QTextEdit()
        self.yaml_preview.setReadOnly(True)
        self.yaml_preview.setFont(mono_font())
        set_style(self.yaml_preview, role="panel")
        preview_layout.addWidget(self.yaml_preview, 1)
        splitter.addWidget(self.preview_panel)
        splitter.setSizes(
            [1000, 0]
            if not app_settings().value("training/yamlPreview", False, type=bool)
            else [720, 310]
        )
        buttons = QHBoxLayout()
        self.validate_button = QPushButton("設定を検証")
        self.save_button = QPushButton("下書き保存")
        self.cancel_queue_edit_button = QPushButton("キャンセル")
        self.cancel_queue_edit_button.hide()
        self.start_button = QPushButton("学習開始")
        self.queue_button = QPushButton("キューに追加")
        self.training_actions = {
            "validate": QAction("設定を検証", self),
            "save": QAction("下書き保存", self),
            "queue": QAction("キューに追加", self),
            "start": QAction("学習開始", self),
            "new": QAction("新しい実験", self),
        }
        self.training_actions["validate"].triggered.connect(self.validate_config)
        self.training_actions["save"].triggered.connect(self._save_button_clicked)
        self.training_actions["queue"].triggered.connect(self.enqueue_config)
        self.training_actions["start"].triggered.connect(lambda: self.start_training())
        self.training_actions["new"].triggered.connect(self._new_experiment)
        self.preview_action = QAction("設定プレビュー（YAML）", self)
        self.preview_action.setCheckable(True)
        self.preview_action.setChecked(
            app_settings().value("training/yamlPreview", False, type=bool)
        )
        self.preview_action.toggled.connect(self._set_yaml_preview_visible)
        self.preview_action.toggled.connect(self.preview_button.setChecked)
        self.preview_button.setChecked(self.preview_action.isChecked())
        self.preview_button.clicked.connect(self.preview_action.toggle)
        self.preview_panel.setVisible(self.preview_action.isChecked())
        self.augmentation_action = QAction("データ拡張を設定…", self)
        self.augmentation_action.triggered.connect(self.open_augmentation_dialog)
        self.reset_defaults_action = QAction("既定値に戻す", self)
        self.reset_defaults_action.setToolTip(
            "現在のモデルの既定値に戻します。実験群・説明・データセット版は保持します。"
        )
        self.reset_defaults_action.triggered.connect(self.reset_defaults)
        bind_button_action(self.validate_button, self.training_actions["validate"])
        bind_button_action(self.save_button, self.training_actions["save"])
        bind_button_action(self.queue_button, self.training_actions["queue"])
        bind_button_action(self.start_button, self.training_actions["start"])
        mark_primary(self.start_button)
        self._build_form()
        self.cancel_queue_edit_button.clicked.connect(self._cancel_queue_edit)
        self.validation_actions = QWidget()
        validation_actions_layout = QHBoxLayout(self.validation_actions)
        validation_actions_layout.setContentsMargins(0, 0, 0, 0)
        validation_actions_layout.addWidget(self.validate_button)
        validation_actions_layout.addWidget(self.save_button)
        self.execution_actions = QWidget()
        execution_actions_layout = QHBoxLayout(self.execution_actions)
        execution_actions_layout.setContentsMargins(0, 0, 0, 0)
        execution_actions_layout.addWidget(self.queue_button)
        execution_actions_layout.addWidget(self.cancel_queue_edit_button)
        execution_actions_layout.addWidget(self.start_button)
        buttons.addWidget(self.validation_actions)
        buttons.addStretch(1)
        buttons.addWidget(self.execution_actions)
        self.content_layout.addLayout(buttons)
        self._refresh_yaml()
        if ctx.queue_controller is not None:
            ctx.queue_controller.changed.connect(self._sync_queue_edit_state)

    def _new_experiment(self):
        if self._queue_edit_id:
            return
        self.on_enter({})

    def menu_actions(self):
        return {
            "edit": [self.reset_defaults_action],
            "file": [self.training_actions["new"], self.training_actions["save"]],
            "training": [
                self.training_actions["validate"],
                self.training_actions["queue"],
                self.training_actions["start"],
                None,
            ],
            "view": [self.preview_action],
            "tools": [self.augmentation_action],
        }

    def reset_defaults(self) -> None:
        """現在のモデルの既定値に戻し、実験の識別情報を保つ。"""
        if (
            QMessageBox.question(self, "既定値に戻す", "現在のモデルの既定値に戻しますか？")
            != QMessageBox.StandardButton.Yes
        ):
            return
        current = self._collect_config()
        defaults = self.ctx.backend.default_experiment_config(self.model_type.currentData())
        defaults["experiment"]["study_id"] = current["experiment"].get("study_id", "foam_study")
        defaults["experiment"]["description"] = current["experiment"].get("description", "")
        defaults["data"]["dataset_version"] = current["data"].get("dataset_version")
        defaults["experiment"]["id"] = current["experiment"].get("id")
        self.config = defaults
        self._configs_by_model[self.model_type.currentData()] = copy.deepcopy(defaults)
        self._clear_validation_results()
        self._build_form()
        self._refresh_yaml()

    def _build_form(self) -> None:
        """共通フォームとモデル別スタックを組み立てる。"""
        self._two_columns = None
        self.fields.clear()
        self._model_widgets.clear()
        for layout in (self.left_column, self.right_column):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                    item.widget().deleteLater()
        self._form_widgets.clear()
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
        self._add_form_widget(experiment, 0)

        section, widgets = self._make_section("data", self.config["data"], "data")
        self.fields.update(widgets)
        self.dataset_note = QLabel(
            "交差検証は学習用データセットの中だけで分割します。"
            "検証用データセット（val 版）は学習にも交差検証にも使いません。"
        )
        set_style(self.dataset_note, role="note")
        cv_section = widgets["data.cv.n_folds"].parentWidget()
        cv_section.form.addRow(self.dataset_note)
        self._add_form_widget(section, 0)
        self.estimate_label = QLabel()
        self.used_items_note = QLabel("実使用データ一覧は学習開始時に確定し、実験に保存されます。")
        set_style(self.used_items_note, role="note")
        self._add_form_widget(self.used_items_note, 0)

        self.model_type = QComboBox()
        self.model_type.addItem("Mask R-CNN", "mask_rcnn")
        self.model_type.addItem("Cellpose", "cellpose")
        self.model_type.setCurrentIndex(0 if self.config["model"]["type"] == "mask_rcnn" else 1)
        self._active_model = self.model_type.currentData()
        self.model_type.setToolTip("model.type")
        model_switch = FormSection("モデル")
        model_switch.add_row("モデル種類", self.model_type, "model.type")
        self._add_form_widget(model_switch, 1)
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
            self._promote_section_heading(section)
            self._model_fields[model_name] = section
            self._model_widgets[model_name] = model_widgets
            self.model_stack.addWidget(section)
        self._add_form_widget(self.model_stack, 1)
        self.model_note = QLabel()
        set_style(self.model_note, role="note")
        self._add_form_widget(self.model_note, 1)
        self.normalization_reset_note = QLabel()
        set_style(self.normalization_reset_note, role="note")
        self.normalization_reset_note.setWordWrap(True)
        self._add_form_widget(self.normalization_reset_note, 1)

        for key in ("training", "augmentation", "checkpoint"):
            section, widgets = self._make_section(key, self.config[key], key)
            self.fields.update(widgets)
            self._add_form_widget(section, 0 if key == "training" else 1)
            if key == "augmentation":
                self.profile_edit_button = QPushButton("データ拡張を設定…")
                self.profile_edit_button.setSizePolicy(
                    QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
                )
                self.profile_edit_button.clicked.connect(self.open_augmentation_dialog)
                profile_label = QLabel("プロファイル")
                profile = widgets["augmentation.profile"]
                row, _role = section.form.getWidgetPosition(profile)
                old_row = section.form.takeRow(row)
                if old_row.labelItem and old_row.labelItem.widget():
                    old_label = old_row.labelItem.widget()
                    old_label.setParent(None)
                    profile_label.deleteLater()
                    profile_label = old_label
                profile_row = QHBoxLayout()
                profile_row.addWidget(profile, 1)
                profile_row.addWidget(self.profile_edit_button)
                profile_row.addStretch(1)
                section.form.addRow(profile_label, profile_row)
        self._bind_signals()
        if isinstance(self.fields.get("data.cv.n_folds"), QSpinBox):
            self.fields["data.cv.n_folds"].setRange(2, 10)
        self._update_model_stack()
        self._update_estimate()
        self._update_summary()
        self._update_form_columns()

    def _add_form_widget(self, widget: QWidget, column: int) -> None:
        if isinstance(widget, FormSection):
            widget.setProperty("trainingSection", True)
            self._promote_section_heading(widget)
        self._form_widgets.append((widget, column))

    @staticmethod
    def _promote_section_heading(section: FormSection) -> None:
        section.setProperty("trainingSection", True)
        section.set_prominent_heading()
        font = body_font(12)
        font.setBold(True)
        section.heading_label.setFont(font)

    def _update_form_columns(self, available_width: int | None = None) -> None:
        if not hasattr(self, "form_layout"):
            return
        width = available_width if available_width is not None else self.scroll.width()
        two_columns = not self.preview_action.isChecked() and width >= 1100
        if self._two_columns is not None and two_columns == self._two_columns:
            return
        focus_widget = QApplication.focusWidget()
        if focus_widget is not None and not (
            focus_widget is self.form_host or self.form_host.isAncestorOf(focus_widget)
        ):
            focus_widget = None
        cursor_widget = focus_widget
        if cursor_widget is not None and not hasattr(cursor_widget, "cursorPosition"):
            editor = getattr(cursor_widget, "lineEdit", None)
            cursor_widget = editor() if callable(editor) else None
        cursor_position = (
            cursor_widget.cursorPosition()
            if cursor_widget is not None and hasattr(cursor_widget, "cursorPosition")
            else None
        )
        selection_start = (
            cursor_widget.selectionStart()
            if cursor_widget is not None and hasattr(cursor_widget, "selectionStart")
            else -1
        )
        selection_length = (
            len(cursor_widget.selectedText())
            if cursor_widget is not None and hasattr(cursor_widget, "selectedText")
            else 0
        )
        focus_path = self._config_path_for_widget(focus_widget)
        for layout in (self.left_column, self.right_column):
            while layout.count():
                layout.takeAt(0)
        for widget, preferred_column in self._form_widgets:
            if two_columns:
                target_layout = self.left_column if preferred_column == 0 else self.right_column
            else:
                target_layout = self.left_column
            target_parent = target_layout.parentWidget()
            if widget.parentWidget() is not target_parent:
                widget.setParent(target_parent)
            target_layout.addWidget(widget)
            if widget is self.model_note:
                widget.setVisible(self.model_type.currentData() == "cellpose")
            else:
                widget.show()
        self.right_column_widget.setVisible(two_columns)
        self._two_columns = two_columns
        self.form_layout.activate()
        if focus_widget is not None and focus_widget.isEnabled():
            focus_widget.setFocus(Qt.FocusReason.OtherFocusReason)
            if cursor_position is not None and cursor_widget is not None:
                cursor_widget.setCursorPosition(cursor_position)
                if selection_start >= 0 and selection_length:
                    cursor_widget.setSelection(selection_start, selection_length)
            if focus_path:
                self._highlight_yaml_path(focus_path)

    def _config_path_for_widget(self, widget: QWidget | None) -> str:
        if widget is None:
            return ""
        while widget is not None and widget is not self.form_host:
            if widget.toolTip():
                return widget.toolTip()
            widget = widget.parentWidget()
        return ""

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_form_columns()

    def _set_yaml_preview_visible(self, visible: bool) -> None:
        self.preview_panel.setVisible(visible)
        app_settings().setValue("training/yamlPreview", visible)
        self.preview_splitter.setSizes([720, 310] if visible else [max(1, self.width()), 0])
        self._update_form_columns()

    def _update_summary(self) -> None:
        if not hasattr(self, "summary_label") or not hasattr(self, "model_type"):
            return
        config = self._collect_config()
        data = config["data"]
        model = config["model"]
        train, per_fold = self.ctx.backend.estimate_training_items(
            data.get("dataset_version"),
            data.get("classification", "all"),
            data.get("quality_filter", "all"),
            data.get("cv", {}).get("n_folds", 5),
        )
        folds = data.get("cv", {}).get("n_folds", 5)
        dataset = data.get("dataset_version") or "未選択"
        classification = training_choice_label(
            "data.classification", str(data.get("classification", "all"))
        )
        quality = training_choice_label(
            "data.quality_filter", str(data.get("quality_filter", "all"))
        )
        if model["type"] == "mask_rcnn":
            model_spec = training_choice_label(
                "model.pretrained_weights", str(model.get("pretrained_weights", "coco"))
            )
        else:
            model_spec = str(model.get("pretrained_model", "cpsam"))
        fields = [
            ("データ", f"{dataset}・{classification}・{quality}・学習 {train} 件"),
            (
                "交差検証",
                "・".join(
                    [
                        f"{folds} 分割",
                        *(
                            ["層別"]
                            if data.get("cv", {}).get("stratify_by_classification", False)
                            else []
                        ),
                        *(
                            ["フォルダ単位"]
                            if data.get("cv", {}).get("group_by_source_folder", False)
                            else []
                        ),
                        f"各検証 約 {per_fold} 件",
                    ]
                ),
            ),
            ("モデル", f"{model_type_label(model['type'])}（{model_spec}）"),
            (
                "学習",
                f"{config['training']['epochs']} エポック・"
                f"バッチ {config['training']['batch_size']}・"
                f"学習率 {config['training']['learning_rate']}",
            ),
        ]
        num_family = numeric_font().family()
        self.summary_label.setText(
            "　│　".join(
                f"<b style='color:{Color.SLATE}'>{title}</b> "
                f"<span style='font-family:{num_family}'>{value}</span>"
                for title, value in fields
            )
        )

    def _clear_validation_results(self) -> None:
        self._validation_results = []
        self.validation_result_button.hide()

    def _show_validation_results(self) -> None:
        if not getattr(self, "_validation_results", None):
            return
        text = "\n".join(f"{item['level']}: {item['message']}" for item in self._validation_results)
        QMessageBox.information(self, "設定の検証", text)

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
            if path == "data.used_item_ids":
                continue
            if path == "model.input_channels":
                continue
            if (
                top == "model"
                and model_name == "cellpose"
                and path
                in {
                    "model.optimizer",
                    "model.class_weights",
                    "model.rescale",
                    "model.normalize",
                }
            ):
                continue
            if (
                top == "model"
                and model_name == "mask_rcnn"
                and path
                in {
                    "model.input.image_mean",
                    "model.input.image_std",
                }
            ):
                continue
            if isinstance(value, dict):
                if key in {"rpn", "roi"}:
                    child = FormSection(LABELS[key])
                    collapsible = CollapsibleSection(f"{key.upper()} 詳細設定", child)
                    self._add_values(child, value, path, widgets, top, model_name)
                    section.form.addRow(collapsible)
                else:
                    child_label = LABELS.get(key, config_key_label(path))
                    child = FormSection(child_label)
                    self._add_values(child, value, path, widgets, top, model_name)
                    if path == "model.input" and model_name == "mask_rcnn":
                        self.model_normalization_note = QLabel()
                        self.model_normalization_note.setFont(numeric_font())
                        self.model_normalization_note.setWordWrap(True)
                        set_style(self.model_normalization_note, role="note")
                        child.form.addRow("画像平均・標準偏差", self.model_normalization_note)
                    if path == "model.input.normalization":
                        section.form.addRow(CollapsibleSection("前処理 詳細設定", child))
                    else:
                        section.form.addRow(child)
                continue
            if path == "model.type":
                continue
            if path == "checkpoint.best_metric":
                section.add_row(
                    "",
                    QLabel("エポック選択の指標：OOF 平均適合率（AP）・最大"),
                    path,
                )
                continue
            if path == "data.input_channels":
                control = QLabel("A（単一チャンネル）")
                control.setToolTip(path)
            elif path == "model.input.normalization.method":
                control = QLabel("画像ごとのパーセンタイル（下位〜上位を 0〜1 に）")
                control.setToolTip(path)
            elif path == "model.bsize":
                control = QLabel("256（cpsam 系は固定）")
                control.setToolTip(path)
            else:
                control = self._control(path, value)
            if path == "model.nimg_per_epoch" and isinstance(control, QLineEdit):
                control.setPlaceholderText("自動（フォールドの学習画像数）")
            if path == "model.bsize":
                control.setEnabled(False)
            control.installEventFilter(self)
            for child in control.findChildren(QWidget):
                child.installEventFilter(self)
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
                    else training_choice_label(path, str(item))
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
        if isinstance(control, QSpinBox | QDoubleSpinBox):
            control.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return control

    def _bind_signals(self) -> None:
        """変更時にプレビューと件数を更新する。"""
        self.model_type.currentIndexChanged.connect(self._switch_model)
        self.study.currentTextChanged.connect(self._on_config_changed)
        self.description_edit.textChanged.connect(self._on_config_changed)
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
        old_defaults = self.ctx.backend.default_experiment_config(old_type)["training"]
        new_defaults = self.ctx.backend.default_experiment_config(model_type)["training"]
        target_saved = self._configs_by_model.get(model_type)
        retained_custom_values = False
        for key in ("learning_rate", "weight_decay", "batch_size"):
            current_value = previous["training"][key]
            if current_value == old_defaults[key]:
                saved_value = (
                    target_saved["training"][key] if target_saved is not None else new_defaults[key]
                )
                self.config["training"][key] = copy.deepcopy(saved_value)
                retained_custom_values |= saved_value != new_defaults[key]
            else:
                self.config["training"][key] = copy.deepcopy(current_value)
                retained_custom_values = True
        if retained_custom_values:
            self.ctx.status.show_message("既定値と異なる値を保持しました")
        self._configs_by_model[model_type] = copy.deepcopy(self.config)
        self._active_model = model_type
        self._clear_validation_results()
        self._build_form()
        self._refresh_yaml()

    def _update_model_stack(self) -> None:
        self.model_stack.setCurrentIndex(0 if self.model_type.currentData() == "mask_rcnn" else 1)
        is_cellpose = self.model_type.currentData() == "cellpose"
        if is_cellpose:
            self.model_stack.setFixedHeight(self._model_fields["cellpose"].sizeHint().height())
        else:
            self.model_stack.setMinimumHeight(0)
            self.model_stack.setMaximumHeight(16_777_215)
        self.model_note.setHidden(not is_cellpose)
        self.model_note.setText("セル確率閾値・フロー閾値は推論設定で管理します。")
        if not is_cellpose:
            self._update_normalization_note()

    def _on_config_changed(self, *_args) -> None:
        self._clear_validation_results()
        self._update_estimate()
        self._update_summary()
        if hasattr(self, "model_normalization_note"):
            self._update_normalization_note()
        self._refresh_yaml()

    def _update_normalization_note(self) -> None:
        """現在選択中の重みに対応する読み取り専用正規化値を表示する。"""
        weights = self._model_widgets["mask_rcnn"]["model.pretrained_weights"].currentData()
        mean, std = normalization_for_weights(weights)
        mean_text = ", ".join(f"{value:.3f}" for value in mean)
        std_text = ", ".join(f"{value:.3f}" for value in std)
        self.model_normalization_note.setText(
            f"事前学習済み重みから自動で決定（平均 {mean_text} / 標準偏差 {std_text}）"
        )

    def eventFilter(self, watched, event) -> bool:
        """フォーカス中の設定に対応する YAML 行を強調する。"""
        if watched is self.scroll and event.type() == QEvent.Type.Resize:
            self._update_form_columns(self.scroll.width())
        if event.type() == QEvent.Type.FocusIn:
            path = self._config_path_for_widget(watched)
            if path:
                self._highlight_yaml_path(path)
        return super().eventFilter(watched, event)

    def _highlight_yaml_path(self, path: str) -> None:
        """設定キーに対応する YAML の行へ CHANGED 色を付ける。"""
        parts = path.split(".")
        parent_keys: list[tuple[int, str]] = []
        block = self.yaml_preview.document().firstBlock()
        while block.isValid():
            line = block.text()
            stripped = line.lstrip()
            if ":" in stripped:
                indent = len(line) - len(stripped)
                key, _separator, _value = stripped.partition(":")
                while parent_keys and parent_keys[-1][0] >= indent:
                    parent_keys.pop()
                current_path = [name for _level, name in parent_keys] + [key.strip()]
                if current_path == parts:
                    selection = QTextEdit.ExtraSelection()
                    selection.cursor = self.yaml_preview.textCursor()
                    selection.cursor.setPosition(block.position())
                    selection.cursor.movePosition(
                        selection.cursor.MoveOperation.EndOfBlock,
                        selection.cursor.MoveMode.KeepAnchor,
                    )
                    selection.format.setBackground(QColor(Color.CHANGED))
                    self.yaml_preview.setExtraSelections([selection])
                    return
                parent_keys.append((indent, key.strip()))
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
            if isinstance(widget, QLabel) and path in {
                "data.input_channels",
                "model.input.normalization.method",
                "model.bsize",
            }:
                continue
            key_path = path.split(".")
            value = self._read_control(path, widget)
            target = result
            for part in key_path[:-1]:
                target = target[part]
            target[key_path[-1]] = value
        if model_type == "mask_rcnn":
            mean, std = normalization_for_weights(result["model"].get("pretrained_weights", "coco"))
            result["model"]["input"]["image_mean"] = mean
            result["model"]["input"]["image_std"] = std
        result.setdefault("checkpoint", {})["best_metric"] = "oof_instance_map"
        result["checkpoint"].pop("best_mode", None)
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
                if not text:
                    return None
                try:
                    return float(text)
                except ValueError:
                    return text
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
                config["data"].get("cv", {}).get("n_folds", 5),
            )
            numeric_family = numeric_font().family()
            self.estimate_label.setText(
                "見込み使用データ数：学習用 "
                f"<span style='font-family:{numeric_family};font-weight:bold'>{train}</span> 件 / "
                f"{config['data'].get('cv', {}).get('n_folds', 5)} 分割・各分割の検証 約 "
                f"<span style='font-family:{numeric_family};font-weight:bold'>{valid}</span> 件"
                f"。交差検証の後、{train} 件すべてで最終学習します。"
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
                display = training_choice_label(path, str(item))
                widget.addItem(display, item)
            index = widget.findData(selected)
            if index < 0:
                default_value = self._lookup_default(path)
                index = widget.findData(default_value)
            widget.setCurrentIndex(max(0, index))
            widget.blockSignals(False)

    def validate_config(self) -> list[dict[str, str]]:
        """Backend 検証結果をダイアログに表示する。"""
        if self._queue_edit_id:
            return []
        self.config = self._collect_config()
        results = self.ctx.backend.validate_experiment_config(self.config)
        self._validation_results = results
        warnings = sum(item["level"] == "warning" for item in results)
        errors = sum(item["level"] == "error" for item in results)
        if warnings or errors:
            parts = []
            if warnings:
                parts.append(f"⚠ 警告 {warnings} 件")
            if errors:
                parts.append(f"⚠ エラー {errors} 件")
            self.validation_result_button.setText("　".join(parts))
            self.validation_result_button.show()
        else:
            self.validation_result_button.hide()
        text = (
            "\n".join(f"{item['level']}: {item['message']}" for item in results)
            or "OK: 設定に問題はありません"
        )
        QMessageBox.information(self, "設定の検証", text)
        return results

    def save_draft(self) -> str:
        """下書きを保存し、実験 ID を保持する。"""
        if self._queue_edit_id:
            return ""
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

    def _save_button_clicked(self) -> None:
        if self._queue_edit_id:
            self._sync_queue_edit_state()
            if self._queue_edit_unavailable:
                return
            self.config = self._collect_config()
            expid = self._queue_edit_id
            try:
                self.ctx.backend.update_training_queue_item(expid, self.config)
            except (KeyError, ValueError) as error:
                QMessageBox.warning(self, "キューに保存できません", str(error))
                self._sync_queue_edit_state()
                return
            self.ctx.status.show_message(f"キューの {expid} を保存しました")
            self._restore_before_queue_edit()
            self.ctx.navigator.navigate(PageId.TRAINING_QUEUE, select=expid)
        else:
            self.save_draft()

    def _cancel_queue_edit(self) -> None:
        if self._queue_edit_id:
            if self._queue_edit_unavailable:
                self._restore_before_queue_edit()
                self.ctx.navigator.navigate(PageId.TRAINING)
                return
            self._restore_before_queue_edit()
            self.ctx.navigator.navigate(PageId.TRAINING_QUEUE)

    def _sync_queue_edit_state(self, *_args) -> None:
        """編集中のキュー行が待機中でなくなったら保存を無効にする。"""
        if not self._queue_edit_id:
            return
        try:
            experiment = self.ctx.backend.get_experiment(self._queue_edit_id)
        except (KeyError, ValueError):
            experiment = None
        if experiment is not None and experiment.status == "queued":
            return
        self._queue_edit_unavailable = True
        if experiment is None:
            message = f"{self._queue_edit_id} は削除されたため、編集を保存できません"
        else:
            message = f"{self._queue_edit_id} は実行を開始したため、編集を保存できません"
        self.queue_edit_banner.setText(message)
        self.save_button.setEnabled(False)
        self.training_actions["save"].setEnabled(False)
        self.cancel_queue_edit_button.setText("閉じる")

    def _restore_before_queue_edit(self) -> None:
        """詳細編集前の学習設定を表示へ戻す。"""
        snapshot = self._pre_queue_edit
        self._queue_edit_id = None
        self._queue_edit_unavailable = False
        self.save_button.setEnabled(True)
        self.training_actions["save"].setEnabled(True)
        self.training_actions["save"].setText("下書き保存")
        self.cancel_queue_edit_button.setText("キャンセル")
        self._set_queue_edit_actions(False)
        if snapshot is not None:
            self.config = copy.deepcopy(snapshot["config"])
            self._edit_id = snapshot["edit_id"]
            self._pre_queue_edit = None
            self._apply_config_to_form(self.config)
            self._configs_by_model = copy.deepcopy(snapshot["configs_by_model"])
            self.experiment_id.setText(self.config["experiment"].get("id") or "")
            self.preview_action.setChecked(snapshot["yaml_visible"])
            self._refresh_yaml()
            self.scroll.verticalScrollBar().setValue(snapshot["scroll_value"])
            for name, state in snapshot["actions"].items():
                action = self.training_actions[name]
                action.setEnabled(state["enabled"])
                action.setToolTip(state["tooltip"])
                action.setStatusTip(state["status_tip"])
            self.training_actions["save"].setText(snapshot["save_action_text"])
            self.save_button.setEnabled(snapshot["save_button_enabled"])
            self.cancel_queue_edit_button.setText(snapshot["cancel_button_text"])
            focus_path = snapshot["focus_path"]
            if focus_path:
                focus_widget = next(
                    (
                        widget
                        for widget in self.fields.values()
                        if self._config_path_for_widget(widget) == focus_path
                    ),
                    None,
                )
                if focus_widget is None:
                    focus_widget = next(
                        (
                            widget
                            for group in self._model_widgets.values()
                            for widget in group.values()
                            if self._config_path_for_widget(widget) == focus_path
                        ),
                        None,
                    )
                if focus_widget is not None:
                    focus_widget.setFocus(Qt.FocusReason.OtherFocusReason)
                    cursor_position = snapshot["cursor_position"]
                    if cursor_position is not None and hasattr(focus_widget, "setCursorPosition"):
                        focus_widget.setCursorPosition(cursor_position)
                    self._highlight_yaml_path(focus_path)
        self.queue_edit_banner.hide()
        self.validate_button.setVisible(True)
        self.queue_button.setVisible(True)
        self.start_button.setVisible(True)
        self.cancel_queue_edit_button.hide()
        if snapshot is None:
            self.refresh_next_identifier()

    def _capture_queue_edit_snapshot(self) -> dict[str, Any]:
        focus_widget = QApplication.focusWidget()
        if focus_widget is not None and not (
            focus_widget is self.form_host or self.form_host.isAncestorOf(focus_widget)
        ):
            focus_widget = None
        cursor_widget = focus_widget
        if cursor_widget is not None and not hasattr(cursor_widget, "cursorPosition"):
            editor = getattr(cursor_widget, "lineEdit", None)
            cursor_widget = editor() if callable(editor) else None
        cursor_position = (
            cursor_widget.cursorPosition()
            if cursor_widget is not None and hasattr(cursor_widget, "cursorPosition")
            else None
        )
        return {
            "config": copy.deepcopy(self._collect_config()),
            "edit_id": self._edit_id,
            "configs_by_model": copy.deepcopy(self._configs_by_model),
            "yaml_visible": self.preview_action.isChecked(),
            "scroll_value": self.scroll.verticalScrollBar().value(),
            "focus_path": self._config_path_for_widget(focus_widget),
            "cursor_position": cursor_position,
            "actions": {
                name: {
                    "enabled": action.isEnabled(),
                    "tooltip": action.toolTip(),
                    "status_tip": action.statusTip(),
                }
                for name, action in self.training_actions.items()
            },
            "save_action_text": self.training_actions["save"].text(),
            "save_button_enabled": self.save_button.isEnabled(),
            "cancel_button_text": self.cancel_queue_edit_button.text(),
        }

    def _apply_config_to_form(self, config: dict[str, Any]) -> None:
        """既存の入力部品を保ち、指定設定をフォームへ反映する。"""
        model_type = config["model"]["type"]
        self.model_type.blockSignals(True)
        self.model_type.setCurrentIndex(max(0, self.model_type.findData(model_type)))
        self.model_type.blockSignals(False)
        self._active_model = model_type
        self._update_model_stack()
        self.study.setCurrentText(config["experiment"].get("study_id", "foam_study"))
        self.description_edit.setText(config["experiment"].get("description", ""))
        self.experiment_id.setText(config["experiment"].get("id", ""))
        widgets = dict(self.fields)
        widgets.update(self._model_widgets.get(model_type, {}))
        for path, widget in widgets.items():
            value: Any = config
            for part in path.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if value is None:
                continue
            widget.blockSignals(True)
            if isinstance(widget, QComboBox):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox | QDoubleSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(
                    ", ".join(map(str, value)) if isinstance(value, list) else str(value)
                )
            widget.blockSignals(False)
        self._clear_validation_results()
        self._update_estimate()
        self._update_summary()
        self._update_normalization_note()
        self._refresh_yaml()

    def _set_queue_edit_actions(self, editing: bool) -> None:
        message = "キューの行を編集中です。『キューに保存』か『キャンセル』で終えてください"
        for name in ("validate", "queue", "start", "new"):
            action = self.training_actions[name]
            action.setEnabled(not editing)
            action.setToolTip(message if editing else "")
            action.setStatusTip(message if editing else "")
        self.reset_defaults_action.setEnabled(not editing)
        self.reset_defaults_action.setToolTip(
            message
            if editing
            else "現在のモデルの既定値に戻します。実験群・説明・データセット版は保持します。"
        )

    def refresh_next_identifier(self) -> None:
        """既存実験やキュー追加と衝突しない次の識別子を表示する。"""
        if self._edit_id or self._queue_edit_id:
            return
        next_id = self.ctx.backend.next_experiment_id()
        if self.experiment_id.text() != next_id:
            self.experiment_id.setText(next_id)
            self.config.setdefault("experiment", {})["id"] = next_id
            self._refresh_yaml()

    def _has_non_draft_id(self, experiment_id: str) -> bool:
        """指定 ID が下書き以外の既存実験か調べる。"""
        for experiment in self.ctx.backend.list_experiments():
            if experiment.experiment_id == experiment_id:
                return experiment.status != "draft"
        return False

    def start_training(self, confirm: bool = True) -> str | None:
        """学習を登録してジョブを開始し、実験一覧へ移る。"""
        if self._queue_edit_id:
            return None
        self.config = self._collect_config()
        results = self.ctx.backend.validate_experiment_config(self.config)
        errors = [item for item in results if item["level"] == "error"]
        if errors:
            QMessageBox.warning(self, "設定エラー", "\n".join(item["message"] for item in errors))
            return None
        controller = self.ctx.queue_controller
        if self.ctx.training_runner.is_busy:
            active_id = self.ctx.training_runner.experiment_id
            answer = QMessageBox.question(
                self,
                "学習中",
                f"学習を実行中です（{active_id}）。この学習をキューの末尾に追加しますか？",
            )
            if answer == QMessageBox.StandardButton.Yes:
                queued_id = self.enqueue_config()
                if queued_id and controller is not None and not controller.executing:
                    controller.start()
                return queued_id
            return None
        if controller is not None and controller.executing:
            answer = QMessageBox.question(
                self, "学習キュー", "キューを実行中です。この設定をキューの末尾に追加しますか？"
            )
            if answer == QMessageBox.StandardButton.Yes:
                return self.enqueue_config()
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
        try:
            self.ctx.training_runner.start(experiment_id)
        except RuntimeError as error:
            QMessageBox.warning(self, "学習を開始できません", str(error))
            return None
        self.ctx.status.show_message(f"{experiment_id} の学習を開始しました")
        self._edit_id = None
        self.config["experiment"]["id"] = self.ctx.backend.next_experiment_id()
        self.experiment_id.setText(self.config["experiment"]["id"])
        self._refresh_yaml()
        self.ctx.navigator.navigate(PageId.EXPERIMENTS, select=experiment_id)
        return experiment_id

    def enqueue_config(self) -> str | None:
        """現在の設定を検証してキュー末尾へ登録する。"""
        if self._queue_edit_id:
            return None
        if self._edit_id:
            QMessageBox.warning(
                self, "キューに追加", "編集中のキュー項目は「キューに保存」で更新してください。"
            )
            return None
        config = self._collect_config()
        issues = self.ctx.backend.validate_experiment_config(config)
        errors = [issue for issue in issues if issue["level"] == "error"]
        if errors:
            QMessageBox.warning(self, "設定エラー", "\n".join(issue["message"] for issue in errors))
            return None
        item = self.ctx.backend.add_training_queue_item(config)
        self.refresh_next_identifier()
        waiting = sum(entry.status == "queued" for entry in self.ctx.backend.list_training_queue())
        self.ctx.status.show_message(
            f"{item.experiment_id} をキューに追加しました（待機 {waiting} 件）"
        )
        return item.experiment_id

    def on_enter(self, params: dict[str, Any]) -> None:
        """複製・下書き編集の設定を読み込む。"""
        self._refresh_options()
        self.normalization_reset_message = ""
        self.normalization_reset_note.clear()
        copy_from = params.get("copy_from")
        edit = params.get("edit")
        queue_edit = params.get("edit_queue")
        if queue_edit:
            # 編集中に別のキュー行を開いたときは、最初のキュー編集前の設定を保つ
            # （編集中だった行の未保存の変更は捨てる）
            if not self._queue_edit_id or self._pre_queue_edit is None:
                self._pre_queue_edit = self._capture_queue_edit_snapshot()
            self._queue_edit_unavailable = False
        elif self._queue_edit_id:
            self._restore_before_queue_edit()
        self._queue_edit_id = queue_edit
        self._set_queue_edit_actions(bool(queue_edit))
        self.save_button.setEnabled(True)
        self.training_actions["save"].setText("キューに保存" if queue_edit else "下書き保存")
        self.cancel_queue_edit_button.setText("キャンセル")
        self.validate_button.setVisible(not bool(queue_edit))
        self.cancel_queue_edit_button.setVisible(bool(queue_edit))
        self.queue_button.setVisible(not bool(queue_edit))
        self.start_button.setVisible(not bool(queue_edit))
        if queue_edit:
            source = self.ctx.backend.get_experiment(queue_edit)
            self.config, migrated = self.ctx.backend.migrate_experiment_config(source.config.values)
            self.config["experiment"]["id"] = queue_edit
            self._edit_id = None
            self._configs_by_model.clear()
            self._apply_config_to_form(self.config)
            banner = getattr(self, "queue_edit_banner", None)
            if banner is None:
                banner = QLabel()
                set_style(banner, role="note")
                self.queue_edit_banner = banner
                self.content_layout.insertWidget(0, banner)
            banner.setText(f"キューの {queue_edit} を編集中")
            banner.show()
            self._refresh_yaml()
            self._sync_queue_edit_state()
            return
        if hasattr(self, "queue_edit_banner"):
            self.queue_edit_banner.hide()
        if not copy_from and not edit:
            return
        source = self.ctx.backend.get_experiment(copy_from or edit)
        self._edit_id = edit
        self.config, migrated = self.ctx.backend.migrate_experiment_config(source.config.values)
        if self.config.get("model", {}).get("type") == "mask_rcnn":
            saved_input = self.config["model"].get("input", {})
            weights = self.config["model"].get("pretrained_weights", "coco")
            expected_mean, expected_std = normalization_for_weights(weights)
            saved_mean, saved_std = saved_input.get("image_mean"), saved_input.get("image_std")
            if saved_mean != expected_mean or saved_std != expected_std:
                self.normalization_reset_message = (
                    "元の実験の値（平均 "
                    f"{', '.join(map(str, saved_mean or []))} / 標準偏差 "
                    f"{', '.join(map(str, saved_std or []))}）から、"
                    "事前学習済み重みの値に置き直しました"
                )
            saved_input["image_mean"] = expected_mean
            saved_input["image_std"] = expected_std
        if migrated:
            migration_notice = "旧形式の設定を現在の形式に移行しました。"
            self.normalization_reset_message = "\n".join(
                filter(None, (self.normalization_reset_message, migration_notice))
            )
        self.config["experiment"]["id"] = edit if edit else self.ctx.backend.next_experiment_id()
        self._configs_by_model.clear()
        self._build_form()
        self.normalization_reset_note.setText(self.normalization_reset_message)
        self._refresh_yaml()

    def refresh_on_activate(self) -> None:
        """前面化時は選択肢だけ更新し、複製時の注記を維持する。"""
        self._refresh_options()
