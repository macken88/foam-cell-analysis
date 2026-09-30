"""データ拡張プロファイル編集と numpy プレビュー。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ....services.backend import Backend
from ....services.models import AugmentationProfile, TransformSetting
from ...theme import Color
from ...widgets.table import mark_primary

TRANSFORMS = [
    ("horizontal_flip", "左右反転", 0.5, None, None),
    ("vertical_flip", "上下反転", 0.5, None, None),
    ("rotation", "回転", 0.5, -180.0, 180.0),
    ("scale", "拡大縮小", 0.3, 0.8, 1.2),
    ("translation", "平行移動", 0.0, -0.1, 0.1),
    ("crop", "切り出し", 0.0, 0.5, 1.0),
    ("elastic", "弾性変形", 0.0, 0.0, 1.0),
    ("brightness", "明るさ", 0.3, -10.0, 10.0),
    ("contrast", "コントラスト", 0.3, 0.9, 1.1),
    ("gamma", "ガンマ", 0.0, 0.7, 1.5),
    ("blur", "ぼかし", 0.2, 0.0, 1.0),
    ("noise", "ノイズ", 0.2, 0.0, 12.0),
    ("channel_dropout", "チャンネル欠落", 0.0, 0.0, 1.0),
    ("channel_intensity", "チャンネル強度変動", 0.0, 0.8, 1.2),
]
TRANSFORM_GROUPS = [
    (
        "幾何変換",
        {"horizontal_flip", "vertical_flip", "rotation", "scale", "translation", "crop", "elastic"},
    ),
    ("輝度", {"brightness", "contrast", "gamma"}),
    ("画質", {"blur", "noise"}),
    ("チャンネル", {"channel_dropout", "channel_intensity"}),
]


def _gray_pixmap(array: np.ndarray, size: int = 220) -> QPixmap:
    """2次元 uint8 画像を縮小表示用 pixmap にする。"""
    pixels = np.ascontiguousarray(array.astype(np.uint8))
    image = QImage(
        pixels.data,
        pixels.shape[1],
        pixels.shape[0],
        pixels.strides[0],
        QImage.Format.Format_Grayscale8,
    )
    return QPixmap.fromImage(image.copy()).scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


def _rgb_pixmap(array: np.ndarray, size: int = 220) -> QPixmap:
    """RGB numpy 画像を縮小表示用 pixmap にする。"""
    pixels = np.ascontiguousarray(array.astype(np.uint8))
    image = QImage(
        pixels.data,
        pixels.shape[1],
        pixels.shape[0],
        pixels.strides[0],
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(image.copy()).scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


class AugmentationDialog(QDialog):
    """編集結果は常に新しい版として保存するダイアログ。"""

    def __init__(
        self,
        backend: Backend,
        profile_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.backend = backend
        self.profiles = backend.list_augmentation_profiles()
        self.source = backend.get_augmentation_profile(profile_id or self.profiles[-1].profile_id)
        self.setWindowTitle("データ拡張プロファイル")
        self.resize(1360, 800)
        self.setMinimumSize(1250, 720)
        root = QVBoxLayout(self)
        menu_bar = QMenuBar(self)
        edit_menu = menu_bar.addMenu("編集")
        reset_action = QAction("既定値に戻す", self)
        reset_action.setToolTip("拡張設定を aug_v001 の初期値に戻します。保存は行いません。")
        reset_action.triggered.connect(self.reset_profile_defaults)
        edit_menu.addAction(reset_action)
        root.setMenuBar(menu_bar)
        root.addWidget(QLabel("簡易プレビュー（学習時の変換とは一致しません）"))
        body = QSplitter(Qt.Orientation.Horizontal)
        self.preview_splitter = body
        root.addWidget(body, 1)
        self.editor_scroll = QScrollArea()
        self.editor_scroll.setWidgetResizable(True)
        editor = QWidget()
        self.editor_layout = QVBoxLayout(editor)
        self.editor_scroll.setWidget(editor)
        body.addWidget(self.editor_scroll)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        body.addWidget(preview)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)
        body.setCollapsible(0, False)
        body.setCollapsible(1, False)
        self.editor_scroll.setMinimumWidth(680)
        preview.setMinimumWidth(380)
        body.setSizes([680, 680])

        meta = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItems([item.profile_id for item in self.profiles])
        self.profile_combo.setCurrentText(self.source.profile_id)
        self.status_label = QLabel(
            "使用済み・元版は読み取り専用"
            if self.source.used_by_experiments
            else "未使用・保存時は新版を作成"
        )
        self.name = QLineEdit(self.source.name)
        self.base = QLabel(self.source.base_profile or "—")
        meta.addRow("プロファイル", self.profile_combo)
        meta.addRow("状態", self.status_label)
        meta.addRow("名前", self.name)
        meta.addRow("ベースプロファイル", self.base)
        self.editor_layout.addLayout(meta)
        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        self.reset_button = QPushButton("既定値に戻す")
        self.reset_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.reset_button.clicked.connect(self.reset_profile_defaults)
        reset_row.addWidget(self.reset_button)
        self.editor_layout.addLayout(reset_row)
        self.controls: dict[
            str, tuple[QCheckBox, QDoubleSpinBox, QDoubleSpinBox | None, QDoubleSpinBox | None]
        ] = {}
        self._labels_to_keys = {label: key for key, label, *_ in TRANSFORMS}
        self.order_controls: dict[str, QSpinBox] = {}
        transforms = {item.key: item for item in self.source.transforms}
        all_keys = [key for key, *_ in TRANSFORMS]
        known = [key for key in self.source.order if key in all_keys]
        known.extend(key for key in all_keys if key not in known)
        order_index = {key: index + 1 for index, key in enumerate(known)}
        self.order_table = QTableWidget()
        self.order_table.setColumnCount(5)
        self.order_table.setHorizontalHeaderLabels(["適用", "順", "変換", "確率", "範囲"])
        self.order_table.verticalHeader().hide()
        self.order_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.order_table.setColumnWidth(0, 54)
        self.order_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.order_table.setColumnWidth(2, 160)
        self.order_table.setColumnWidth(3, 100)
        self.order_table.setColumnWidth(4, 280)
        self.order_table.setMinimumWidth(670)
        self.order_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        table_row = 0
        self._order_keys = []
        for title, group_keys in TRANSFORM_GROUPS:
            category_row = table_row
            table_row += 1
            self.order_table.insertRow(category_row)
            category_item = QTableWidgetItem(title)
            category_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.order_table.setItem(category_row, 0, category_item)
            self.order_table.setSpan(category_row, 0, 1, 5)
            for key, label, probability, lower, upper in TRANSFORMS:
                if key not in group_keys:
                    continue
                setting = transforms.get(
                    key, TransformSetting(key, label, False, probability, lower, upper)
                )
                self.order_table.insertRow(table_row)
                self._order_keys.append(key)
                enabled = QCheckBox()
                enabled.setChecked(setting.enabled)
                chance = QDoubleSpinBox()
                chance.setRange(0.0, 1.0)
                chance.setSingleStep(0.05)
                chance.setDecimals(2)
                chance.setValue(setting.probability)
                minimum = self._range_editor(setting.range_min)
                maximum = self._range_editor(setting.range_max)
                range_widget = QWidget()
                range_layout = QHBoxLayout(range_widget)
                range_layout.setContentsMargins(0, 0, 0, 0)
                if minimum is not None and maximum is not None:
                    unit = {
                        "rotation": "角度（度）",
                        "scale": "倍率",
                        "translation": "画像寸法に対する比率",
                        "crop": "画像寸法に対する割合",
                        "elastic": "変形強度係数",
                        "brightness": "明るさ変化（%）",
                        "contrast": "コントラスト倍率",
                        "gamma": "ガンマ値",
                        "blur": "ぼかし sigma（pixel）",
                        "noise": "ノイズ標準偏差（8 bit 濃度値）",
                        "channel_dropout": "欠落比率",
                        "channel_intensity": "チャンネル強度倍率",
                    }.get(key, "値")
                    minimum.setToolTip(unit)
                    maximum.setToolTip(unit)
                    range_layout.addWidget(minimum)
                    range_layout.addWidget(QLabel("〜"))
                    range_layout.addWidget(maximum)
                order = QSpinBox()
                order.setRange(0, len(TRANSFORMS))
                order.setSpecialValueText("\u00a0")
                order.setValue(order_index[key] if setting.enabled else 0)
                order.setEnabled(setting.enabled)
                order.valueChanged.connect(
                    lambda value, transform=key: self._set_order(transform, value)
                )
                self.order_controls[key] = order
                self.controls[key] = (enabled, chance, minimum, maximum)
                enabled.toggled.connect(lambda _checked: self._normalize_order())
                self.order_table.setCellWidget(table_row, 0, enabled)
                self.order_table.setCellWidget(table_row, 1, order)
                name_item = QTableWidgetItem(label)
                self.order_table.setItem(table_row, 2, name_item)
                self.order_table.setCellWidget(table_row, 3, chance)
                self.order_table.setCellWidget(table_row, 4, range_widget)
                table_row += 1
        self.editor_layout.addWidget(self.order_table, 1)

        selectors = QFormLayout()
        self.dataset = QComboBox()
        datasets = sorted(backend.list_dataset_versions("train"), key=lambda item: item.version)
        self.dataset.addItems([item.version for item in datasets])
        if datasets:
            self.dataset.setCurrentText(datasets[-1].version)
        self.classification = QComboBox()
        self.classification.addItems(["すべて", "分類A", "分類B", "分類C"])
        self.sample = QComboBox()
        self.items = (
            backend.get_dataset_version_items(self.dataset.currentText())
            if self.dataset.currentText()
            else []
        )
        self.sample.addItems([item.item_id for item in self.items])
        selectors.addRow("データセット", self.dataset)
        selectors.addRow("画像分類", self.classification)
        selectors.addRow("画像", self.sample)
        preview_layout.addLayout(selectors)
        preview_buttons = QHBoxLayout()
        self.random_sample = QPushButton("ランダム画像")
        self.regenerate = QPushButton("再生成")
        self.eight = QCheckBox("8パターン表示")
        self.random_sample.clicked.connect(self.select_random_sample)
        self.regenerate.clicked.connect(self.refresh_preview)
        self.eight.toggled.connect(self.refresh_preview)
        preview_buttons.addWidget(self.random_sample)
        preview_buttons.addWidget(self.regenerate)
        preview_buttons.addWidget(QLabel("表示形式:"))
        preview_buttons.addWidget(self.eight)
        preview_layout.addLayout(preview_buttons)
        self.preview_stack = QStackedWidget()
        self.single_preview = QWidget()
        self.preview_grid = QGridLayout(self.single_preview)
        self.preview_labels: dict[str, QLabel] = {}
        for index, title in enumerate(
            ("元画像", "拡張後画像", "拡張後の正解ラベル画像", "オーバーレイ")
        ):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label = QLabel("画像なし")
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setMinimumSize(150, 150)
            cell_layout.addWidget(title_label)
            cell_layout.addWidget(image_label, 1)
            self.preview_labels[title] = image_label
            self.preview_grid.addWidget(cell, index // 2, index % 2)
        self.eight_preview = QWidget()
        self.eight_grid = QGridLayout(self.eight_preview)
        self.pattern_labels: list[QLabel] = []
        for index in range(8):
            label = QLabel(f"パターン {index + 1}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(90, 105)
            self.pattern_labels.append(label)
            self.eight_grid.addWidget(label, index // 4, index % 4)
        self.preview_stack.addWidget(self.single_preview)
        self.preview_stack.addWidget(self.eight_preview)
        self._preview_resize_timer = QTimer(self)
        self._preview_resize_timer.setSingleShot(True)
        self._preview_resize_timer.setInterval(80)
        self._preview_resize_timer.timeout.connect(self.refresh_preview)
        self._normalize_order()
        for label in (*self.preview_labels.values(), *self.pattern_labels):
            # 表示中の画像の大きさで枠が広がると、再描画のたびに拡大が続いて切れてしまう
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            label.installEventFilter(self)
        preview_layout.addWidget(self.preview_stack, 1)
        body.setSizes([700, 600])
        self.refresh_preview()
        self.profile_combo.currentTextChanged.connect(self._load_profile)
        self.sample.currentTextChanged.connect(self.refresh_preview)
        for enabled, probability, minimum, maximum in self.controls.values():
            enabled.toggled.connect(self.refresh_preview)
            probability.valueChanged.connect(self.refresh_preview)
            if minimum is not None:
                minimum.valueChanged.connect(self.refresh_preview)
            if maximum is not None:
                maximum.valueChanged.connect(self.refresh_preview)
        self.classification.currentTextChanged.connect(self._update_sample_options)
        self.dataset.currentTextChanged.connect(self._update_sample_options)
        self._update_sample_options()
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("新しい版として保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        mark_primary(buttons.button(QDialogButtonBox.StandardButton.Save))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _range_editor(value: float | None) -> QDoubleSpinBox | None:
        if value is None:
            return None
        widget = QDoubleSpinBox()
        widget.setRange(-10000.0, 10000.0)
        widget.setDecimals(3)
        widget.setValue(value)
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return widget

    def _set_order(self, key: str, value: int) -> None:
        """順序の重複を後続番号へ押し出す。"""
        ordered = [name for name in self._order_keys if self.controls[name][0].isChecked()]
        ordered.sort(key=lambda name: self.order_controls[name].value())
        if key not in ordered:
            return
        ordered.remove(key)
        ordered.insert(max(0, min(value - 1, len(ordered))), key)
        for position, name in enumerate(ordered, start=1):
            control = self.order_controls[name]
            control.blockSignals(True)
            control.setValue(position)
            control.blockSignals(False)
        self._order_keys = ordered + [name for name in self._order_keys if name not in ordered]
        self.refresh_preview()

    def _normalize_order(self) -> None:
        """有効・無効の区分を保ち、画面上の番号を 1..N に詰める。"""
        ordered = [name for name in self._order_keys if self.controls[name][0].isChecked()]
        ordered.sort(
            key=lambda name: (
                self.order_controls[name].value() or len(TRANSFORMS) + self._order_keys.index(name)
            )
        )
        for name in self._order_keys:
            control = self.order_controls[name]
            control.setEnabled(name in ordered)
            if name not in ordered:
                control.blockSignals(True)
                control.setValue(0)
                control.blockSignals(False)
        for position, name in enumerate(ordered, start=1):
            control = self.order_controls[name]
            control.blockSignals(True)
            control.setValue(position)
            control.blockSignals(False)
        self._order_keys = ordered + [name for name in self._order_keys if name not in ordered]
        self.order_table.resizeColumnToContents(1)
        self._preview_resize_timer.start()

    def select_random_sample(self) -> None:
        """現在の分類からランダム画像を選ぶ。"""
        if self.sample.count():
            index = int(np.random.default_rng().integers(0, self.sample.count()))
            self.sample.setCurrentIndex(index)

    def _update_sample_options(self, *_args) -> None:
        """データセット・分類条件に合う画像を選択肢へ反映する。"""
        version = self.dataset.currentText()
        self.items = self.backend.get_dataset_version_items(version) if version else []
        classifications = sorted(
            {item.classification for item in self.items if item.classification}
        )
        selected_classification = self.classification.currentText()
        self.classification.blockSignals(True)
        self.classification.clear()
        self.classification.addItems(["すべて", *classifications])
        if selected_classification in classifications:
            self.classification.setCurrentText(selected_classification)
        self.classification.blockSignals(False)
        classification = self.classification.currentText()
        matching = [
            item
            for item in self.items
            if classification == "すべて" or item.classification == classification
        ]
        if not matching:
            matching = self.items
        current = self.sample.currentText()
        self.sample.blockSignals(True)
        self.sample.clear()
        self.sample.addItems([item.item_id for item in matching])
        if current in [item.item_id for item in matching]:
            self.sample.setCurrentText(current)
        self.sample.blockSignals(False)
        self.refresh_preview()

    def build_profile(self) -> AugmentationProfile:
        """画面の値から元版を変更せずに新規保存用データを作る。"""
        settings = []
        labels = dict((key, label) for key, label, *_ in TRANSFORMS)
        for key, (enabled, chance, minimum, maximum) in self.controls.items():
            settings.append(
                TransformSetting(
                    key,
                    labels[key],
                    enabled.isChecked(),
                    chance.value(),
                    minimum.value() if minimum else None,
                    maximum.value() if maximum else None,
                )
            )
        ordered = sorted(
            self._order_keys,
            key=lambda key: (
                not self.controls[key][0].isChecked(),
                self.order_controls[key].value(),
            ),
        )
        next_number = max(int(item.profile_id[-3:]) for item in self.profiles) + 1
        default_name = f"foam_cell_aug_v{next_number:03d}"
        return AugmentationProfile(
            "", self.name.text().strip() or default_name, self.source.profile_id, settings, ordered
        )

    def _load_profile(self, profile_id: str) -> None:
        """選択したプロファイルを編集欄へ読み込む。"""
        if not profile_id:
            return
        self.source = self.backend.get_augmentation_profile(profile_id)
        self.name.setText(self.source.name)
        self.base.setText(self.source.base_profile or "—")
        self.status_label.setText(
            "使用済み・元版は読み取り専用"
            if self.source.used_by_experiments
            else "未使用・保存時は新版を作成"
        )
        settings = {setting.key: setting for setting in self.source.transforms}
        for key, label, probability, lower, upper in TRANSFORMS:
            setting = settings.get(
                key, TransformSetting(key, label, False, probability, lower, upper)
            )
            enabled, chance, minimum, maximum = self.controls[key]
            enabled.setChecked(setting.enabled)
            chance.setValue(setting.probability)
            if minimum is not None and setting.range_min is not None:
                minimum.setValue(setting.range_min)
            if maximum is not None and setting.range_max is not None:
                maximum.setValue(setting.range_max)
        all_keys = [key for key, *_ in TRANSFORMS]
        known = [key for key in self.source.order if key in all_keys]
        known.extend(key for key in all_keys if key not in known)
        self._order_keys = known
        order_values = {key: index + 1 for index, key in enumerate(known)}
        for key, control in self.order_controls.items():
            control.blockSignals(True)
            control.setValue(order_values[key] if self.controls[key][0].isChecked() else 0)
            control.setEnabled(self.controls[key][0].isChecked())
            control.blockSignals(False)
        self._normalize_order()
        self.refresh_preview()

    def reset_profile_defaults(self) -> None:
        """確認後、aug_v001 相当の値を編集画面に読み込む。"""
        if self.profile_combo.findText("aug_v001") < 0:
            return
        if (
            QMessageBox.question(self, "既定値に戻す", "拡張設定を aug_v001 の初期値に戻しますか？")
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.profile_combo.setCurrentText("aug_v001")

    def refresh_preview(self) -> None:
        """numpy で軽量変換を適用してプレビューを描き直す。"""
        self._update_order_colors()
        if not self.items or not self.sample.currentText() or not self.dataset.currentText():
            self._preview_images = []
            self._preview_masks = []
            for label in self.preview_labels.values():
                label.setPixmap(QPixmap())
                label.setText("学習画像がありません")
            return
        selected_id = self.sample.currentText()
        selected = next((item for item in self.items if item.item_id == selected_id), self.items[0])
        version = self.dataset.currentText()
        channel = selected.channels[0] if selected.channels else None
        source = self.backend.get_dataset_item_image(version, selected.item_id, channel)
        source_mask = self.backend.get_dataset_item_mask(
            version, selected.item_id, selected.selected_mask_revision
        )
        self._preview_images = []
        self._preview_masks = []
        count = 8 if self.eight.isChecked() else 1
        for index in range(count):
            rng = np.random.default_rng(selected.seed + index + self._preview_seed())
            transformed = source.copy()
            transformed_mask = source_mask.copy()
            for key in sorted(
                self._order_keys,
                key=lambda item: (
                    not self.controls[item][0].isChecked(),
                    self.order_controls[item].value(),
                ),
            ):
                enabled, probability, minimum, maximum = self.controls[key]
                if not enabled.isChecked() or rng.random() > probability.value():
                    continue
                if key == "horizontal_flip":
                    transformed = transformed[:, ::-1]
                    transformed_mask = transformed_mask[:, ::-1]
                elif key == "vertical_flip":
                    transformed = transformed[::-1, :]
                    transformed_mask = transformed_mask[::-1, :]
                elif key == "rotation":
                    turns = int(rng.integers(0, 4))
                    transformed = np.rot90(transformed, turns)
                    transformed_mask = np.rot90(transformed_mask, turns)
                elif key == "brightness":
                    amount = (minimum.value() if minimum else -10) + rng.random() * (
                        (maximum.value() if maximum else 10) - (minimum.value() if minimum else -10)
                    )
                    transformed = np.clip(
                        transformed.astype(float) * (1 + amount / 100), 0, 255
                    ).astype(np.uint8)
                elif key == "contrast":
                    factor = minimum.value() + rng.random() * (maximum.value() - minimum.value())
                    transformed = np.clip(
                        (transformed.astype(float) - 127.5) * factor + 127.5, 0, 255
                    ).astype(np.uint8)
                elif key == "noise":
                    scale = maximum.value() if maximum else 12.0
                    transformed = np.clip(
                        transformed.astype(float) + rng.normal(0, scale, transformed.shape), 0, 255
                    ).astype(np.uint8)
                elif key == "blur":
                    transformed = (
                        (
                            transformed.astype(float)
                            + np.roll(transformed, 1, 0)
                            + np.roll(transformed, -1, 0)
                        )
                        / 3
                    ).astype(np.uint8)
            self._preview_images.append(transformed.copy())
            self._preview_masks.append(transformed_mask.copy())
        if count == 1:
            transformed = self._preview_images[0]
            transformed_mask = self._preview_masks[0]
            overlay = self._overlay(transformed, transformed_mask)
            source_size = self._preview_size(self.preview_labels["元画像"])
            transformed_size = self._preview_size(self.preview_labels["拡張後画像"])
            mask_size = self._preview_size(self.preview_labels["拡張後の正解ラベル画像"])
            overlay_size = self._preview_size(self.preview_labels["オーバーレイ"])
            self.preview_labels["元画像"].setPixmap(_gray_pixmap(source, source_size))
            self.preview_labels["拡張後画像"].setPixmap(_gray_pixmap(transformed, transformed_size))
            self.preview_labels["拡張後の正解ラベル画像"].setPixmap(
                _rgb_pixmap(self._colorize(transformed_mask), mask_size)
            )
            self.preview_labels["オーバーレイ"].setPixmap(_rgb_pixmap(overlay, overlay_size))
            self.preview_stack.setCurrentWidget(self.single_preview)
        else:
            for index, array in enumerate(self._preview_images):
                self.pattern_labels[index].setPixmap(
                    _gray_pixmap(array, self._preview_size(self.pattern_labels[index]))
                )
            self.preview_stack.setCurrentWidget(self.eight_preview)

    @staticmethod
    def _preview_size(label: QLabel) -> int:
        """ラベルの現在の表示領域内に収まる正方形サイズを返す。"""
        return max(48, min(label.width(), label.height()) - 12)

    def _update_order_colors(self) -> None:
        """無効な変換を適用順序リストで灰色にする。"""
        for row in range(self.order_table.rowCount()):
            item = self.order_table.item(row, 2)
            if item is not None:
                key = self._labels_to_keys[item.text()]
                enabled = self.controls[key][0].isChecked()
                item.setForeground(QColor(Color.GRAPHITE if enabled else Color.IDLE))

    @staticmethod
    def _colorize(mask: np.ndarray) -> np.ndarray:
        """インスタンス番号ごとに決定的な色を割り当てる。"""
        result = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for label in np.unique(mask):
            if label == 0:
                continue
            rng = np.random.default_rng(int(label))
            result[mask == label] = rng.integers(45, 240, size=3, dtype=np.uint8)
        return result

    @classmethod
    def _overlay(cls, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """原画像と色付きインスタンスマスクを合成する。"""
        gray = np.repeat(image[:, :, None], 3, axis=2).astype(np.float32)
        color = cls._colorize(mask).astype(np.float32)
        selected = mask > 0
        gray[selected] = 0.55 * gray[selected] + 0.45 * color[selected]
        return np.clip(gray, 0, 255).astype(np.uint8)

    def _preview_seed(self) -> int:
        """再生成ごとに変わる乱数種を返す。"""
        self._preview_counter = getattr(self, "_preview_counter", 0) + 1
        return self._preview_counter

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_preview_images"):
            self._preview_resize_timer.start()

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Resize and watched in (
            *self.preview_labels.values(),
            *self.pattern_labels,
        ):
            self._preview_resize_timer.start()
        return super().eventFilter(watched, event)
