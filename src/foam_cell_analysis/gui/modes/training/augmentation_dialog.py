"""データ拡張プロファイル編集と numpy プレビュー。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ....services.backend import Backend
from ....services.models import AugmentationProfile, TransformSetting
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
    ("輝度・画質", {"brightness", "contrast", "gamma", "blur", "noise"}),
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
        self.resize(1100, 750)
        self.setMinimumSize(1000, 680)
        root = QVBoxLayout(self)
        body = QHBoxLayout()
        root.addLayout(body, 1)
        self.editor_scroll = QScrollArea()
        self.editor_scroll.setWidgetResizable(True)
        editor = QWidget()
        self.editor_layout = QVBoxLayout(editor)
        self.editor_scroll.setWidget(editor)
        body.addWidget(self.editor_scroll, 3)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        body.addWidget(preview, 2)

        meta = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItems([item.profile_id for item in self.profiles])
        self.profile_combo.setCurrentText(self.source.profile_id)
        self.profile_combo.setToolTip("augmentation.profile")
        self.status_label = QLabel(
            "使用済み・元版は読み取り専用"
            if self.source.used_by_experiments
            else "未使用・保存時は新版を作成"
        )
        self.name = QLineEdit(self.source.name)
        self.name.setToolTip("augmentation.name")
        self.base = QLabel(self.source.base_profile or "—")
        meta.addRow("プロファイル", self.profile_combo)
        meta.addRow("状態", self.status_label)
        meta.addRow("名前", self.name)
        meta.addRow("ベースプロファイル", self.base)
        self.editor_layout.addLayout(meta)
        self.controls: dict[
            str, tuple[QCheckBox, QDoubleSpinBox, QDoubleSpinBox | None, QDoubleSpinBox | None]
        ] = {}
        transforms = {item.key: item for item in self.source.transforms}
        for title, group_keys in TRANSFORM_GROUPS:
            group = QGroupBox(title)
            grid = QGridLayout(group)
            grid.addWidget(QLabel("有効な変換"), 0, 0)
            grid.addWidget(QLabel("適用確率"), 0, 1)
            grid.addWidget(QLabel("範囲（最小 ～ 最大）"), 0, 2, 1, 3)
            row_index = 1
            for key, label, probability, lower, upper in TRANSFORMS:
                if key not in group_keys:
                    continue
                setting = transforms.get(
                    key, TransformSetting(key, label, False, probability, lower, upper)
                )
                enabled = QCheckBox(label)
                enabled.setChecked(setting.enabled)
                enabled.setToolTip(f"{key}.enabled")
                chance = QDoubleSpinBox()
                chance.setRange(0.0, 1.0)
                chance.setSingleStep(0.05)
                chance.setDecimals(2)
                chance.setValue(setting.probability)
                chance.setToolTip(f"{key}.probability")
                minimum = self._range_editor(setting.range_min)
                maximum = self._range_editor(setting.range_max)
                if minimum is not None:
                    minimum.setToolTip(f"{key}.range_min")
                if maximum is not None:
                    maximum.setToolTip(f"{key}.range_max")
                grid.addWidget(enabled, row_index, 0)
                grid.addWidget(chance, row_index, 1)
                if minimum is not None and maximum is not None:
                    grid.addWidget(minimum, row_index, 2)
                    grid.addWidget(QLabel("〜"), row_index, 3, Qt.AlignmentFlag.AlignCenter)
                    grid.addWidget(maximum, row_index, 4)
                self.controls[key] = (enabled, chance, minimum, maximum)
                row_index += 1
            self.editor_layout.addWidget(group)

        self.order = QListWidget()
        known = [key for key in self.source.order if key in self.controls]
        known.extend(key for key, *_ in TRANSFORMS if key not in known)
        labels = dict((key, label) for key, label, *_ in TRANSFORMS)
        for key in known:
            item = QListWidgetItem(labels[key])
            item.setToolTip(key)
            self.order.addItem(item)
        self._order_keys = known
        order_buttons = QHBoxLayout()
        self.up_button = QPushButton("上へ")
        self.down_button = QPushButton("下へ")
        self.up_button.setToolTip("選択した変換を一つ上へ移動")
        self.down_button.setToolTip("選択した変換を一つ下へ移動")
        self.up_button.clicked.connect(lambda: self._move_order(-1))
        self.down_button.clicked.connect(lambda: self._move_order(1))
        order_buttons.addWidget(self.up_button)
        order_buttons.addWidget(self.down_button)
        self.editor_layout.addWidget(QLabel("適用順序"))
        self.editor_layout.addWidget(self.order, 1)
        self.editor_layout.addLayout(order_buttons)

        selectors = QFormLayout()
        self.dataset = QComboBox()
        datasets = sorted(
            (item for item in backend.list_dataset_versions() if item.purpose == "train"),
            key=lambda item: int(item.version[-3:]),
        )
        self.dataset.addItems([item.version for item in datasets])
        if datasets:
            self.dataset.setCurrentText(datasets[-1].version)
        self.classification = QComboBox()
        self.classification.addItems(["すべて", "分類A", "分類B", "分類C"])
        self.sample = QComboBox()
        self.items = backend.get_working_dataset("train").items
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
        for index, title in enumerate(("元画像", "拡張後画像", "拡張後マスク", "オーバーレイ")):
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
        preview_layout.addWidget(self.preview_stack, 1)
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
        self.order.model().rowsMoved.connect(self.refresh_preview)
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
        return widget

    def _move_order(self, offset: int) -> None:
        row = self.order.currentRow()
        target = row + offset
        if row < 0 or not 0 <= target < self.order.count():
            return
        item = self.order.takeItem(row)
        self.order.insertItem(target, item)
        self.order.setCurrentRow(target)
        self._order_keys.insert(target, self._order_keys.pop(row))
        self.refresh_preview()

    def select_random_sample(self) -> None:
        """現在の分類からランダム画像を選ぶ。"""
        if self.sample.count():
            index = int(np.random.default_rng().integers(0, self.sample.count()))
            self.sample.setCurrentIndex(index)

    def _update_sample_options(self, *_args) -> None:
        """データセット・分類条件に合う画像を選択肢へ反映する。"""
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
        ordered = [
            self._order_keys[self.order.row(self.order.item(row))]
            for row in range(self.order.count())
        ]
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
        labels = {key: label for key, label, *_ in TRANSFORMS}
        known = [key for key in self.source.order if key in self.controls]
        known.extend(key for key, *_ in TRANSFORMS if key not in known)
        self._order_keys = known
        self.order.clear()
        for key in known:
            item = QListWidgetItem(labels[key])
            item.setToolTip(key)
            self.order.addItem(item)
        self.refresh_preview()

    def refresh_preview(self) -> None:
        """numpy で軽量変換を適用してプレビューを描き直す。"""
        self._update_order_colors()
        selected_id = self.sample.currentText()
        selected = next((item for item in self.items if item.item_id == selected_id), self.items[0])
        source = self.backend.get_item_image("train", selected.item_id, selected.channels[0])
        source_mask = self.backend.get_item_mask(
            "train", selected.item_id, selected.selected_mask_revision
        )
        self._preview_images = []
        self._preview_masks = []
        count = 8 if self.eight.isChecked() else 1
        for index in range(count):
            rng = np.random.default_rng(selected.seed + index + self._preview_seed())
            transformed = source.copy()
            transformed_mask = source_mask.copy()
            for key in self._order_keys:
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
            self.preview_labels["元画像"].setPixmap(_gray_pixmap(source, 190))
            self.preview_labels["拡張後画像"].setPixmap(_gray_pixmap(transformed, 190))
            self.preview_labels["拡張後マスク"].setPixmap(
                _rgb_pixmap(self._colorize(transformed_mask), 190)
            )
            self.preview_labels["オーバーレイ"].setPixmap(_rgb_pixmap(overlay, 190))
            self.preview_stack.setCurrentWidget(self.single_preview)
        else:
            for index, array in enumerate(self._preview_images):
                self.pattern_labels[index].setPixmap(_gray_pixmap(array, 100))
            self.preview_stack.setCurrentWidget(self.eight_preview)

    def _update_order_colors(self) -> None:
        """無効な変換を適用順序リストで灰色にする。"""
        for row, key in enumerate(self._order_keys):
            enabled = self.controls[key][0].isChecked()
            self.order.item(row).setForeground(QColor("#222" if enabled else "#999"))

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
