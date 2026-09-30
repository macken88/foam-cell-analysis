"""複数候補の抽出結果を同期表示する画面。"""

import logging
from collections import OrderedDict

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...context import DEFAULT_CHANNEL
from ...labels import model_type_label
from ...navigation import PageId
from ...theme import body_font
from ...widgets.image_convert import DisplayMode, array_to_pixmap, render
from ...widgets.image_view import ImageView, ViewSynchronizer
from ...widgets.marks import DisplayToggle
from ...widgets.page_base import BasePage

logger = logging.getLogger(__name__)

ALL_CLASSIFICATIONS = "すべて"
NO_CANDIDATES_TEXT = "候補一覧で比較する候補を選んでください"
# 表示のキャッシュに残す件数（原画像・描画済み画像）
_IMAGE_CACHE_SIZE = 8
_PIXMAP_CACHE_SIZE = 48


def comparison_block_reason(backend, validation_version: str | None, candidate_ids) -> str:
    """抽出結果比較を開けない理由を返す。開けるときは空文字（比較・評価設計 11 章）。"""
    if not candidate_ids:
        return NO_CANDIDATES_TEXT
    if not validation_version:
        return "確定済みの検証用データセットがありません"
    reasons = []
    for candidate_id in candidate_ids:
        try:
            record = backend.get_candidate_evaluation(candidate_id, validation_version)
        except (KeyError, ValueError, OSError):
            logger.exception("採用する評価を読めません: %s", candidate_id)
            reasons.append(f"{candidate_id} の評価結果を読み込めません")
            continue
        if record is None:
            reasons.append(f"{candidate_id} はこの検証用データセットで未評価です")
        elif record.broken or record.status != "completed":
            reasons.append(f"{candidate_id} の評価結果が壊れています。評価をやり直してください")
    return "\n".join(reasons)


class MaskComparisonPage(BasePage):
    """原画像と候補予測を最大4候補まで比較する。"""

    def __init__(self, ctx, parent=None, *, show_heading: bool = True) -> None:
        super().__init__(
            ctx,
            "抽出結果比較",
            "候補モデルの予測結果を比較します。",
            parent,
            show_heading=show_heading,
        )
        self.validation: str | None = None
        self.candidate_ids: list[str] = []
        # 画面を開いた時点で固定した、候補ごとの評価 ID（7.7）
        self.evaluation_ids: dict[str, str] = {}
        self.block_reason = NO_CANDIDATES_TEXT
        self.items = []
        self.index = 0
        self._model_names: dict[str, str] = {}
        self._image_cache: OrderedDict = OrderedDict()
        self._pixmap_cache: OrderedDict = OrderedDict()
        self._render_signature = None
        self.shortcuts = ctx.shortcuts
        self.shortcuts.changed.connect(self._shortcuts_changed)
        self.ctx.display.changed.connect(self._display_changed)
        self.back = QPushButton("← 候補一覧へ戻る")
        self.validation_label = QLabel("検証用データセット: —")
        self.classification = QComboBox()
        self.classification.addItem(ALL_CLASSIFICATIONS)
        self.item_select = QComboBox()
        controls = QHBoxLayout()
        controls.addWidget(self.back)
        controls.addWidget(self.validation_label)
        controls.addWidget(QLabel("画像分類:"))
        self.classification.setMaximumWidth(130)
        controls.addWidget(self.classification)
        controls.addWidget(QLabel("対象画像:"))
        controls.addWidget(self.item_select, 1)
        self.display_toggle = DisplayToggle(ctx.display, label_scope="prediction")
        self.display_toggle.alternate_selected.connect(self._toggle_display)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.display_toggle)
        mode_row.addStretch(1)
        self.previous = QPushButton("← 前の画像")
        self.next = QPushButton("次の画像 →")
        self.fit = QPushButton("全体表示")
        self.position = QLabel()
        mode_row.addWidget(self.previous)
        mode_row.addWidget(self.next)
        mode_row.addWidget(self.fit)
        mode_row.addWidget(self.position)
        self.previous.setToolTip(f"前の画像（{self.shortcuts['previous_image']}）")
        self.next.setToolTip(f"次の画像（{self.shortcuts['next_image']}）")
        self.view_actions = {}
        for key, label, callback in (
            ("display_mode", "原画像と切り替える", self._cycle_display_mode),
            ("previous_image", "前の画像", lambda: self._move(-1)),
            ("next_image", "次の画像", lambda: self._move(1)),
            ("fit_view", "全体表示", self._fit_all),
        ):
            action = QAction(self._menu_text(label, key), self)
            action.triggered.connect(callback)
            self.view_actions[key] = action
        self.grid = QGridLayout()
        self.views: list[ImageView] = []
        area = QWidget()
        area.setLayout(self.grid)
        self.placeholder = QLabel(NO_CANDIDATES_TEXT)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setFont(body_font(12))
        self.content_layout.addLayout(controls)
        self.content_layout.addLayout(mode_row)
        self.content_layout.addWidget(self.placeholder)
        self.content_layout.addWidget(area, 1)
        self.area = area
        self.area.installEventFilter(self)
        self.back.clicked.connect(lambda: ctx.navigator.navigate(PageId.CANDIDATES))
        self.previous.clicked.connect(lambda: self._move(-1))
        self.next.clicked.connect(lambda: self._move(1))
        self.fit.clicked.connect(self._fit_all)
        self.classification.currentTextChanged.connect(self._load_items)
        self.item_select.currentIndexChanged.connect(self._select_item)

    def _menu_text(self, label: str, key: str) -> str:
        return f"{label}\t{self.shortcuts.display_key(self.shortcuts[key])}"

    def menu_actions(self):
        return {"view": list(self.view_actions.values())}

    def on_enter(self, params: dict) -> None:
        """候補ごとに採用する評価を固定してから画面を組み立てる。"""
        self.candidate_ids = list(params.get("candidate_ids", []))
        supplied_version = params.get("validation_version")
        if supplied_version:
            self.validation = supplied_version
        elif self.candidate_ids:
            versions = {
                self.ctx.backend.get_candidate(candidate_id).validation_version
                for candidate_id in self.candidate_ids
            }
            self.validation = (
                next(iter(versions)) if len(versions) == 1 else self._default_validation()
            )
        else:
            self.validation = self._default_validation()
        self.validation_label.setText(f"検証用データセット: {self.validation or '—'}")
        self._image_cache.clear()
        self._pixmap_cache.clear()
        self._render_signature = None
        self.index = 0
        if not self.candidate_ids:
            self.block_reason = NO_CANDIDATES_TEXT
        elif not 2 <= len(self.candidate_ids) <= 4:
            self.block_reason = (
                "比較する候補を 2〜4 件選んでください。候補一覧へ戻って選び直してください"
            )
        else:
            versions = {
                self.ctx.backend.get_candidate(candidate_id).validation_version
                for candidate_id in self.candidate_ids
            }
            if len(versions) != 1 or self.validation not in versions:
                self.block_reason = "同じ検証用データセットで評価済みの候補を選んでください"
            else:
                self.block_reason = comparison_block_reason(
                    self.ctx.backend, self.validation, self.candidate_ids
                )
        self.evaluation_ids = {}
        if not self.block_reason:
            for candidate_id in self.candidate_ids:
                record = self.ctx.backend.get_candidate_evaluation(candidate_id, self.validation)
                self.evaluation_ids[candidate_id] = record.evaluation_id
        if self.block_reason:
            self.candidate_ids = []
            text = self.block_reason
            if text != NO_CANDIDATES_TEXT:
                text = f"抽出結果比較を開けません。\n{text}"
            self.placeholder.setText(text)
        self.placeholder.setVisible(bool(self.block_reason))
        self._model_names = {
            candidate_id: self._model_name(candidate_id) for candidate_id in self.candidate_ids
        }
        self._build_views()
        self._load_classifications()
        self._load_items()

    def refresh_on_activate(self) -> None:
        """選択済み候補、固定した評価、検証版、分類、画像位置を保って再描画する。"""
        self._load_items()

    def _default_validation(self) -> str | None:
        try:
            versions = self.ctx.backend.list_validation_versions()
            return versions[-1].version if versions else None
        except (KeyError, ValueError, OSError):
            logger.exception("既定の検証用データセットを決められません")
            return None

    def _model_name(self, candidate_id: str) -> str:
        """候補のモデル種別の表示名を返す（読めなければ空）。"""
        try:
            candidate = self.ctx.backend.get_candidate(candidate_id)
            if candidate.snapshot is not None:
                model_type = candidate.snapshot.experiment_config["model"]["type"]
            else:
                model_type = self.ctx.backend.get_experiment(candidate.experiment_id).model_type
        except (KeyError, ValueError, OSError, TypeError):
            logger.exception("候補のモデル種別を読めません: %s", candidate_id)
            return ""
        return model_type_label(model_type)

    def _load_classifications(self) -> None:
        """検証版に現れる分類を選択肢にする。"""
        names = []
        if not self.block_reason and self.validation:
            try:
                items = self.ctx.backend.list_validation_items(self.validation, None)
            except (KeyError, ValueError, OSError):
                logger.exception("検証用データセットを読めません: %s", self.validation)
                items = []
            names = sorted({item.classification for item in items if item.classification})
        self.classification.blockSignals(True)
        self.classification.clear()
        self.classification.addItem(ALL_CLASSIFICATIONS)
        self.classification.addItems(names)
        self.classification.blockSignals(False)

    def _build_views(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.views, self.labels = [], []
        specs = [("原画像", None)] + [
            (candidate_id, candidate_id) for candidate_id in self.candidate_ids
        ]
        if self.block_reason:
            specs = []
        for position, (_title, _candidate_id) in enumerate(specs):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            view = ImageView()
            view.setMinimumSize(220, 230)
            view.set_scrollbars_visible(False)
            layout.addWidget(view, 1)
            self.grid.addWidget(panel, 0, position)
            self.views.append(view)
            view.installEventFilter(self)
            view.viewport().installEventFilter(self)
        self.candidate_for_view = [None, *self.candidate_ids]
        self.synchronizer = ViewSynchronizer(self.views)

    def _load_items(self, _value=None) -> None:
        classification = self.classification.currentText()
        self.items = []
        if not self.block_reason and self.validation:
            try:
                self.items = self.ctx.backend.list_validation_items(
                    self.validation,
                    None if classification == ALL_CLASSIFICATIONS else classification,
                )
            except (KeyError, ValueError, OSError):
                logger.exception("検証用データセットを読めません: %s", self.validation)
        enabled = not self.block_reason
        for widget in (self.classification, self.item_select, self.previous, self.next, self.fit):
            widget.setEnabled(enabled)
        self.item_select.blockSignals(True)
        self.item_select.clear()
        self.item_select.addItems(
            [f"{item.item_id} ({item.source_filename})" for item in self.items]
        )
        self.item_select.blockSignals(False)
        self.index = min(self.index, max(0, len(self.items) - 1))
        self.item_select.setCurrentIndex(self.index if self.items else -1)
        self._render_current()

    def _select_item(self, index: int) -> None:
        if index >= 0:
            self.index = index
            self._render_current()

    def _move(self, delta: int) -> None:
        if self.items:
            self.index = (self.index + delta) % len(self.items)
            self.item_select.setCurrentIndex(self.index)
            self._render_current()

    def _render_current(self) -> None:
        if not self.items or not self.views:
            self.position.setText("対象画像なし")
            for view in self.views:
                view.set_image(None)
                view.set_overlay_labels("", "")
            self._render_signature = None
            return
        item = self.items[self.index]
        mode = self._display_mode()
        channel = DEFAULT_CHANNEL if DEFAULT_CHANNEL in item.channels else item.channels[0]
        evaluations = tuple(self.evaluation_ids.get(cid) for cid in self.candidate_ids)
        signature = (
            self.validation,
            item.item_id,
            channel,
            mode,
            self.ctx.display.value,
            tuple(self.candidate_ids),
            evaluations,
        )
        self.position.setText(f"({self.index + 1} / {len(self.items)})")
        if signature == self._render_signature:
            return
        image = self._original_image(item.item_id, channel)
        for index, view in enumerate(self.views):
            candidate_id = self.candidate_for_view[index]
            if candidate_id is None:
                title = "原画像"
            else:
                title = f"{candidate_id}　{self._model_names.get(candidate_id, '')}".rstrip()
            entry = None
            if image is not None:
                entry = self._cached_pixmap(item.item_id, channel, candidate_id, image, mode)
            if entry is None:
                view.set_image(None)
                note = "原画像を読み込めません" if image is None else "予測を読み込めません"
                view.set_overlay_labels(title, note)
                continue
            pixmap, count = entry
            view.set_image(pixmap)
            view.set_overlay_labels(title, f"検出 {count} 個" if candidate_id else "")
        self._render_signature = signature
        QTimer.singleShot(0, self.synchronizer.fit_all)

    def _original_image(self, item_id: str, channel: str):
        """確定済みの検証版から原画像を読む（作業中データは読まない）。"""
        key = (self.validation, item_id, channel)
        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            return self._image_cache[key]
        try:
            image = self.ctx.backend.get_dataset_item_image(self.validation, item_id, channel)
        except (KeyError, ValueError, OSError):
            logger.exception("原画像を読めません: %s/%s", self.validation, item_id)
            return None
        self._image_cache[key] = image
        while len(self._image_cache) > _IMAGE_CACHE_SIZE:
            self._image_cache.popitem(last=False)
        return image

    def _cached_pixmap(self, item_id, channel, candidate_id, image, mode):
        """描画済み画像と検出数を返す。キーに検証版・評価 ID・表示形式を含める。"""
        evaluation_id = self.evaluation_ids.get(candidate_id) if candidate_id else None
        view_mode = mode if candidate_id else DisplayMode.IMAGE
        key = (
            self.validation,
            item_id,
            channel,
            candidate_id,
            evaluation_id,
            view_mode,
            self.ctx.display.value if candidate_id else "",
        )
        if key in self._pixmap_cache:
            self._pixmap_cache.move_to_end(key)
            return self._pixmap_cache[key]
        labels = None
        count = 0
        try:
            if candidate_id:
                labels = self.ctx.backend.get_candidate_prediction(
                    candidate_id, evaluation_id, item_id
                )
                count = len(set(labels.ravel())) - (1 if 0 in labels else 0)
            pixmap = array_to_pixmap(render(image, labels, view_mode))
        except (KeyError, ValueError, OSError):
            logger.exception("予測を読めません: %s/%s/%s", candidate_id, evaluation_id, item_id)
            return None
        entry = (pixmap, count)
        self._pixmap_cache[key] = entry
        while len(self._pixmap_cache) > _PIXMAP_CACHE_SIZE:
            self._pixmap_cache.popitem(last=False)
        return entry

    def _cycle_display_mode(self) -> None:
        self.display_toggle.set_alternate(not self.display_toggle.is_alternate)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            if isinstance(self.focusWidget(), QComboBox):
                return super().eventFilter(watched, event)
            if self.shortcuts.matches("zoom_in", event):
                for view in self.views:
                    view.zoom_by(1.2)
                return True
            if self.shortcuts.matches("zoom_out", event):
                for view in self.views:
                    view.zoom_by(1 / 1.2)
                return True
            if self.shortcuts.matches("fit_view", event):
                self._fit_all()
                return True
            if self.shortcuts.matches("display_mode", event):
                self._cycle_display_mode()
                return True
            if self.shortcuts.matches("previous_image", event):
                self._move(-1)
                return True
            if self.shortcuts.matches("next_image", event):
                self._move(1)
                return True
        return super().eventFilter(watched, event)

    def _shortcuts_changed(self) -> None:
        """共有キー変更をヒント表示へ反映する。"""
        self.fit.setToolTip(f"全体表示（{self.shortcuts.display_key(self.shortcuts['fit_view'])}）")
        self.previous.setToolTip(
            f"前の画像（{self.shortcuts.display_key(self.shortcuts['previous_image'])}）"
        )
        self.next.setToolTip(
            f"次の画像（{self.shortcuts.display_key(self.shortcuts['next_image'])}）"
        )
        for key, action in self.view_actions.items():
            action.setText(self._menu_text(action.text().split("\t", 1)[0], key))

    def _display_mode(self) -> DisplayMode:
        """二択表示を候補画像の描画モードへ変換する。"""
        if not self.display_toggle.is_alternate:
            return DisplayMode.IMAGE
        return {
            "オーバーレイ": DisplayMode.OVERLAY,
            "インスタンスラベル": DisplayMode.INSTANCE_LABEL,
            "二値マスク": DisplayMode.BINARY,
        }[self.ctx.display.value]

    def _toggle_display(self, _alternate: bool) -> None:
        """表示切替後に候補画像を描き直す。"""
        self._render_current()

    def _display_changed(self, _name: str) -> None:
        """他画面で変わった表示設定を反映する。"""
        self._render_current()

    def _fit_all(self) -> None:
        self.synchronizer.fit_all()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.shortcuts.matches("previous_image", event):
            self._move(-1)
        elif self.shortcuts.matches("next_image", event):
            self._move(1)
        elif self.shortcuts.matches("fit_view", event):
            self._fit_all()
        else:
            super().keyPressEvent(event)
