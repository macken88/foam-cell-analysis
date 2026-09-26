"""複数候補のマスクを同期表示する画面。"""

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QKeyEvent
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


class MaskComparisonPage(BasePage):
    """原画像と候補予測を最大4候補まで比較する。"""

    def __init__(self, ctx, parent=None, *, show_heading: bool = True) -> None:
        super().__init__(
            ctx,
            "マスク比較",
            "候補モデルの予測結果を比較します。",
            parent,
            show_heading=show_heading,
        )
        self.validation = "val_v003"
        self.candidate_ids: list[str] = []
        self.items = []
        self.index = 0
        self.shortcuts = ctx.shortcuts
        self.shortcuts.changed.connect(self._shortcuts_changed)
        self.ctx.display.changed.connect(self._display_changed)
        self.back = QPushButton("← 候補一覧へ戻る")
        self.validation_label = QLabel("検証用データセット: val_v003")
        self.classification = QComboBox()
        self.classification.addItems(["すべて", "分類A", "分類B", "分類C"])
        self.item_select = QComboBox()
        controls = QHBoxLayout()
        controls.addWidget(self.back)
        controls.addWidget(self.validation_label)
        controls.addWidget(QLabel("画像分類:"))
        self.classification.setMaximumWidth(130)
        controls.addWidget(self.classification)
        controls.addWidget(QLabel("対象画像:"))
        controls.addWidget(self.item_select, 1)
        self.display_toggle = DisplayToggle(ctx.display)
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
        self.grid = QGridLayout()
        self.views: list[ImageView] = []
        area = QWidget()
        area.setLayout(self.grid)
        self.placeholder = QLabel("候補一覧で比較する候補を選んでください")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

    def on_enter(self, params: dict) -> None:
        self.validation = params.get("validation_version", "val_v003")
        self.validation_label.setText(f"検証用データセット: {self.validation}")
        self.candidate_ids = params.get("candidate_ids", [])[:4]
        self.placeholder.setVisible(not self.candidate_ids)
        self._build_views()
        self._load_items()

    def refresh_on_activate(self) -> None:
        """選択済み候補、検証版、分類、画像位置を保って再描画する。"""
        self._load_items()

    def _build_views(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.views, self.labels = [], []
        specs = [("原画像", None)] + [
            (candidate_id, candidate_id) for candidate_id in self.candidate_ids
        ]
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
        self.items = self.ctx.backend.list_validation_items(
            self.validation, None if classification == "すべて" else classification
        )
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
            return
        item = self.items[self.index]
        mode = self._display_mode()
        signature = (item.item_id, DEFAULT_CHANNEL, mode, tuple(self.candidate_ids))
        if signature == getattr(self, "_render_signature", None):
            self.position.setText(f"({self.index + 1} / {len(self.items)})")
            return
        image = self.ctx.backend.get_item_image("val", item.item_id, DEFAULT_CHANNEL)
        for index, view in enumerate(self.views):
            candidate_id = self.candidate_for_view[index]
            labels = (
                self.ctx.backend.get_candidate_prediction(candidate_id, item.item_id)
                if candidate_id
                else None
            )
            view.set_image(array_to_pixmap(render(image, labels, mode)))
            if candidate_id:
                candidate = self.ctx.backend.get_candidate(candidate_id)
                experiment = self.ctx.backend.get_experiment(candidate.experiment_id)
                count = len(set(labels.ravel())) - (1 if 0 in labels else 0)
                view.set_overlay_labels(
                    f"{candidate_id}　{model_type_label(experiment.model_type)}",
                    f"検出 {count} 個",
                )
            else:
                view.set_overlay_labels("原画像", "")
        self.position.setText(f"({self.index + 1} / {len(self.items)})")
        self._render_signature = signature
        QTimer.singleShot(0, self.synchronizer.fit_all)

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
        self._render_signature = None
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
