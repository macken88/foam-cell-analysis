"""リリース済みモデルと分類振り分けを管理する画面。"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...labels import config_key_label, format_datetime, format_score, model_type_label
from ...navigation import PageId
from ...theme import Color, numeric_font, set_style
from ...widgets.form import FormSection
from ...widgets.page_base import BasePage
from ...widgets.table import (
    add_row_context_menu,
    bind_button_action,
    fit_table_columns,
    mark_primary,
    restore_row_selection,
    setup_table,
)

logger = logging.getLogger(__name__)

ROUTING_CONFLICT_TEXT = "振り分けが別の操作で変更されました。画面を開き直してください"
ROUTING_FAILED_TEXT = (
    "振り分けを適用できませんでした。画面を開き直してから、もう一度適用してください。"
)


class RoutingChangesDialog(QDialog):
    """振り分けの変更内容を確認する。"""

    def __init__(self, changes: list[tuple[str, str | None, str | None]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("振り分け変更の確認")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("次の振り分け変更を適用します。"))
        self.table = QTableWidget(len(changes), 3)
        self.table.setHorizontalHeaderLabels(["画像分類", "変更前", "変更後"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        setup_table(self.table, stretch_column=0)
        for row, (classification, before, after) in enumerate(changes):
            for column, value in enumerate(
                (classification, before or "未割り当て", after or "未割り当て")
            ):
                self.table.setItem(row, column, QTableWidgetItem(value))
        fit_table_columns(self.table)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("適用")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        self.setMinimumSize(440, max(220, min(520, 150 + len(changes) * 34)))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ReleasedModelsPage(BasePage):
    """リリース済みモデルの参照と振り分けを行う。"""

    def __init__(self, ctx, parent=None, *, show_heading: bool = True) -> None:
        super().__init__(
            ctx,
            "リリース済みモデル・振り分け",
            "リリース済みモデルは読み取り専用です。振り分けの変更は適用後に有効になります。",
            parent,
            show_heading=show_heading,
        )
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.model_table = QTableWidget(0, 12)
        self.model_table.setHorizontalHeaderLabels(
            [
                "モデルID",
                "モデル",
                "候補",
                "実験",
                "途中保存モデル",
                "検証用データセット",
                "検証 AP",
                "OOF AP",
                "推論設定",
                "リリース日時",
                "割り当て中の分類",
                "コメント",
            ]
        )
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        setup_table(
            self.model_table,
            stretch_column=10,
            selection_mode=QTableWidget.SelectionMode.ExtendedSelection,
        )
        self.model_table.itemSelectionChanged.connect(self._show_model_detail)

        lower = QWidget()
        lower_layout = QVBoxLayout(lower)
        self.detail = FormSection("選択モデルの詳細")
        self.new_release_button = QPushButton("設定を変えて新しいリリースを作る")
        self.new_release_button.setEnabled(False)
        self.new_release_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.release_actions = {
            "new": QAction("設定を変えて新しいリリースを作る", self),
            "detail": QAction("リリースの詳細を表示…", self),
            "discard": QAction("振り分けの変更を破棄", self),
        }
        self.release_actions["new"].triggered.connect(self._create_new_release)
        self.release_actions["detail"].triggered.connect(self._show_detail_dialog)
        self.release_actions["discard"].triggered.connect(self.discard_changes)
        bind_button_action(self.new_release_button, self.release_actions["new"])
        self.detail_values: dict[str, QLabel] = {}
        for label in (
            "実験・途中保存モデル",
            "検証 AP",
            "OOF AP",
            "推論設定",
            "検証用データセット",
            "リリース日時",
            "コメント",
        ):
            value = QLabel("モデルを選択してください" if not self.detail_values else "—")
            value.setWordWrap(True)
            self.detail_values[label] = value
            self.detail.form.addRow(label, value)
        self.detail_button = QPushButton("詳細を表示…")
        self.detail_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.detail_button.setEnabled(False)
        bind_button_action(self.detail_button, self.release_actions["detail"])
        self.detail_button.hide()
        self.new_release_button.hide()
        # 複数行の値（分類別 AP・推論設定）が詰まって重ならないよう、詳細はスクロールで見せる
        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.detail_scroll.setWidget(self.detail)
        self.detail_scroll.setMinimumHeight(230)
        splitter.addWidget(self.model_table)

        routing_section = FormSection("モデル振り分け")
        routing_section.form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.routing_table = QTableWidget(3, 3)
        self.routing_table.setHorizontalHeaderLabels(["画像分類", "現在の有効モデル", "変更後"])
        self.routing_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.routing_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.routing_table.setMinimumWidth(720)
        self.routing_table.horizontalHeader().setStretchLastSection(False)
        self.routing_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.routing_table.setColumnWidth(0, 110)
        self.routing_table.setColumnWidth(1, 190)
        routing_section.form.addRow(self.routing_table)
        # 振り分けと選択モデルの詳細を横に並べ、一覧の表は横幅いっぱいに使う
        middle = QHBoxLayout()
        middle.addWidget(routing_section, 0, Qt.AlignmentFlag.AlignTop)
        middle.addWidget(self.detail_scroll, 1)
        lower_layout.addLayout(middle)
        note = QLabel("新しいリリース済みモデルを登録しても、自動では切り替わりません")
        set_style(note, role="note")
        actions = QHBoxLayout()
        self.apply_button = QPushButton("変更を適用…")
        self.discard_button = QPushButton("変更を破棄")
        self.apply_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.discard_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        mark_primary(self.apply_button)
        self.change_count_label = QLabel()
        self.apply_action = QAction("変更を適用…", self)
        self.apply_action.triggered.connect(self._confirm_apply)
        bind_button_action(self.apply_button, self.apply_action)
        bind_button_action(self.discard_button, self.release_actions["discard"])
        actions.addWidget(self.change_count_label)
        actions.addWidget(self.apply_button)
        actions.addStretch(1)
        actions.addWidget(note)
        lower_layout.addLayout(actions)
        self.context_menu = QMenu(self)
        self.context_menu.addAction(self.release_actions["detail"])
        self.context_menu.addAction(self.release_actions["new"])
        add_row_context_menu(self.model_table, self.context_menu)
        self.routing_context_menu = QMenu(self)
        self.routing_context_menu.addAction(self.release_actions["discard"])
        self.routing_context_menu.addAction(self.apply_action)
        add_row_context_menu(self.routing_table, self.routing_context_menu)

        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["日時", "画像分類", "変更前", "変更後"])
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        setup_table(self.history_table, stretch_column=1)
        lower_layout.addWidget(QLabel("変更履歴"))
        lower_layout.addWidget(self.history_table)
        splitter.addWidget(lower)
        splitter.setSizes([280, 520])
        splitter.setChildrenCollapsible(False)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setWidget(splitter)
        self.content_layout.addWidget(self.scroll_area)
        self._models = {}
        self._baseline: dict[str, str | None] = {}
        self._classifications: list[str] = []
        self._assignments: dict[str, str | None] = {}
        # 画面に読み込んだ振り分けの revision。適用時に expected_revision として渡す（14 章）
        self._revision: int | None = None
        self._routing_controls: dict[str, QComboBox] = {}
        self._rendering = False
        self.routing_table.setColumnCount(3)
        setup_table(self.routing_table, stretch_column=2)
        self.routing_table.setColumnWidth(0, 110)
        self.routing_table.setColumnWidth(1, 190)

    def menu_actions(self):
        return {
            "release": [
                self.release_actions["detail"],
                self.release_actions["new"],
                None,
                self.apply_action,
                self.release_actions["discard"],
            ]
        }

    def on_enter(self, params: dict) -> None:
        """最新モデル・振り分け・履歴を読み直す。"""
        self._reload()
        selected = params.get("select")
        if selected in self._models:
            self.select_model(str(selected))
        elif self._models:
            self.select_model(next(iter(self._models)))

    def refresh_on_activate(self) -> None:
        """一覧を読み直し、選択中のリリースモデルを保つ。

        未適用の変更があるときは、読み込み時の振り分けと revision を保つ。別の操作で
        振り分けが変わっていれば、適用時に衝突として知らせる。
        """
        if self._pending_changes():
            self._load_models()
            self._load_history()
            return
        self._reload()

    def _reload(self) -> None:
        """振り分けの現在値・revision・分類を読み直し、全体を描き直す。"""
        self._read_routing_state()
        self._load_models()
        self._load_routing()
        self._load_history()

    def _read_routing_state(self) -> None:
        state = self.ctx.backend.get_routing_state()
        self._revision = state.revision
        self._assignments = dict(state.assignments)
        self._classifications = list(self.ctx.backend.list_routing_classifications())

    def _load_models(self) -> None:
        current_row = self.model_table.currentRow()
        current_id = (
            self.model_table.item(current_row, 0).text()
            if current_row >= 0 and self.model_table.item(current_row, 0)
            else None
        )
        selected_ids = {
            item.text()
            for index in self.model_table.selectionModel().selectedRows()
            if (item := self.model_table.item(index.row(), 0)) is not None
        }
        scroll_value = self.model_table.verticalScrollBar().value()
        self._models = {model.model_id: model for model in self.ctx.backend.list_released_models()}
        assignments: dict[str, list[str]] = {model_id: [] for model_id in self._models}
        for classification, model_id in self._assignments.items():
            if model_id in assignments:
                assignments[model_id].append(classification)
        self.model_table.setRowCount(len(self._models))
        header_height = self.model_table.horizontalHeader().sizeHint().height()
        row_height = self.model_table.verticalHeader().defaultSectionSize()
        table_height = header_height + row_height * len(self._models) + 4
        self.model_table.setMinimumHeight(table_height)
        self.model_table.setMaximumHeight(min(table_height, 190))
        for row, model in enumerate(self._models.values()):
            values = (
                model.model_id,
                model_type_label(model.model_type) if model.model_type else "—",
                model.candidate_id,
                model.experiment_id,
                model.checkpoint,
                model.validation_dataset,
                format_score(model.evaluation_result.overall_map),
                format_score(model.oof_evaluation.overall_map) if model.oof_evaluation else "—",
                model.inference_config_id or "—",
                format_datetime(model.released_at),
                ", ".join(assignments[model.model_id]) or "なし",
                model.comment,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column in (0, 2, 3, 4, 5, 6, 7, 8, 9):
                    item.setFont(numeric_font())
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                item.setData(Qt.ItemDataRole.UserRole, model.model_id)
                self.model_table.setItem(row, column, item)
        fit_table_columns(self.model_table)
        model_rows = {
            self.model_table.item(row, 0).text(): row for row in range(self.model_table.rowCount())
        }
        rows = {model_rows[model_id] for model_id in selected_ids if model_id in model_rows}
        restore_row_selection(self.model_table, rows, model_rows.get(current_id))
        self.model_table.verticalScrollBar().setValue(scroll_value)

    def _show_model_detail(self) -> None:
        rows = self.model_table.selectionModel().selectedRows()
        if not rows:
            self._selected_model = None
            self.new_release_button.setEnabled(False)
            self.detail_button.setEnabled(False)
            self.set_menu_action_enabled(self.release_actions["new"], False)
            self.set_menu_action_enabled(self.release_actions["detail"], False)
            self.release_actions["new"].setToolTip("リリース済みモデルの行を選ぶと使えます")
            self.release_actions["detail"].setToolTip("リリース済みモデルの行を選ぶと使えます")
            for index, value in enumerate(self.detail_values.values()):
                value.setText("モデルを選択してください" if index == 0 else "—")
            return
        model_id = self.model_table.item(rows[0].row(), 0).text()
        model = self._models.get(model_id)
        if model is None:
            return
        evaluation = model.evaluation_result
        per_class = "、".join(
            f"{classification} {format_score(score)}"
            for classification, (score, _count) in evaluation.per_class.items()
        )
        summary = {
            "実験・途中保存モデル": (
                f"{model.experiment_id} ・ 試行 {model.source_attempt_number}/{model.checkpoint}"
            ),
            "検証 AP": f"全体 {format_score(evaluation.overall_map)}"
            + (f"\n{per_class}" if per_class else ""),
            "OOF AP": format_score(model.oof_evaluation.overall_map)
            if model.oof_evaluation
            else "—",
            "推論設定": self._inference_summary(model.inference_config),
            "検証用データセット": model.validation_dataset,
            "リリース日時": format_datetime(model.released_at),
            "コメント": model.comment or "なし",
        }
        for label, text in summary.items():
            widget = self.detail_values[label]
            widget.setText(text)
        self._selected_model = model
        self.new_release_button.setEnabled(True)
        self.detail_button.setEnabled(True)
        self.set_menu_action_enabled(self.release_actions["new"], True)
        self.set_menu_action_enabled(self.release_actions["detail"], True)
        self.release_actions["new"].setToolTip("")
        self.release_actions["detail"].setToolTip("")

    def _show_detail_dialog(self) -> None:
        """選択中モデルの全設定を読み取り専用で表示する。"""
        if not hasattr(self, "_selected_model"):
            return
        model = self._selected_model
        rows = [
            ("モデルID", model.model_id),
            ("モデル", model_type_label(model.model_type) if model.model_type else "—"),
            ("候補ID", model.candidate_id),
            ("実験識別子", model.experiment_id),
            ("途中保存モデル", f"試行 {model.source_attempt_number}/{model.checkpoint}"),
            ("検証用データセット", model.validation_dataset),
            (
                "OOF AP",
                format_score(model.oof_evaluation.overall_map) if model.oof_evaluation else "—",
            ),
            ("リリース日時", format_datetime(model.released_at)),
            ("コメント", model.comment or "なし"),
        ]
        rows.extend(self._flatten_detail("前処理設定", model.preprocessing_config))
        rows.extend(self._flatten_detail("推論設定", model.inference_config))
        evaluation = model.evaluation_result
        rows.append(("評価結果 / 全体 AP", format_score(evaluation.overall_map)))
        rows.extend(
            (f"評価結果 / {classification} AP", format_score(score))
            for classification, (score, _count) in evaluation.per_class.items()
        )
        rows.extend(
            (f"評価結果 / {classification} 件数", str(count))
            for classification, (_score, count) in evaluation.per_class.items()
        )
        if model.oof_evaluation:
            rows.append(("OOF / 全体 AP", format_score(model.oof_evaluation.overall_map)))
            rows.extend(
                (f"OOF / {classification} AP", format_score(score))
                for classification, (score, _count) in model.oof_evaluation.per_class.items()
            )
            rows.extend(
                (f"OOF / {classification} 件数", str(count))
                for classification, (_score, count) in model.oof_evaluation.per_class.items()
            )
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{model.model_id} の詳細")
        dialog.resize(680, 560)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["項目", "値"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        setup_table(table, stretch_column=1)
        for row, (label, value) in enumerate(rows):
            table.setItem(row, 0, QTableWidgetItem(label))
            item = QTableWidgetItem(str(value))
            item.setFont(numeric_font())
            table.setItem(row, 1, item)
        layout.addWidget(table)
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.exec()

    @classmethod
    def _flatten_detail(
        cls, prefix: str, value: object, key_prefix: str = "model"
    ) -> list[tuple[str, str]]:
        """設定の階層を日本語ラベルの行へ展開する。"""
        if isinstance(value, dict):
            rows = []
            for key, nested in value.items():
                path = f"{key_prefix}.{key}" if key_prefix else key
                if isinstance(nested, dict):
                    rows.extend(cls._flatten_detail(prefix, nested, path))
                    continue
                label = config_key_label(path)
                if label == path:
                    label = config_key_label(key)
                if label == key:
                    label = "その他の設定"
                rows.extend(cls._flatten_detail(f"{prefix} / {label}", nested, path))
            return rows
        if isinstance(value, (list, tuple)):
            rendered = "、".join(cls._display_detail_value(item) for item in value)
        else:
            rendered = cls._display_detail_value(value, key_prefix)
        return [(prefix, rendered)]

    @staticmethod
    def _display_detail_value(value: object, key: str = "") -> str:
        """設定値を内部表現を避けて日本語に整える。"""
        if value is None:
            return "未設定"
        if isinstance(value, bool):
            return "有効" if value else "無効"
        if isinstance(value, float):
            return f"{value:.6g}"
        if key.endswith(".type"):
            return model_type_label(str(value))
        if key.endswith(".pretrained_weights"):
            return {"coco": "COCO", "imagenet": "ImageNet"}.get(str(value), str(value))
        if key.endswith(".backbone"):
            return {"resnet50_fpn_v2": "ResNet-50 FPN v2", "resnet101_fpn": "ResNet-101 FPN"}.get(
                str(value), str(value)
            )
        if key.endswith(".best_metric") and value == "oof_instance_map":
            return "OOF AP（Cellpose 方式）・最大"
        if value in {"good_only", "good_and_acceptable", "all"}:
            return {"good_only": "良のみ", "good_and_acceptable": "良・可", "all": "すべて"}[
                str(value)
            ]
        return str(value)

    @classmethod
    def _inference_summary(cls, params: dict) -> str:
        """推論設定を「検出スコア閾値 0.5」の形で 1 行ずつ並べる。"""
        rows = cls._flatten_detail("", params or {})
        return "\n".join(f"{label.removeprefix(' / ')} {value}" for label, value in rows) or "—"

    def select_model(self, model_id: str) -> bool:
        """モデルIDの行を選択する。"""
        for row in range(self.model_table.rowCount()):
            if self.model_table.item(row, 0).text() == model_id:
                self.model_table.selectRow(row)
                return True
        return False

    def _create_new_release(self) -> None:
        rows = self.model_table.selectionModel().selectedRows()
        if not rows:
            return
        model = self._models[self.model_table.item(rows[0].row(), 0).text()]
        self.ctx.navigator.navigate(
            PageId.CANDIDATES,
            action="add_candidate",
            experiment_id=model.experiment_id,
            checkpoint=model.checkpoint,
        )

    def _load_routing(self) -> None:
        self._baseline = {key: self._assignments.get(key) for key in self._classifications}
        self._routing_controls.clear()
        self.routing_table.setRowCount(len(self._classifications))
        self._rendering = True
        try:
            for row, classification in enumerate(self._classifications):
                self.routing_table.setItem(row, 0, QTableWidgetItem(classification))
                current = self._baseline[classification]
                current_item = QTableWidgetItem(current or "未割り当て")
                if current is None:
                    current_item.setForeground(QColor(Color.ERROR))
                self.routing_table.setItem(row, 1, current_item)
                combo = QComboBox()
                combo.setMinimumWidth(180)
                combo.setAccessibleName(f"{classification}の変更後モデル")
                combo.addItem("未割り当て", None)
                for model_id in self._models:
                    combo.addItem(model_id, model_id)
                if current is not None and combo.findData(current) < 0:
                    # 読めないリリースに割り当てられている分類も、今の値を残して表示する
                    combo.addItem(f"{current}（読み込めません）", current)
                combo.setCurrentIndex(combo.findData(current))
                combo.currentIndexChanged.connect(lambda _index: self._update_routing_rows())
                self._routing_controls[classification] = combo
                container = QWidget()
                cell_layout = QHBoxLayout(container)
                cell_layout.setContentsMargins(4, 0, 4, 0)
                cell_layout.addWidget(combo)
                cell_layout.addStretch(1)
                self.routing_table.setCellWidget(row, 2, container)
        finally:
            self._rendering = False
        fit_table_columns(self.routing_table)
        self._update_routing_rows()

    def _pending_changes(self) -> dict[str, str | None]:
        return {
            classification: control.currentData()
            for classification, control in self._routing_controls.items()
            if control.currentData() != self._baseline.get(classification)
        }

    def _update_routing_rows(self) -> None:
        changes = self._pending_changes()
        for row, classification in enumerate(self._classifications):
            changed = classification in changes
            color = Color.CHANGED if changed else Color.SLIDE
            for column in (0, 1):
                item = self.routing_table.item(row, column)
                if item:
                    item.setBackground(QColor(color))
            combo = self._routing_controls.get(classification)
            if combo:
                set_style(combo, state="changed" if changed else "")
        self.apply_button.setEnabled(bool(changes))
        self.discard_button.setEnabled(bool(changes))
        self.set_menu_action_enabled(self.apply_action, bool(changes))
        self.set_menu_action_enabled(self.release_actions["discard"], bool(changes))
        self.apply_action.setToolTip("振り分けを変更すると使えます" if not changes else "")
        self.release_actions["discard"].setToolTip(
            "破棄する振り分け変更はありません" if not changes else ""
        )
        self.change_count_label.setText(f"{len(changes)} 件の変更があります" if changes else "")
        self.change_count_label.setVisible(bool(changes))

    def build_change_rows(self) -> list[tuple[str, str | None, str | None]]:
        """確認ダイアログに渡す振り分け差分を作る。"""
        pending = self._pending_changes()
        return [
            (classification, self._baseline[classification], pending[classification])
            for classification in self._classifications
            if classification in pending
        ]

    def apply_pending_changes(self) -> int:
        """読み込み時の revision を付けて振り分け変更を適用し、適用した件数を返す。

        別の操作で振り分けが変わっていたときは何も変えずに知らせ、画面を読み直す。
        """
        changes = self._pending_changes()
        if not changes:
            return 0
        try:
            self.ctx.backend.apply_routing(changes, expected_revision=self._revision)
        except (ValueError, KeyError, OSError):
            logger.exception("振り分けを適用できません")
            try:
                current = self.ctx.backend.get_routing_state().revision
            except (ValueError, KeyError, OSError):
                current = None
            conflict = current is not None and current != self._revision
            QMessageBox.warning(
                self,
                "振り分けを適用できません",
                ROUTING_CONFLICT_TEXT if conflict else ROUTING_FAILED_TEXT,
            )
            self._reload()
            return 0
        self._reload()
        return len(changes)

    def _confirm_apply(self) -> None:
        changes = self.build_change_rows()
        if not changes:
            return
        dialog = RoutingChangesDialog(changes, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            count = self.apply_pending_changes()
            if count:
                self.ctx.status.show_message(f"振り分けを{count}件適用しました。")

    def discard_changes(self) -> None:
        """未適用の変更を捨て、振り分けを読み直す。"""
        self._reload()

    def _load_history(self) -> None:
        history = self.ctx.backend.list_routing_history()
        self.history_table.setRowCount(len(history))
        for row, record in enumerate(reversed(history)):
            values = (
                format_datetime(record.changed_at),
                record.classification,
                record.before_model_id or "未割り当て",
                record.after_model_id or "未割り当て",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setFont(numeric_font())
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.history_table.setItem(row, column, item)
        fit_table_columns(self.history_table)
