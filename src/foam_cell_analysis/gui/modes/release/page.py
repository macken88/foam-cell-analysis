"""リリース済みモデルと分類振り分けを管理する画面。"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...labels import config_key_label, format_datetime, format_score, model_type_label
from ...navigation import PageId
from ...widgets.form import FormSection
from ...widgets.page_base import BasePage
from ...widgets.table import mark_primary, setup_table


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
        self.table.resizeColumnsToContents()
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

    classifications = ("分類A", "分類B", "分類C")

    def __init__(self, ctx, parent=None) -> None:
        super().__init__(
            ctx,
            "リリース済みモデル・振り分け",
            "リリース済みモデルは読み取り専用です。振り分けの変更は適用後に有効になります。",
            parent,
        )
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.model_table = QTableWidget(0, 11)
        self.model_table.setHorizontalHeaderLabels(
            [
                "モデルID",
                "モデル",
                "候補",
                "実験",
                "途中保存モデル",
                "推論設定",
                "検証用データセット",
                "mAP",
                "リリース日時",
                "割り当て中の分類",
                "コメント",
            ]
        )
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        setup_table(self.model_table, stretch_column=10)
        self.model_table.itemSelectionChanged.connect(self._show_model_detail)
        splitter.addWidget(self.model_table)

        lower = QWidget()
        lower_layout = QVBoxLayout(lower)
        self.detail = FormSection("選択モデルの詳細（読み取り専用）")
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlaceholderText("モデルを選択してください")
        self.detail.form.addRow("モデル詳細", self.detail_text)
        lower_layout.addWidget(self.detail)
        self.new_release_button = QPushButton("設定を変えて新しいリリースを作る")
        self.new_release_button.clicked.connect(self._create_new_release)
        lower_layout.addWidget(self.new_release_button, alignment=Qt.AlignmentFlag.AlignRight)

        routing_section = FormSection("モデル振り分け")
        self.routing_table = QTableWidget(3, 3)
        self.routing_table.setHorizontalHeaderLabels(["画像分類", "現在の有効モデル", "変更後"])
        self.routing_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.routing_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        routing_section.form.addRow(self.routing_table)
        lower_layout.addWidget(routing_section)
        note = QLabel("新しいリリース済みモデルを登録しても自動では切り替わりません。")
        note.setStyleSheet("color: #555;")
        lower_layout.addWidget(note)
        actions = QHBoxLayout()
        self.apply_button = QPushButton("変更を適用…")
        self.discard_button = QPushButton("変更を破棄")
        mark_primary(self.apply_button)
        self.apply_button.clicked.connect(self._confirm_apply)
        self.discard_button.clicked.connect(self.discard_changes)
        actions.addStretch(1)
        actions.addWidget(self.apply_button)
        actions.addWidget(self.discard_button)
        lower_layout.addLayout(actions)

        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["日時", "画像分類", "変更前", "変更後"])
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        setup_table(self.history_table, stretch_column=1)
        lower_layout.addWidget(QLabel("変更履歴"))
        lower_layout.addWidget(self.history_table)
        splitter.addWidget(lower)
        splitter.setChildrenCollapsible(False)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setWidget(splitter)
        self.content_layout.addWidget(self.scroll_area)
        self._models = {}
        self._baseline: dict[str, str | None] = {}
        self._routing_controls: dict[str, QComboBox] = {}
        self._rendering = False
        self.routing_table.setColumnCount(3)
        setup_table(self.routing_table, stretch_column=0)

    def on_enter(self, params: dict) -> None:
        """最新モデル・振り分け・履歴を読み直す。"""
        self._load_models()
        self._load_routing()
        self._load_history()
        selected = params.get("select")
        if selected in self._models:
            self.select_model(str(selected))
        elif self._models:
            self.select_model(next(iter(self._models)))

    def _load_models(self) -> None:
        self._models = {model.model_id: model for model in self.ctx.backend.list_released_models()}
        assignments: dict[str, list[str]] = {model_id: [] for model_id in self._models}
        for classification, model_id in self.ctx.backend.get_routing().items():
            if model_id in assignments:
                assignments[model_id].append(classification)
        self.model_table.setRowCount(len(self._models))
        header_height = self.model_table.horizontalHeader().sizeHint().height()
        row_height = self.model_table.verticalHeader().defaultSectionSize()
        table_height = header_height + row_height * len(self._models) + 4
        self.model_table.setMinimumHeight(table_height)
        self.model_table.setMaximumHeight(min(table_height, 190))
        for row, model in enumerate(self._models.values()):
            candidate = self.ctx.backend.get_candidate(model.candidate_id)
            experiment = self.ctx.backend.get_experiment(model.experiment_id)
            values = (
                model.model_id,
                model_type_label(experiment.model_type),
                model.candidate_id,
                model.experiment_id,
                model.checkpoint,
                candidate.inference_config_id,
                model.validation_dataset,
                format_score(model.evaluation_result.overall_map),
                format_datetime(model.released_at),
                ", ".join(assignments[model.model_id]) or "なし",
                model.comment,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, model.model_id)
                self.model_table.setItem(row, column, item)
        for column, width in {
            0: 92,
            1: 110,
            2: 78,
            3: 90,
            4: 120,
            5: 110,
            6: 140,
            7: 70,
            8: 140,
            10: 170,
        }.items():
            self.model_table.setColumnWidth(column, width)

    def _show_model_detail(self) -> None:
        rows = self.model_table.selectionModel().selectedRows()
        if not rows:
            return
        model_id = self.model_table.item(rows[0].row(), 0).text()
        model = self._models.get(model_id)
        if model is None:
            return
        candidate = self.ctx.backend.get_candidate(model.candidate_id)
        details = [
            ("モデルID", model.model_id),
            ("実験識別子", model.experiment_id),
            ("途中保存モデル", model.checkpoint),
            ("前処理設定", model.preprocessing_config),
            ("推論設定", model.inference_config),
            ("検証用データセット", model.validation_dataset),
            ("評価結果", model.evaluation_result),
            ("リリース日時", format_datetime(model.released_at)),
            ("コメント", model.comment or "なし"),
            ("候補ID", candidate.candidate_id),
            ("推論設定ID", candidate.inference_config_id),
        ]
        self.detail_text.setPlainText(
            "\n".join(self._format_detail(key, value) for key, value in details)
        )

    @staticmethod
    def _format_detail(key: str, value: object) -> str:
        """設定辞書をキーごとの読みやすい複数行に整形する。"""
        if isinstance(value, dict):
            lines = [
                f"  {config_key_label(nested_key)}: {nested_value}"
                for nested_key, nested_value in value.items()
            ]
            return f"{key}:\n" + "\n".join(lines)
        if hasattr(value, "overall_map") and hasattr(value, "per_class"):
            lines = [f"  overall_map: {format_score(value.overall_map)}"]
            lines.extend(
                f"  {classification}: mAP {format_score(score)}, 件数 {count}"
                for classification, (score, count) in value.per_class.items()
            )
            return f"{key}:\n" + "\n".join(lines)
        return f"{key}: {value}"

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
        routing = self.ctx.backend.get_routing()
        self._baseline = {key: routing.get(key) for key in self.classifications}
        self._routing_controls.clear()
        self.routing_table.setRowCount(len(self.classifications))
        self._rendering = True
        try:
            for row, classification in enumerate(self.classifications):
                self.routing_table.setItem(row, 0, QTableWidgetItem(classification))
                current = self._baseline[classification]
                self.routing_table.setItem(row, 1, QTableWidgetItem(current or "未割り当て"))
                combo = QComboBox()
                combo.setAccessibleName(f"{classification}の変更後モデル")
                combo.addItem("未割り当て", None)
                for model_id in self._models:
                    combo.addItem(model_id, model_id)
                combo.setCurrentIndex(combo.findData(current))
                combo.currentIndexChanged.connect(lambda _index: self._update_routing_rows())
                self._routing_controls[classification] = combo
                container = QWidget()
                cell_layout = QVBoxLayout(container)
                cell_layout.setContentsMargins(4, 0, 4, 0)
                cell_layout.addWidget(combo)
                self.routing_table.setCellWidget(row, 2, container)
        finally:
            self._rendering = False
        self.routing_table.resizeColumnsToContents()
        self._update_routing_rows()

    def _pending_changes(self) -> dict[str, str | None]:
        return {
            classification: control.currentData()
            for classification, control in self._routing_controls.items()
            if control.currentData() != self._baseline.get(classification)
        }

    def _update_routing_rows(self) -> None:
        changes = self._pending_changes()
        for row, classification in enumerate(self.classifications):
            changed = classification in changes
            color = Qt.GlobalColor.yellow if changed else Qt.GlobalColor.white
            for column in (0, 1):
                item = self.routing_table.item(row, column)
                if item:
                    item.setBackground(color)
            combo = self._routing_controls.get(classification)
            if combo:
                combo.setStyleSheet("background-color: #fff1bf;" if changed else "")
        self.apply_button.setEnabled(bool(changes))
        self.discard_button.setEnabled(bool(changes))

    def build_change_rows(self) -> list[tuple[str, str | None, str | None]]:
        """確認ダイアログに渡す振り分け差分を作る。"""
        pending = self._pending_changes()
        return [
            (classification, self._baseline[classification], pending[classification])
            for classification in self.classifications
            if classification in pending
        ]

    def apply_pending_changes(self) -> list:
        """現在の振り分け変更をBackendへ適用して画面を更新する。"""
        changes = self._pending_changes()
        if not changes:
            return []
        records = self.ctx.backend.apply_routing(changes)
        self._baseline.update(changes)
        self._load_models()
        self._load_routing()
        self._load_history()
        return records

    def _confirm_apply(self) -> None:
        changes = self.build_change_rows()
        if not changes:
            return
        dialog = RoutingChangesDialog(changes, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            records = self.apply_pending_changes()
            if records:
                self.ctx.status.show_message(f"振り分けを{len(records)}件適用しました。")

    def discard_changes(self) -> None:
        """未適用の変更を読み込み時の状態に戻す。"""
        self._load_routing()

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
                self.history_table.setItem(row, column, QTableWidgetItem(value))
        self.history_table.resizeColumnsToContents()
