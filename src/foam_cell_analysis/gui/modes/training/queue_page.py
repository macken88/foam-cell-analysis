"""順番に学習する設定を編集・実行する model/view 表。"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableView,
)

from ...context import AppContext
from ...navigation import PageId
from ...settings import app_settings
from ...theme import set_style
from ...widgets.page_base import BasePage
from ...widgets.table import (
    add_row_context_menu,
    bind_button_action,
    fit_table_columns,
    mark_primary,
    restore_row_selection,
    setup_table,
)
from .queue_model import FIELDS, TrainingQueueDelegate, TrainingQueueModel


class TrainingQueuePage(BasePage):
    """キュー項目の閲覧・編集と実行操作を提供する。"""

    def __init__(self, ctx: AppContext, parent=None, *, show_heading=True):
        super().__init__(
            ctx,
            "学習キュー",
            "学習する設定を並べ、順番に実行します。実行前に表で設定を変更できます。",
            parent,
            show_heading,
        )
        self.model = TrainingQueueModel(ctx, self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setItemDelegate(TrainingQueueDelegate(self.model, self.table))
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        setup_table(
            self.table,
            selection_mode=QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        self.table.setColumnWidth(1, max(self.table.columnWidth(1), 260))
        self.table.doubleClicked.connect(self._double_clicked)
        self.model.edit_failed.connect(self.ctx.status.show_message)

        self.status_line = QLabel()
        self.status_line.setWordWrap(True)
        self.empty_label = QLabel("キューは空です。学習設定の『キューに追加』で設定を追加します。")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_style(self.empty_label, role="note")
        status_row = QHBoxLayout()
        status_row.addWidget(self.status_line, 1)
        self.content_layout.addLayout(status_row)
        self.content_layout.addWidget(self.empty_label)
        self.content_layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.column_button = QPushButton("表示する列 ▾")
        self.column_menu = QMenu(self.column_button)
        self.column_menu.setTitle("表示する列（学習キュー）")
        self.column_button.setMenu(self.column_menu)
        self._build_column_menu()
        buttons.addWidget(self.column_button)
        self.run_button = QPushButton("キューを実行")
        self.duplicate_button = QPushButton("複製")
        self.delete_button = QPushButton("削除")
        self.up_button = QPushButton("上へ")
        self.down_button = QPushButton("下へ")
        self.edit_button = QPushButton("設定を開いて編集…")
        self.clear_button = QPushButton("終了した行を片付ける")
        mark_primary(self.run_button)
        self.run_action = QAction("キューを実行", self)
        self.run_action.triggered.connect(self._toggle_run)
        bind_button_action(self.run_button, self.run_action)
        for button in (
            self.duplicate_button,
            self.delete_button,
            self.up_button,
            self.down_button,
            self.edit_button,
            self.clear_button,
        ):
            button.hide()
        self.queue_actions = {
            "duplicate": QAction("複製", self),
            "delete": QAction("削除", self),
            "up": QAction("上へ移動\tCtrl+↑", self),
            "down": QAction("下へ移動\tCtrl+↓", self),
            "edit": QAction("設定を開いて編集…", self),
            "clear": QAction("終了した行を片付ける", self),
        }
        for key, callback in (
            ("duplicate", self.duplicate_selected),
            ("delete", self.delete_selected),
            ("up", lambda: self.move_selected(-1)),
            ("down", lambda: self.move_selected(1)),
            ("edit", self.edit_selected),
            ("clear", self.clear_finished),
        ):
            self.queue_actions[key].triggered.connect(callback)
        row_layout = status_row
        row_layout.addWidget(self.column_button)
        row_layout.addWidget(self.run_button)
        self.context_menu = QMenu(self)
        for key in ("edit", "duplicate", "delete", "up", "down", "clear"):
            self.context_menu.addAction(self.queue_actions[key])
        add_row_context_menu(self.table, self.context_menu)
        self.up_shortcut = QShortcut(QKeySequence("Ctrl+Up"), self)
        self.up_shortcut.activated.connect(lambda: self.move_selected(-1))
        self.down_shortcut = QShortcut(QKeySequence("Ctrl+Down"), self)
        self.down_shortcut.activated.connect(lambda: self.move_selected(1))
        self.table.selectionModel().selectionChanged.connect(self._update_buttons)
        self.model.modelReset.connect(self._update_empty_state)
        ctx.queue_controller.changed.connect(self.refresh)
        ctx.queue_controller.progressed.connect(self._update_status_line)
        self.refresh(initial=True)

    def menu_actions(self):
        return {
            "edit": [
                self.queue_actions["duplicate"],
                self.queue_actions["delete"],
                None,
                self.queue_actions["up"],
                self.queue_actions["down"],
                None,
                self.queue_actions["edit"],
            ],
            "training": [self.run_action, self.queue_actions["clear"], None],
            "view": [None, self.column_menu.menuAction()],
        }

    def _build_column_menu(self) -> None:
        categories: dict[str, list[tuple[int, str]]] = {}
        for column, (_path, label, group) in enumerate(FIELDS, 3):
            categories.setdefault(group, []).append((column, label))
        settings = app_settings()
        for group, columns in categories.items():
            submenu = self.column_menu.addMenu(group)
            for column, label in columns:
                action = submenu.addAction(label)
                action.setCheckable(True)
                action.setChecked(
                    settings.value(f"training_queue/columns/{column}", True, type=bool)
                )
                action.toggled.connect(
                    lambda visible, col=column: self._set_column_visible(col, visible)
                )
                self.table.setColumnHidden(column, not action.isChecked())

    def _set_column_visible(self, column: int, visible: bool) -> None:
        self.table.setColumnHidden(column, not visible)
        app_settings().setValue(f"training_queue/columns/{column}", visible)

    def _selected_ids(self) -> list[str]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return [
            self.model.entries[row].queue_id or self.model.entries[row].experiment_id
            for row in rows
        ]

    def _current_id(self) -> str | None:
        row = self.table.currentIndex().row()
        return (
            self.model.entries[row].queue_id or self.model.entries[row].experiment_id
            if 0 <= row < len(self.model.entries)
            else None
        )

    def refresh(self, *_args, initial: bool = False) -> None:
        selected_ids = set(self._selected_ids())
        current_id = self._current_id()
        scroll = self.table.verticalScrollBar().value()
        structure_changed = self.model.refresh()
        if structure_changed:
            row_by_id = {
                entry.queue_id or entry.experiment_id: row
                for row, entry in enumerate(self.model.entries)
            }
            current_row = row_by_id.get(current_id)
            restore_row_selection(
                self.table,
                [row_by_id[key] for key in selected_ids if key in row_by_id],
                current_row,
            )
            if initial or structure_changed:
                fit_table_columns(self.table)
            if initial:
                self.table.setColumnWidth(1, max(self.table.columnWidth(1), 260))
            self.table.verticalScrollBar().setValue(scroll)
        self._update_empty_state()
        self._update_status_line()
        self._update_buttons()

    def _update_empty_state(self, *_args) -> None:
        self.empty_label.setVisible(not self.model.entries)

    def _update_status_line(self) -> None:
        entries = self.model.entries
        controller = self.ctx.queue_controller
        waiting = sum(entry.status == "queued" for entry in entries)
        active = None
        if controller.active_id:
            try:
                active = self.ctx.backend.get_experiment(controller.active_id)
            except KeyError:
                active = None
        running = f"実行中 {controller.active_id}"
        if active:
            running += f"（{self.model._progress(active)}）"
        elif not controller.active_id:
            running = (
                f"{controller.waiting_for_id or '実行中の学習'}の学習終了後にキューを開始"
                if controller.waiting_for_training
                else "実行中・単発待ち"
                if controller.executing
                else "停止中"
            )
        counts = {
            status: sum(entry.status == status for entry in entries)
            for status in ("completed", "failed", "stopped")
        }
        note = ""
        if controller.stop_requested:
            note = (
                "\n今の学習が終わったら停止します。"
                "すぐに止めるには実験一覧の「中断」を使ってください。"
            )
        self.status_line.setText(
            f"待機 {waiting} 件・{running}・完了 {counts['completed']} 件・"
            f"失敗 {counts['failed']} 件・中断 {counts['stopped']} 件{note}"
        )

    def _update_buttons(self, *_args) -> None:
        selected = self._selected_ids()
        entries_by_id = {
            entry.queue_id or entry.experiment_id: entry for entry in self.model.entries
        }
        queued = [
            key
            for key in selected
            if entries_by_id.get(key) and entries_by_id[key].status == "queued"
        ]
        can_duplicate = bool(selected) and all(
            not entries_by_id[key].queue_is_retry for key in selected
        )
        self.duplicate_button.setEnabled(can_duplicate)
        self.set_menu_action_enabled(self.queue_actions["duplicate"], can_duplicate)
        self.queue_actions["duplicate"].setToolTip(
            "複製する学習キューの行を選んでください"
            if not selected
            else "再試行予約は複製できません"
            if not can_duplicate
            else ""
        )
        self.duplicate_button.setToolTip(
            "複製する行を選択してください"
            if not selected
            else "再試行予約は複製できません"
            if not can_duplicate
            else ""
        )
        self.delete_button.setEnabled(bool(queued))
        self.set_menu_action_enabled(self.queue_actions["delete"], bool(queued))
        self.queue_actions["delete"].setToolTip(
            "削除する待機中の学習項目を選んでください" if not queued else ""
        )
        self.delete_button.setToolTip("待機中の項目だけ削除できます" if not queued else "")
        controller = self.ctx.queue_controller
        waiting = any(entry.status == "queued" for entry in self.model.entries)
        self.run_button.setText("実行を停止" if controller.executing else "キューを実行")
        self.run_button.setEnabled(controller.executing or waiting)
        self.run_action.setText("実行を停止" if controller.executing else "キューを実行")
        self.set_menu_action_enabled(self.run_action, controller.executing or waiting)
        self.run_action.setToolTip(
            "待機中の学習がありません" if not (controller.executing or waiting) else ""
        )
        self.run_button.setToolTip(
            "待機中の学習がありません" if not self.run_button.isEnabled() else ""
        )
        ids = [entry.queue_id or entry.experiment_id for entry in self.model.entries]
        can_move_up = any(
            key in ids
            and entries_by_id.get(key) is not None
            and entries_by_id[key].status == "queued"
            and ids.index(key) > 0
            and ids[ids.index(key) - 1] not in selected
            and entries_by_id[ids[ids.index(key) - 1]].status == "queued"
            for key in queued
        )
        can_move_down = any(
            key in ids
            and entries_by_id.get(key) is not None
            and entries_by_id[key].status == "queued"
            and ids.index(key) + 1 < len(ids)
            and ids[ids.index(key) + 1] not in selected
            and entries_by_id[ids[ids.index(key) + 1]].status == "queued"
            for key in queued
        )
        self.up_button.setEnabled(can_move_up)
        self.down_button.setEnabled(can_move_down)
        self.set_menu_action_enabled(self.queue_actions["up"], can_move_up)
        self.set_menu_action_enabled(self.queue_actions["down"], can_move_down)
        self.queue_actions["up"].setToolTip(
            "上へ移動できる待機中の学習項目を選んでください" if not can_move_up else ""
        )
        self.queue_actions["down"].setToolTip(
            "下へ移動できる待機中の学習項目を選んでください" if not can_move_down else ""
        )
        self.up_button.setToolTip(
            "移動できる待機行を選択してください" if not can_move_up else "Ctrl+↑"
        )
        self.down_button.setToolTip(
            "移動できる待機行を選択してください" if not can_move_down else "Ctrl+↓"
        )
        current_id = self._current_id()
        current_entry = entries_by_id.get(current_id)
        editable = bool(
            current_entry and current_entry.status == "queued" and not current_entry.queue_is_retry
        )
        self.edit_button.setEnabled(editable)
        self.set_menu_action_enabled(self.queue_actions["edit"], editable)
        self.queue_actions["edit"].setToolTip(
            "編集する待機中の学習項目を選んでください" if not editable else ""
        )
        self.edit_button.setToolTip("編集できる待機行を選択してください" if not editable else "")
        can_clear = any(
            entry.status in {"completed", "failed", "stopped"} for entry in self.model.entries
        )
        self.clear_button.setEnabled(can_clear)
        self.set_menu_action_enabled(self.queue_actions["clear"], can_clear)
        self.queue_actions["clear"].setToolTip(
            "片付ける終了行がありません" if not can_clear else ""
        )
        self.clear_button.setToolTip("片付ける終了行がありません" if not can_clear else "")

    def _double_clicked(self, index) -> None:
        if index.column() < self.model.fixed_column_count:
            entry = self.model.entries[index.row()]
            experiment_id = entry.experiment_id
            if entry.status == "queued" and not entry.queue_is_retry:
                self.edit_row(index.row())
            else:
                self.ctx.navigator.navigate(PageId.EXPERIMENTS, select=experiment_id)
        else:
            self.table.edit(index)

    def edit_cell(self, row: int, path: str, value) -> bool:
        """テストとアクセシビリティ操作から同じ model 編集経路を呼ぶ。"""
        if not 0 <= row < self.model.rowCount():
            return False
        column = self.model.column_for_path(path)
        return self.model.setData(self.model.index(row, column), value)

    def _toggle_run(self) -> None:
        controller = self.ctx.queue_controller
        controller.stop() if controller.executing else controller.start()

    def duplicate_selected(self) -> None:
        self.ctx.backend.duplicate_training_queue_items(self._selected_ids())
        self.ctx.queue_controller.sync_training_identifier()
        self.refresh()
        self.ctx.queue_controller.changed.emit()

    def delete_selected(self) -> None:
        ids = self._selected_ids()
        if (
            ids
            and QMessageBox.question(
                self, "キューから削除", f"選択した {len(ids)} 件を削除しますか？"
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.ctx.backend.delete_training_queue_items(ids)
            self.refresh()
            self.ctx.queue_controller.changed.emit()

    def move_selected(self, delta: int) -> None:
        entries = self.model.entries
        ids = [entry.queue_id or entry.experiment_id for entry in entries]
        entries_by_id = {entry.queue_id or entry.experiment_id: entry for entry in entries}
        selected = self._selected_ids()
        indexes = range(len(ids)) if delta < 0 else range(len(ids) - 1, -1, -1)
        for index in indexes:
            neighbor = index + delta
            if (
                0 <= neighbor < len(ids)
                and ids[index] in selected
                and entries_by_id[ids[index]].status == "queued"
                and not entries_by_id[ids[index]].queue_is_retry
                and ids[neighbor] not in selected
                and entries_by_id[ids[neighbor]].status == "queued"
            ):
                ids[index], ids[neighbor] = ids[neighbor], ids[index]
        self.ctx.backend.reorder_training_queue(ids)
        self.refresh()

    def clear_finished(self) -> None:
        self.ctx.backend.clear_finished_training_queue_items()
        self.refresh()
        self.ctx.queue_controller.changed.emit()

    def edit_selected(self) -> None:
        if self._selected_ids():
            self.edit_row(self.table.currentIndex().row())

    def edit_row(self, row: int) -> None:
        if 0 <= row < len(self.model.entries) and not self.model.entries[row].queue_is_retry:
            self.ctx.navigator.navigate(
                PageId.TRAINING, edit_queue=self.model.entries[row].experiment_id
            )

    def on_enter(self, params) -> None:
        self.refresh()
        selected = params.get("select")
        if selected:
            row = self.model.row_for_id(selected)
            if row >= 0:
                self.table.selectRow(row)
                self.table.setCurrentIndex(self.model.index(row, 2))

    def refresh_on_activate(self) -> None:
        """前面化で状態列だけ更新し、選択や編集中のデリゲートを維持する。"""
        self.refresh()
