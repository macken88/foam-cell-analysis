"""単一モードのページを表示するウィンドウ。"""

from datetime import datetime

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .context import AppContext
from .labels import autosave_label, running_jobs_label
from .navigation import ModeId, PageId
from .theme import numeric_font
from .widgets.marks import LayoutButton

MODE_LABELS = {
    ModeId.DATA_PREPARATION: "データ準備",
    ModeId.TRAINING: "モデル学習",
    ModeId.COMPARISON: "モデル比較・リリース",
    ModeId.INFERENCE: "本番推論",
}
MODE_TAB_LABELS = {
    PageId.DATA_PREPARATION: (
        "作業中データ",
        "作業データを編集し、整合性を確認してデータセット版を確定します。",
    ),
    PageId.DATASET_HISTORY: (
        "データセット版履歴",
        "確定済みの学習用版・検証用版を確認します。",
    ),
    PageId.TRAINING: ("学習設定", "学習設定を作成し、再現可能な実験として記録します。"),
    PageId.TRAINING_QUEUE: (
        "学習キュー",
        "学習する設定を並べ、順番に実行します。実行前に表で設定を変更できます。",
    ),
    PageId.EXPERIMENTS: ("実験一覧", "実験の状態、設定、学習結果を確認します。"),
    PageId.CANDIDATES: ("リリース候補", "検証用データセットを切り替えて候補を比較します。"),
    PageId.MASK_COMPARISON: ("マスク比較", "候補モデルの予測結果を比較します。"),
    PageId.RELEASED_MODELS: (
        "リリース済みモデル・振り分け",
        "リリース済みモデルと分類の振り分けを管理します。",
    ),
    PageId.INFERENCE: ("本番推論", "画像分類に応じたリリース済みモデルで推論します。"),
}


class ModeWindow(QMainWindow):
    """モード内タブと共通ステータスを持つ。"""

    closed = Signal(object)
    home_requested = Signal()
    tab_requested = Signal(object)
    activated = Signal(object)

    def __init__(self, mode: ModeId, ctx: AppContext, page_ids: list[PageId], parent=None) -> None:
        super().__init__(parent)
        self.mode = ModeId(mode)
        self.ctx = ctx
        self.page_ids = page_ids
        self._page_widgets = {}
        self._page_menu_actions = {}
        self._page_action_owners = {}
        self._page_action_guard = set()
        self._generated_page_actions = {}
        self._menus = {}
        self.setWindowTitle(f"{MODE_LABELS[self.mode]} — 気泡インスタンスセグメンテーション")
        self.setMinimumSize(900, 600)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        topbar = QWidget()
        topbar.setObjectName("modeTopbar")
        top = QHBoxLayout(topbar)
        top.setContentsMargins(16, 0, 16, 0)
        top.setSpacing(4)
        self.home_button = LayoutButton()
        self.home_button.setProperty("role", "ghost")
        home_layout = QHBoxLayout(self.home_button)
        home_layout.setContentsMargins(4, 2, 4, 2)
        home_layout.setSpacing(6)
        self.home_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        home_label = QLabel("⌂ ホーム")
        home_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        home_layout.addWidget(home_label)
        self.home_button.setToolTip(
            f"ホームへ戻る（{ctx.shortcuts.display_key(ctx.shortcuts['home'])}）"
        )
        self.home_button.clicked.connect(self.home_requested.emit)
        top.addWidget(self.home_button, 0, Qt.AlignmentFlag.AlignLeft)
        top.addSpacing(16)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("modeTabs")
        self.tabs.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.tabs.setAutoFillBackground(False)
        self.tab_bar = QTabBar(topbar)
        self.tab_bar.setObjectName("modeTabBar")
        self.tab_bar.setDrawBase(False)
        self.tab_bar.setExpanding(False)
        self.tab_bar.setVisible(self.mode != ModeId.INFERENCE)
        for page_id in self.page_ids:
            title, description = MODE_TAB_LABELS[page_id]
            self.tabs.addTab(QWidget(), title)
            self.tabs.setTabToolTip(self.tabs.count() - 1, description)
            self.tab_bar.addTab(title)
            self.tab_bar.setTabToolTip(self.tab_bar.count() - 1, description)
        top.addWidget(self.tab_bar, 1)
        self.tab_tools = QWidget(topbar)
        self.tab_tools_layout = QHBoxLayout(self.tab_tools)
        self.tab_tools_layout.setContentsMargins(0, 0, 0, 0)
        self.tab_tools_layout.setSpacing(8)
        top.addWidget(self.tab_tools)
        layout.addWidget(topbar)
        self.stack = QStackedWidget()
        self.tabs.currentChanged.connect(self._select_tab)
        self.tabs.currentChanged.connect(self.tab_bar.setCurrentIndex)
        self.tab_bar.currentChanged.connect(self._select_requested_tab)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.status_text = QLabel("準備完了")
        self.job_count = QLabel()
        self.job_count.setFont(numeric_font())
        self.autosave_text = QLabel()
        self.autosave_text.setFont(numeric_font())
        self.statusBar().addWidget(self.status_text, 1)
        self.statusBar().addPermanentWidget(self.job_count)
        self.statusBar().addPermanentWidget(self.autosave_text)
        self.ctx.status.message.connect(self.status_text.setText)
        self.ctx.status.saved.connect(self._update_saved_time)
        self.ctx.jobs.jobs_changed.connect(self._update_job_count)
        self._update_job_count(self.ctx.jobs.running_count)
        self._update_saved_time(ctx.backend.get_last_saved_at())
        self._install_home_shortcut()
        self.ctx.shortcuts.changed.connect(self._shortcuts_changed)
        self._build_menus()
        self._help_shortcut = QShortcut(QKeySequence(self.ctx.shortcuts["help"]), self)
        self._help_shortcut.activated.connect(self._open_keymap)
        self._f1_shortcut = QShortcut(QKeySequence("F1"), self)
        self._f1_shortcut.activated.connect(self._open_keymap)

    def _build_menus(self) -> None:
        if self.mode == ModeId.INFERENCE:
            help_menu = self.menuBar().addMenu("ヘルプ(&H)")
            self._help_menu_action = self._add_action(
                help_menu, "キー割り当て一覧…", self._open_keymap, "help"
            )
            return
        menu_specs = [("file", "ファイル(&F)")]
        if self.mode in (ModeId.DATA_PREPARATION, ModeId.TRAINING):
            menu_specs.append(("edit", "編集(&E)"))
        menu_specs.append(("view", "表示(&V)"))
        for key, title in menu_specs:
            self._menus[key] = self.menuBar().addMenu(title)
        if self.mode == ModeId.DATA_PREPARATION:
            self._menus["dataset"] = self.menuBar().addMenu("データセット(&D)")
        elif self.mode == ModeId.TRAINING:
            self._menus["training"] = self.menuBar().addMenu("学習(&L)")
            self.stop_training_action = self._add_action(
                self._menus["training"], "■ 今すぐ停止", self._stop_training_now
            )
            self.stop_training_action.setToolTip(
                "今の学習をすぐに止め、キューも止めます。今の学習は「中断」になり、"
                "途中までの結果だけが残ります。止める前に確認します。"
            )
            self.ctx.queue_controller.changed.connect(self._update_stop_training_action)
            self.ctx.training_runner.busy_changed.connect(self._update_stop_training_action)
            self._update_stop_training_action()
        elif self.mode == ModeId.COMPARISON:
            self._menus["candidate"] = self.menuBar().addMenu("候補(&C)")
            self._menus["release"] = self.menuBar().addMenu("リリース(&R)")
        self._menus["tools"] = self.menuBar().addMenu("ツール(&T)")
        self._menus["help"] = self.menuBar().addMenu("ヘルプ(&H)")
        self._file_common_separator = self._menus["file"].addSeparator()
        self._home_menu_action = self._add_action(
            self._menus["file"],
            f"ホームに戻る\t{self.ctx.shortcuts.display_key(self.ctx.shortcuts['home'])}",
            self.home_requested.emit,
        )
        self._add_action(self._menus["file"], "ウィンドウを閉じる", self.close)
        if self.mode == ModeId.TRAINING:
            self._tools_keymap_separator = self._menus["tools"].addSeparator()
        self._tools_keymap_action = self._add_action(
            self._menus["tools"], "キー割り当て…", self._open_keymap
        )
        self._display_menu = None
        if self.mode in (ModeId.DATA_PREPARATION, ModeId.COMPARISON):
            self._display_menu = self._menus["tools"].addMenu("原画像と切り替える表示")
            self._display_actions = {}
            for name in sorted(self.ctx.display.MODES):
                action = self._display_menu.addAction(name)
                action.setCheckable(True)
                action.setChecked(name == self.ctx.display.value)
                action.triggered.connect(
                    lambda _checked=False, value=name: self.ctx.display.set_value(value)
                )
                self._display_actions[name] = action
            self.ctx.display.changed.connect(self._display_changed)
        self._help_menu_action = self._add_action(
            self._menus["help"], "キー割り当て一覧…", self._open_keymap, "help"
        )
        self._tab_menu_actions = {}
        for target in self.page_ids:
            action = self._menus["view"].addAction(MODE_TAB_LABELS[target][0])
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=target: self.tab_requested.emit(value)
            )
            self._tab_menu_actions[target] = action
        self._view_page_separator = self._menus["view"].addSeparator()
        for menu in self._menus.values():
            menu.aboutToShow.connect(lambda target=menu: self._normalize_menu_tree(target))

    def _normalize_menu_tree(self, menu) -> None:
        """Remove leading, trailing, and repeated separators throughout a menu tree."""
        previous_item = False
        pending_separator = None
        for action in menu.actions():
            if action.isSeparator():
                if previous_item and pending_separator is None:
                    pending_separator = action
                else:
                    menu.removeAction(action)
                continue
            if not action.isVisible():
                continue
            if pending_separator is not None:
                pending_separator.setVisible(True)
                pending_separator = None
            previous_item = True
            if action.menu():
                self._normalize_menu_tree(action.menu())
        if pending_separator is not None:
            menu.removeAction(pending_separator)

    def _update_stop_training_action(self, *_args) -> None:
        controller = self.ctx.queue_controller
        can_stop = (
            controller.executing
            or controller.waiting_for_training
            or self.ctx.training_runner.is_busy
        )
        self.stop_training_action.setEnabled(can_stop)
        if not can_stop:
            self.stop_training_action.setToolTip("学習を実行していないときは使えません")
        else:
            self.stop_training_action.setToolTip(
                "今の学習をすぐに止め、キューも止めます。今の学習は「中断」になり、"
                "途中までの結果だけが残ります。止める前に確認します。"
            )

    def _stop_training_now(self) -> None:
        controller = self.ctx.queue_controller
        runner = self.ctx.training_runner
        if not (controller.executing or controller.waiting_for_training or runner.is_busy):
            return
        if (
            QMessageBox.question(
                self,
                "今すぐ停止",
                "今の学習をすぐに止め、キューも止めますか？\n"
                "今の学習は「中断」になり、途中までの結果だけが残ります。",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        if controller.executing or controller.waiting_for_training:
            controller.stop_now()
        elif runner.is_busy:
            runner.request_stop("user_stop")

    def _display_changed(self, name: str) -> None:
        for value, action in self._display_actions.items():
            action.setChecked(value == name)

    def _add_action(self, menu, label, callback=None, shortcut_key=None):
        action = QAction(label, self)
        if shortcut_key:
            action.setText(
                f"{label}\t{self.ctx.shortcuts.display_key(self.ctx.shortcuts[shortcut_key])}"
            )
            action.triggered.connect(callback)
        elif callback:
            action.triggered.connect(callback)
        menu.addAction(action)
        return action

    def _clear_tab_tools(self) -> None:
        while self.tab_tools_layout.count():
            item = self.tab_tools_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

    def _select_tab(self, index: int) -> None:
        if 0 <= index < len(self.page_ids):
            page_id = self.page_ids[index]
            widget = self._page_widgets.get(page_id)
            if widget:
                self.stack.setCurrentWidget(widget)
            self._install_page_menu(page_id)
            self._install_tab_tools(page_id)

    def _install_page_menu(self, page_id: PageId) -> None:
        menus = list(self._menus.values())
        while menus:
            menu = menus.pop()
            menu.setToolTipsVisible(True)
            menus.extend(action.menu() for action in menu.actions() if action.menu() is not None)
        for owner_page in self.page_ids:
            for menu_name, actions in self._page_menu_actions.get(owner_page, {}).items():
                menu = self._menus.get(menu_name)
                if menu is None or not actions:
                    continue
                active = owner_page == page_id
                tab_label = MODE_TAB_LABELS[owner_page][0]
                for action in actions:
                    if action is None:
                        continue
                    if active:
                        prior_tooltip = action.property("_page_action_tooltip")
                        if prior_tooltip is not None:
                            self._set_page_action(
                                action,
                                enabled=bool(action.property("_page_action_enabled")),
                                tooltip=str(prior_tooltip),
                            )
                    else:
                        if action.property("_page_action_enabled") is None:
                            action.setProperty("_page_action_enabled", action.isEnabled())
                            action.setProperty("_page_action_tooltip", action.toolTip())
                        self._set_page_action(
                            action,
                            enabled=False,
                            tooltip=f"{tab_label}タブで利用できます",
                        )
        page = self._page_widgets.get(page_id)
        refresh_menu_actions = getattr(page, "refresh_menu_actions", None)
        if callable(refresh_menu_actions):
            refresh_menu_actions()
        elif callable(getattr(page, "_update_buttons", None)):
            page._update_buttons()
        elif callable(getattr(page, "_update_thumbnail_action", None)):
            page._update_thumbnail_action()
        elif callable(getattr(page, "_show_model_detail", None)):
            page._show_model_detail()
            update_routing = getattr(page, "_update_routing_rows", None)
            if callable(update_routing):
                update_routing()
        for target, action in getattr(self, "_tab_menu_actions", {}).items():
            action.setChecked(target == page_id)
        for menu in self._menus.values():
            self._normalize_menu_tree(menu)

    def _set_page_action(self, action: QAction, *, enabled: bool, tooltip: str) -> None:
        """ページ状態と現在タブによる表示状態を分けて適用する。"""
        self._page_action_guard.add(action)
        try:
            action.setEnabled(enabled)
            action.setToolTip(tooltip)
        finally:
            self._page_action_guard.discard(action)

    def set_page_action_enabled(self, action: QAction, enabled: bool) -> None:
        """ページ本来の有効状態を記録し、現在タブの可否と合わせて適用する。"""
        owner = self._page_action_owners.get(action)
        if owner is None:
            action.setEnabled(enabled)
            return
        action.setProperty("_page_action_enabled", bool(enabled))
        active_page = self.page_ids[self.tabs.currentIndex()]
        effective = bool(enabled) and active_page == owner
        tooltip = action.toolTip()
        if active_page != owner:
            tooltip = f"{MODE_TAB_LABELS[owner][0]}タブで利用できます"
        self._set_page_action(action, enabled=effective, tooltip=tooltip)

    def _register_page_actions(self, page_id: PageId, groups: dict) -> None:
        """ページ QAction の状態変更を監視し、非表示タブでは無効を保つ。"""
        pending = [action for actions in groups.values() for action in actions if action]
        while pending:
            action = pending.pop()
            if action in self._page_action_owners:
                continue
            self._page_action_owners[action] = page_id
            action.setProperty("_page_action_enabled", action.isEnabled())
            action.setProperty("_page_action_tooltip", action.toolTip())
            action.changed.connect(lambda a=action: self._page_action_changed(a))
            if action.menu():
                pending.extend(action.menu().actions())

    def _page_action_changed(self, action: QAction) -> None:
        """ページが共有 QAction を更新した意図を記録して、タブ可否を再適用する。"""
        if action in self._page_action_guard:
            return
        owner = self._page_action_owners.get(action)
        if owner is None:
            return
        inactive_tooltip = f"{MODE_TAB_LABELS[owner][0]}タブで利用できます"
        if action.toolTip() != inactive_tooltip:
            action.setProperty("_page_action_tooltip", action.toolTip())
        active_page = self.page_ids[self.tabs.currentIndex()]
        if active_page != owner:
            if action.isEnabled():
                action.setProperty("_page_action_enabled", True)
            self._set_page_action(
                action,
                enabled=False,
                tooltip=inactive_tooltip,
            )
        else:
            action.setProperty("_page_action_enabled", action.isEnabled())

    def _install_tab_tools(self, page_id: PageId) -> None:
        self._clear_tab_tools()
        page = self._page_widgets.get(page_id)
        factory = getattr(page, "tab_tools_widget", None) if page else None
        widget = factory() if factory else None
        if widget:
            self.tab_tools_layout.addWidget(widget)

    def _install_home_shortcut(self) -> None:
        self.home_shortcut = QShortcut(QKeySequence(self.ctx.shortcuts["home"]), self)
        self.home_shortcut.activated.connect(self.home_requested.emit)

    def _shortcuts_changed(self) -> None:
        """共有キー割り当てをホーム操作へ反映する。"""
        self.home_shortcut.setKey(QKeySequence(self.ctx.shortcuts["home"]))
        self._help_shortcut.setKey(QKeySequence(self.ctx.shortcuts["help"]))
        self.home_button.setToolTip(
            f"ホームへ戻る（{self.ctx.shortcuts.display_key(self.ctx.shortcuts['home'])}）"
        )
        if self.mode != ModeId.INFERENCE:
            self._install_page_menu(self.page_ids[self.tabs.currentIndex()])
            self._home_menu_action.setText(
                f"ホームに戻る\t{self.ctx.shortcuts.display_key(self.ctx.shortcuts['home'])}"
            )
        self._help_menu_action.setText(
            f"キー割り当て一覧…\t{self.ctx.shortcuts.display_key(self.ctx.shortcuts['help'])}"
        )

    def _open_keymap(self) -> None:
        """キー割り当て一覧を前面に表示する。"""
        from .keymap_dialog import show_keymap_window

        show_keymap_window(self, self.ctx)

    def add_page(self, page_id: PageId, page: QWidget, title: str, description: str) -> None:
        """ページをタブとして追加する。"""
        index = self.page_ids.index(PageId(page_id))
        self.stack.insertWidget(index, page)
        self._page_widgets[PageId(page_id)] = page
        provider = getattr(page, "menu_action_groups", None)
        if provider is None:
            provider = getattr(page, "menu_actions", lambda: {})
        self._page_menu_actions[PageId(page_id)] = provider() if callable(provider) else provider
        self._register_page_actions(PageId(page_id), self._page_menu_actions[PageId(page_id)])
        self._rebuild_page_menus()
        self.tabs.setTabText(index, title)
        self.tabs.setTabToolTip(index, description)
        self.tab_bar.setTabText(index, title)
        self.tab_bar.setTabToolTip(index, description)
        if index == self.tabs.currentIndex():
            self._install_page_menu(PageId(page_id))
            self._install_tab_tools(PageId(page_id))

    @staticmethod
    def _append_token(menu, token, before=None):
        if token is None:
            action = QAction(menu)
            action.setSeparator(True)
        else:
            action = token
            action.setStatusTip("")
        if before is None:
            menu.addAction(action)
        else:
            menu.insertAction(before, action)
        return action

    def _rebuild_page_menus(self) -> None:
        """Reassemble page-owned actions so empty groups never leave separators."""
        for menu_name, generated in self._generated_page_actions.items():
            menu = self._menus.get(menu_name)
            if menu is None:
                continue
            for action in generated:
                menu.removeAction(action)
                if action.isSeparator():
                    action.deleteLater()
        self._generated_page_actions = {}

        for menu_name, menu in self._menus.items():
            tokens = []
            for page_id in self.page_ids:
                for token in self._page_menu_actions.get(page_id, {}).get(menu_name, []):
                    if token is None:
                        if tokens and tokens[-1] is not None:
                            tokens.append(None)
                    else:
                        tokens.append(token)
            while tokens and tokens[-1] is None:
                tokens.pop()
            if not tokens:
                continue
            before = (
                self._file_common_separator
                if menu_name == "file"
                else self._tools_keymap_separator
                if menu_name == "tools" and self.mode == ModeId.TRAINING
                else None
            )
            generated = []
            for token in tokens:
                generated.append(self._append_token(menu, token, before))
            self._generated_page_actions[menu_name] = generated

    def select_page(self, page_id: PageId) -> None:
        """指定ページのタブを選択する。"""
        index = self.page_ids.index(PageId(page_id))
        self.tab_bar.blockSignals(True)
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(index)
        self.tab_bar.setCurrentIndex(index)
        self.tabs.blockSignals(False)
        self.tab_bar.blockSignals(False)
        widget = self._page_widgets.get(PageId(page_id))
        if widget:
            self.stack.setCurrentWidget(widget)
        self._install_page_menu(PageId(page_id))
        self._install_tab_tools(PageId(page_id))

    def _select_requested_tab(self, index: int) -> None:
        """利用者が押した未生成タブを管理側へ通知する。"""
        if 0 <= index < len(self.page_ids):
            self.tab_requested.emit(self.page_ids[index])

    def _update_job_count(self, count: int) -> None:
        self.job_count.setText(running_jobs_label(count))

    def _update_saved_time(self, value: datetime) -> None:
        self.autosave_text.setText(autosave_label(value))

    def event(self, event) -> bool:
        """タスクバー等から前面に戻ったことを管理側へ通知する。"""
        result = super().event(event)
        if event.type() == QEvent.Type.WindowActivate:
            self.activated.emit(self.mode)
            if self.mode != ModeId.INFERENCE:
                self._install_page_menu(self.page_ids[self.tabs.currentIndex()])
        return result

    def closeEvent(self, event) -> None:
        """閉じたウィンドウを管理側へ通知する。"""
        self.closed.emit(self.mode)
        super().closeEvent(event)
