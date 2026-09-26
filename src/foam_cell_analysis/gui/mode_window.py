"""単一モードのページを表示するウィンドウ。"""

from datetime import datetime

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
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
        help_menu = self.menuBar().addMenu("ヘルプ")
        help_menu.addAction("キー割り当て一覧…", self._open_keymap)
        self._help_shortcut = QShortcut(QKeySequence(self.ctx.shortcuts["help"]), self)
        self._help_shortcut.activated.connect(self._open_keymap)
        self._f1_shortcut = QShortcut(QKeySequence("F1"), self)
        self._f1_shortcut.activated.connect(self._open_keymap)

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

    def _open_keymap(self) -> None:
        """キー割り当て一覧を前面に表示する。"""
        from .keymap_dialog import show_keymap_window

        show_keymap_window(self, self.ctx)

    def add_page(self, page_id: PageId, page: QWidget, title: str, description: str) -> None:
        """ページをタブとして追加する。"""
        index = self.page_ids.index(PageId(page_id))
        self.stack.insertWidget(index, page)
        self._page_widgets[PageId(page_id)] = page
        self.tabs.setTabText(index, title)
        self.tabs.setTabToolTip(index, description)
        self.tab_bar.setTabText(index, title)
        self.tab_bar.setTabToolTip(index, description)

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

    def _select_tab(self, index: int) -> None:
        if 0 <= index < len(self.page_ids):
            widget = self._page_widgets.get(self.page_ids[index])
            if widget:
                self.stack.setCurrentWidget(widget)

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
        return result

    def closeEvent(self, event) -> None:
        """閉じたウィンドウを管理側へ通知する。"""
        self.closed.emit(self.mode)
        super().closeEvent(event)
