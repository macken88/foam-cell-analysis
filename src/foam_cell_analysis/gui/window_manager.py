"""ページ遷移をモードウィンドウへ振り分ける。"""

from time import monotonic

from PySide6.QtCore import QObject, QSettings, Signal

from .context import AppContext
from .mode_window import ModeWindow
from .navigation import ModeId, PageId
from .settings import app_settings

PAGE_TO_MODE_TAB = {
    PageId.DATA_PREPARATION: (ModeId.DATA_PREPARATION, 0),
    PageId.DATASET_HISTORY: (ModeId.DATA_PREPARATION, 1),
    PageId.TRAINING: (ModeId.TRAINING, 0),
    PageId.TRAINING_QUEUE: (ModeId.TRAINING, 1),
    PageId.EXPERIMENTS: (ModeId.TRAINING, 2),
    PageId.CANDIDATES: (ModeId.COMPARISON, 0),
    PageId.MASK_COMPARISON: (ModeId.COMPARISON, 1),
    PageId.RELEASED_MODELS: (ModeId.COMPARISON, 2),
    PageId.INFERENCE: (ModeId.INFERENCE, 0),
}
MODE_PAGES = {
    ModeId.DATA_PREPARATION: [PageId.DATA_PREPARATION, PageId.DATASET_HISTORY],
    ModeId.TRAINING: [PageId.TRAINING, PageId.TRAINING_QUEUE, PageId.EXPERIMENTS],
    ModeId.COMPARISON: [PageId.CANDIDATES, PageId.MASK_COMPARISON, PageId.RELEASED_MODELS],
    ModeId.INFERENCE: [PageId.INFERENCE],
}
PAGE_LABELS = {
    PageId.DATA_PREPARATION: (
        "作業中データ",
        "作業データを編集し、整合性を確認してデータセット版を確定します。",
    ),
    PageId.DATASET_HISTORY: ("データセット版履歴", "確定済みの学習用版・検証用版を確認します。"),
    PageId.TRAINING: ("学習設定", "学習設定を作成し、再現可能な実験として記録します。"),
    PageId.TRAINING_QUEUE: (
        "学習キュー",
        "学習する設定を並べ、順番に実行します。実行前に表で設定を変更できます。",
    ),
    PageId.EXPERIMENTS: ("実験一覧", "実験の状態、設定、学習結果を確認します。"),
    PageId.CANDIDATES: ("リリース候補", "検証用データセットを切り替えて候補を比較します。"),
    PageId.MASK_COMPARISON: ("抽出結果比較", "候補モデルの予測結果を比較します。"),
    PageId.RELEASED_MODELS: (
        "リリース済みモデル・振り分け",
        "リリース済みモデルと分類の振り分けを管理します。",
    ),
    PageId.INFERENCE: ("本番推論", "画像分類に応じたリリース済みモデルで推論します。"),
}
PAGE_TYPES = {}


class WindowManager(QObject):
    """ページを必要時に作り、モードごとに1つのウィンドウで管理する。"""

    home_requested = Signal()
    mode_closed = Signal(object)

    def __init__(self, ctx: AppContext, settings: QSettings | None = None, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        if self.ctx.queue_controller is None:
            from .modes.training.queue_controller import TrainingQueueController

            self.ctx.queue_controller = TrainingQueueController(ctx, self)
        self.settings = settings or app_settings()
        self._windows: dict[ModeId, ModeWindow] = {}
        self._pages = {}
        self._current: dict[ModeId, PageId] = {}
        self._last_page: PageId | None = None
        self._last_status = "準備完了"
        self._last_navigation_at: dict[ModeId, float] = {}
        self._last_activation_refresh_at: dict[ModeId, float] = {}
        self.ctx.navigator.navigation_requested.connect(self.navigate)
        self.ctx.status.message.connect(self._remember_status)

    def _remember_status(self, message: str) -> None:
        self._last_status = message

    def page(self, page_id: PageId):
        """ページのインスタンスを返し、必要なら遅延生成する。"""
        page_id = PageId(page_id)
        if page_id not in self._pages:
            if not PAGE_TYPES:
                from .modes.comparison.candidates_page import CandidatesPage
                from .modes.comparison.mask_compare import MaskComparisonPage
                from .modes.data_preparation.page import DataPreparationPage, DatasetHistoryPage
                from .modes.inference.page import InferencePage
                from .modes.release.page import ReleasedModelsPage
                from .modes.training.experiment_list import ExperimentListPage
                from .modes.training.page import TrainingPage
                from .modes.training.queue_page import TrainingQueuePage

                PAGE_TYPES.update(
                    {
                        PageId.DATA_PREPARATION: DataPreparationPage,
                        PageId.DATASET_HISTORY: DatasetHistoryPage,
                        PageId.TRAINING: TrainingPage,
                        PageId.TRAINING_QUEUE: TrainingQueuePage,
                        PageId.EXPERIMENTS: ExperimentListPage,
                        PageId.CANDIDATES: CandidatesPage,
                        PageId.MASK_COMPARISON: MaskComparisonPage,
                        PageId.RELEASED_MODELS: ReleasedModelsPage,
                        PageId.INFERENCE: InferencePage,
                    }
                )
            mode, _ = PAGE_TO_MODE_TAB[page_id]
            window = self._ensure_window(mode)
            page = PAGE_TYPES[page_id](self.ctx, show_heading=False)
            self._pages[page_id] = page
            title, description = PAGE_LABELS[page_id]
            window.add_page(page_id, page, title, description)
        return self._pages[page_id]

    def _ensure_window(self, mode: ModeId) -> ModeWindow:
        mode = ModeId(mode)
        if mode not in self._windows:
            window = ModeWindow(mode, self.ctx, MODE_PAGES[mode])
            window.status_text.setText(self._last_status)
            window.home_requested.connect(self.show_home_requested)
            window.tab_requested.connect(lambda page_id: self.navigate(page_id, {}))
            window.closed.connect(self._save_window)
            window.activated.connect(self._refresh_active_page)
            self._windows[mode] = window
            geometry = self.settings.value(f"windows/{mode.value}/geometry")
            if geometry:
                window._saved_geometry = tuple(int(value) for value in geometry)
                window._saved_maximized = self.settings.value(
                    f"windows/{mode.value}/maximized", False, type=bool
                )
        return self._windows[mode]

    def window(self, mode: ModeId) -> ModeWindow | None:
        """モードウィンドウを返す。未生成の場合は None。"""
        return self._windows.get(ModeId(mode))

    def current_page_id(self, mode: ModeId) -> PageId | None:
        """モードごとの現在のページを返す。"""
        return self._current.get(ModeId(mode))

    def _refresh_active_page(self, mode: ModeId) -> None:
        """OS から前面に戻った画面を軽量な再表示更新で同期する。"""
        mode = ModeId(mode)
        now = monotonic()
        if now - self._last_navigation_at.get(mode, 0.0) < 0.5:
            return
        if now - self._last_activation_refresh_at.get(mode, 0.0) < 1.0:
            return
        page_id = self._current.get(mode)
        page = self._pages.get(page_id)
        if page:
            self._last_activation_refresh_at[mode] = now
            page.refresh_on_activate()

    def navigate(self, page_id: PageId, params: dict | None = None) -> None:
        """遷移先のウィンドウを開き、タブ選択後に on_enter を1回呼ぶ。"""
        page_id = PageId(page_id)
        mode, _tab = PAGE_TO_MODE_TAB[page_id]
        page = self.page(page_id)
        for sibling in MODE_PAGES[mode]:
            if sibling not in self._pages:
                self.page(sibling)
        window = self._ensure_window(mode)
        self._last_navigation_at[mode] = monotonic()
        keep_current = bool((params or {}).get("_preserve_current_tab")) and window.isVisible()
        if keep_current:
            page_id = self._current.get(mode, page_id)
            page = self.page(page_id)
        else:
            window.select_page(page_id)
            self._current[mode] = page_id
            self._last_page = page_id
        if not window.isVisible():
            saved_geometry = getattr(window, "_saved_geometry", None)
            if saved_geometry:
                window.show()
                window.setGeometry(*saved_geometry)
                if getattr(window, "_saved_maximized", False):
                    window.showMaximized()
                del window._saved_geometry
                if hasattr(window, "_saved_maximized"):
                    del window._saved_maximized
            else:
                window.showMaximized()
        else:
            window.show()
        window.raise_()
        window.activateWindow()
        if not keep_current:
            page.on_enter(
                {
                    key: value
                    for key, value in (params or {}).items()
                    if key != "_preserve_current_tab"
                }
            )
        else:
            page.refresh_on_activate()

    def show_home_requested(self) -> None:
        """Ctrl+H またはホームボタンをホーム側へ通知する。"""
        self.home_requested.emit()

    def set_home_callback(self, callback) -> None:
        """ホームを前面に出すコールバックを登録する。"""
        self.home_requested.connect(callback)

    def _save_window(self, mode: ModeId) -> None:
        window = self._windows.get(ModeId(mode))
        if window:
            self._store_geometry(mode, window)
        self.mode_closed.emit(ModeId(mode))

    def _store_geometry(self, mode: ModeId, window: ModeWindow) -> None:
        rect = window.geometry()
        self.settings.setValue(
            f"windows/{mode.value}/geometry",
            [rect.x(), rect.y(), rect.width(), rect.height()],
        )
        self.settings.setValue(f"windows/{mode.value}/maximized", window.isMaximized())
        window._saved_geometry = (rect.x(), rect.y(), rect.width(), rect.height())
        window._saved_maximized = window.isMaximized()
        self.settings.sync()

    def open_modes(self) -> list[ModeId]:
        """現在表示中のモードを返す。"""
        return [mode for mode, window in self._windows.items() if window.isVisible()]

    def save_all_windows(self) -> None:
        """アプリ終了前にモードウィンドウの位置とサイズを保存する。"""
        for mode, window in self._windows.items():
            self._store_geometry(mode, window)
