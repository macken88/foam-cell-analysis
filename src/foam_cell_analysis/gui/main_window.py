"""サイドバー、ページスタック、ステータスを持つメイン画面。"""

from datetime import datetime
from typing import ClassVar

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)

from .context import AppContext
from .modes.comparison.candidates_page import CandidatesPage
from .modes.comparison.mask_compare import MaskComparisonPage
from .modes.data_preparation.page import DataPreparationPage
from .modes.inference.page import InferencePage
from .modes.release.page import ReleasedModelsPage
from .modes.training.experiment_list import ExperimentListPage
from .modes.training.page import TrainingPage
from .navigation import PageId
from .theme import Color, numeric_font


class MainWindow(QMainWindow):
    """全ページを登録し、サイドバーから切り替える。"""

    page_labels: ClassVar[dict[PageId, str]] = {
        PageId.DATA_PREPARATION: "データ準備",
        PageId.TRAINING: "モデル学習",
        PageId.EXPERIMENTS: "実験一覧",
        PageId.CANDIDATES: "モデル比較・リリース",
        PageId.RELEASED_MODELS: "リリース済みモデル・振り分け",
        PageId.INFERENCE: "本番推論",
        PageId.MASK_COMPARISON: "マスク比較",
    }

    def __init__(self, ctx: AppContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("気泡インスタンスセグメンテーション（モック動作中）")
        self.setMinimumSize(1280, 800)
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("mainSidebar")
        self.stack = QStackedWidget()
        self.pages = {}
        self.sidebar.setFixedWidth(220)
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(self.sidebar)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.sidebar.addItem(self._header("管理者機能"))
        for page_id in [
            PageId.DATA_PREPARATION,
            PageId.TRAINING,
            PageId.EXPERIMENTS,
            PageId.CANDIDATES,
            PageId.RELEASED_MODELS,
        ]:
            self._add_page(page_id)
        self.sidebar.addItem(self._header("利用者機能"))
        self._add_page(PageId.INFERENCE)
        for page_id, page_type in [
            (PageId.DATA_PREPARATION, DataPreparationPage),
            (PageId.TRAINING, TrainingPage),
            (PageId.EXPERIMENTS, ExperimentListPage),
            (PageId.CANDIDATES, CandidatesPage),
            (PageId.MASK_COMPARISON, MaskComparisonPage),
            (PageId.RELEASED_MODELS, ReleasedModelsPage),
            (PageId.INFERENCE, InferencePage),
        ]:
            page = page_type(ctx)
            self.pages[page_id] = page
            self.stack.addWidget(page)
        self.sidebar.currentRowChanged.connect(self._sidebar_changed)
        self.ctx.navigator.navigation_requested.connect(self.navigate)
        self.status_text = QLabel("準備完了")
        self.job_count = QLabel("実行中ジョブ: 0")
        self.job_count.setFont(numeric_font())
        self.statusBar().addWidget(self.status_text, 1)
        self.statusBar().addPermanentWidget(self.job_count)
        self.ctx.jobs.jobs_changed.connect(
            lambda count: self.job_count.setText(f"実行中ジョブ: {count}")
        )
        self.autosave_text = QLabel(
            f"自動保存 {ctx.backend.get_last_saved_at().astimezone().strftime('%H:%M:%S')}"
        )
        self.autosave_text.setFont(numeric_font())
        self.statusBar().addPermanentWidget(self.autosave_text)
        self.ctx.status.message.connect(self.status_text.setText)
        self.ctx.status.saved.connect(self._update_saved_time)
        file_menu = self.menuBar().addMenu("ファイル")
        file_menu.addAction("終了", self.close)
        help_menu = self.menuBar().addMenu("ヘルプ")
        help_menu.addAction(
            "バージョン情報",
            lambda: QMessageBox.about(
                self, "バージョン情報", "気泡インスタンスセグメンテーション 0.1.0（モック）"
            ),
        )
        self._navigate(PageId.DATA_PREPARATION, {})

    @staticmethod
    def _header(label: str) -> QListWidgetItem:
        item = QListWidgetItem(label)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
        item.setForeground(QColor(Color.SLATE))
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        return item

    def _add_page(self, page_id: PageId) -> None:
        item = QListWidgetItem(self.page_labels[page_id])
        item.setData(256, page_id)
        self.sidebar.addItem(item)

    def _sidebar_changed(self, row: int) -> None:
        item = self.sidebar.item(row) if row >= 0 else None
        if item and item.data(256):
            self.navigate(item.data(256), {})

    def navigate(self, page_id: PageId, params: dict | None = None) -> None:
        """Navigator からの遷移要求を処理する。"""
        self._navigate(PageId(page_id), params or {})

    def _navigate(self, page_id: PageId, params: dict) -> None:
        self.stack.setCurrentWidget(self.pages[page_id])
        sidebar_id = PageId.CANDIDATES if page_id == PageId.MASK_COMPARISON else page_id
        for row in range(self.sidebar.count()):
            item = self.sidebar.item(row)
            if item.data(256) == sidebar_id:
                self.sidebar.blockSignals(True)
                self.sidebar.setCurrentRow(row)
                self.sidebar.blockSignals(False)
                break
        self.pages[page_id].on_enter(params)

    def _update_saved_time(self, value: datetime) -> None:
        """保存時刻をステータスバーに表示する。"""
        self.autosave_text.setText(f"自動保存 {value.astimezone().strftime('%H:%M:%S')}")
