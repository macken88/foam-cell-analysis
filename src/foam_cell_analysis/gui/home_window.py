"""起動時に表示するホームウィンドウ。"""

import os
from collections import deque
from dataclasses import replace
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .context import AppContext
from .home_summary import HomeSummary, build_home_summary
from .labels import running_jobs_label
from .navigation import ModeId, PageId
from .theme import Color, body_font, numeric_font, set_style
from .widgets.marks import display_mode_label
from .window_manager import WindowManager


class ClickablePanel(QFrame):
    """マウスとキーボードで開けるホームパネル。"""

    activated = Signal(object)

    def __init__(self, target: object, object_name: str, parent=None) -> None:
        super().__init__(parent)
        self.target = target
        self.setObjectName(object_name)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.activated.emit(self.target)
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.activated.emit(self.target)
            event.accept()
            return
        super().keyPressEvent(event)


class PipelineWidget(QFrame):
    """1つの白いパネルに工程と接続線を描く。"""

    activated = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pipelinePanel")
        self.setProperty("role", "panel")
        self.summary: HomeSummary | None = None
        self._pulse = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._stages: list[ClickablePanel] = []
        self._values: list[QLabel] = []
        self._details: list[QLabel] = []
        self._notes: list[QLabel] = []
        self._data_errors: QLabel | None = None
        stages = (
            ("データ準備", ModeId.DATA_PREPARATION),
            ("モデル学習", ModeId.TRAINING),
            ("モデル比較・リリース", ModeId.COMPARISON),
        )
        for index, (title, mode) in enumerate(stages):
            stage = ClickablePanel(mode, "pipelineStage")
            if index == len(stages) - 1:
                stage.setProperty("lastStage", True)
            content = QVBoxLayout(stage)
            content.setContentsMargins(16, 26, 16, 14)
            content.setSpacing(3)
            heading = QLabel(title)
            heading.setFont(body_font(11))
            heading.setStyleSheet("font-weight: 600")
            value_row = QHBoxLayout()
            value_row.setContentsMargins(0, 0, 0, 0)
            value_row.setSpacing(5)
            value = QLabel()
            value.setFont(numeric_font(22))
            detail = QLabel()
            detail.setFont(body_font(9))
            detail.setStyleSheet(f"color: {Color.SLATE}")
            detail.setWordWrap(True)
            value_row.addWidget(value, 0, Qt.AlignmentFlag.AlignVCenter)
            value_row.addWidget(detail, 1, Qt.AlignmentFlag.AlignVCenter)
            note = QLabel()
            note.setFont(body_font(9))
            note.setStyleSheet(f"color: {Color.SLATE}")
            note.setWordWrap(True)
            note.setMinimumHeight(31)
            for label in (heading, value, detail, note):
                label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            content.addWidget(heading)
            content.addLayout(value_row)
            if mode == ModeId.DATA_PREPARATION:
                self._data_errors = QLabel()
                self._data_errors.setFont(body_font(9))
                self._data_errors.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                content.addWidget(self._data_errors)
            content.addWidget(note)
            content.addStretch(1)
            stage.activated.connect(self.activated.emit)
            layout.addWidget(stage, 1)
            self._stages.append(stage)
            self._values.append(value)
            self._details.append(detail)
            self._notes.append(note)

    def set_summary(self, summary: HomeSummary) -> None:
        """工程情報を表示する。"""
        self.summary = summary
        data_value = summary.unassigned_items
        data_detail = "件 未振り分け"
        data_note = f"最新の版 {summary.latest_train} / {summary.latest_validation}"
        training_value = (
            summary.running_epoch if summary.running_experiment else summary.completed_experiments
        )
        training_detail = (
            f"/ {summary.running_epochs} エポック"
            if summary.running_experiment
            else "件 完了"
            if summary.completed_experiments
            else "完了した実験はありません"
        )
        if not summary.running_experiment and not summary.completed_experiments:
            training_value = "学習を始めましょう"
            training_detail = ""
        training_note = (
            f"{summary.running_experiment} を学習中"
            if summary.running_experiment
            else "実行中の学習はありません"
        )
        if summary.queue_waiting or summary.queue_running:
            queue_state = "実行中" if summary.queue_running else "停止中"
            training_note += f"\n学習キュー 待機 {summary.queue_waiting} 件（{queue_state}）"
        comparison_value = summary.unevaluated_candidates or summary.candidate_count
        comparison_detail = "件 未評価の候補" if summary.unevaluated_candidates else "件 候補"
        if not summary.unevaluated_candidates and not summary.candidate_count:
            comparison_value = "候補を登録しましょう"
            comparison_detail = ""
        comparison_note = (
            f"候補 {summary.candidate_count} 件 ・ リリース済み {summary.released_count} 件"
        )
        for label, value in zip(
            self._values, (data_value, training_value, comparison_value), strict=True
        ):
            label.setText(str(value))
        for label, text in zip(
            self._details, (data_detail, training_detail, comparison_detail), strict=True
        ):
            label.setText(text)
        for label, text in zip(
            self._notes, (data_note, training_note, comparison_note), strict=True
        ):
            label.setText(text)
        self._data_errors.setText(f"⚠ {summary.data_errors} 件のエラー")
        set_style(self._data_errors, role="note", state="error" if summary.data_errors else "idle")
        should_pulse = bool(summary.running_experiment) and os.getenv("FOAM_REDUCED_MOTION") != "1"
        if should_pulse and not self._timer.isActive():
            self._timer.start(80)
        elif not should_pulse and self._timer.isActive():
            self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._pulse = (self._pulse + 0.035) % 1.0
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        points = [stage.mapTo(self, stage.rect().topLeft()) for stage in self._stages]
        centers = [stage.mapTo(self, stage.rect().center()).x() for stage in self._stages]
        rail_y = points[0].y() + 13
        first_x = points[0].x() + 22
        painter.setPen(QPen(QColor(Color.RULE), 2))
        painter.drawLine(first_x, rail_y, centers[-1], rail_y)
        for index, (stage, point) in enumerate(zip(self._stages, points, strict=True)):
            x = point.x() + 16
            running = (
                index == 1 and self.summary is not None and bool(self.summary.running_experiment)
            )
            alpha = int(110 + 145 * abs(0.5 - self._pulse) * 2) if running else 255
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(Color.GRAPHITE if running else Color.IDLE).toRgb())
            painter.setOpacity(alpha / 255)
            painter.drawEllipse(x, point.y() + 7, 12, 12)
            painter.setOpacity(1.0)
            if index < len(self._stages) - 1:
                separator_x = stage.geometry().right()
                painter.setPen(QPen(QColor(Color.RULE_SOFT), 1))
                painter.drawLine(separator_x, 1, separator_x, self.height() - 1)


class HomeWindow(QMainWindow):
    """アプリ機能の入口と状態概要を表示する。"""

    def __init__(self, ctx: AppContext, manager: WindowManager, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.manager = manager
        self.setWindowTitle("気泡インスタンスセグメンテーション")
        self.resize(900, 560)
        saved_home = self.manager.settings.value("home/geometry")
        if saved_home:
            self._saved_geometry = tuple(int(value) for value in saved_home)
        self._recent: deque[tuple[datetime, str]] = deque(maxlen=2)
        self._summary: HomeSummary | None = None
        self._observed_jobs: set[int] = set()
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 16, 22, 14)
        layout.setSpacing(8)
        title_row = QHBoxLayout()
        title = QLabel("気泡インスタンスセグメンテーション")
        title.setFont(body_font(16))
        mock_label = QLabel("モック動作中")
        mock_label.setFont(body_font(9))
        mock_label.setStyleSheet(f"color: {Color.SLATE}")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(mock_label)
        layout.addLayout(title_row)
        section = QLabel("管理者機能（左から順に進む工程）")
        section.setFont(body_font(9))
        section.setStyleSheet(f"color: {Color.SLATE}")
        layout.addWidget(section)
        self.pipeline = PipelineWidget()
        self.pipeline.activated.connect(self._open_mode)
        layout.addWidget(self.pipeline)
        user_section = QLabel("利用者機能")
        user_section.setFont(body_font(9))
        user_section.setStyleSheet(f"color: {Color.SLATE}")
        layout.addWidget(user_section)
        self.user_panel = ClickablePanel(PageId.INFERENCE, "homeUserRow")
        self.user_panel.activated.connect(self.manager.navigate)
        user_layout = QHBoxLayout(self.user_panel)
        user_layout.setContentsMargins(16, 12, 16, 12)
        user_layout.setSpacing(18)
        user_title = QLabel("本番推論")
        user_title.setFont(body_font(11))
        user_title.setStyleSheet("font-weight: 600")
        user_title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        user_layout.addWidget(user_title)
        self.routing_layout = QHBoxLayout()
        self.routing_layout.setSpacing(18)
        user_layout.addLayout(self.routing_layout, 1)
        layout.addWidget(self.user_panel)
        self.recent_widget = QFrame()
        self.recent_widget.setObjectName("recentOperations")
        recent_layout = QHBoxLayout(self.recent_widget)
        recent_layout.setContentsMargins(0, 6, 0, 0)
        recent_layout.setSpacing(7)
        recent_title = QLabel("最近の操作")
        recent_title.setFont(body_font(9))
        recent_title.setStyleSheet(f"color: {Color.SLATE}")
        recent_layout.addWidget(recent_title)
        self._recent_slots = []
        for _ in range(2):
            time_label = QLabel()
            time_label.setFont(numeric_font(9))
            time_label.setStyleSheet(f"color: {Color.SLATE}")
            message_label = QLabel()
            message_label.setFont(body_font(9))
            message_label.setStyleSheet(f"color: {Color.SLATE}")
            recent_layout.addWidget(time_label)
            recent_layout.addWidget(message_label, 1)
            self._recent_slots.append((time_label, message_label))
        layout.addWidget(self.recent_widget)
        layout.addStretch(1)
        self.setCentralWidget(root)
        self.status_text = QLabel("準備完了")
        self.job_count = QLabel()
        self.job_count.setFont(numeric_font())
        self.statusBar().addWidget(self.status_text, 1)
        self.statusBar().addPermanentWidget(self.job_count)
        self.ctx.status.message.connect(self._status_message)
        self.ctx.jobs.jobs_changed.connect(self._jobs_changed)
        self.ctx.training_runner.progressed.connect(lambda _experiment_id: self.refresh_summary())
        self.ctx.training_runner.ended.connect(self._training_ended)
        self._jobs_changed(self.ctx.jobs.running_count)
        self.manager.set_home_callback(self.show_home)
        self.manager.mode_closed.connect(lambda _mode: self.refresh_summary())
        self._build_menus()
        self.refresh_summary()

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル(&F)")
        file_menu.addAction("終了", self.close)
        tools_menu = self.menuBar().addMenu("ツール(&T)")
        tools_menu.addAction("キー割り当て", self._show_shortcuts_info)
        display_menu = tools_menu.addMenu("原画像と切り替える表示")
        self.display_actions = {}
        for name in sorted(self.ctx.display.MODES):
            action = display_menu.addAction(display_mode_label(name))
            action.setCheckable(True)
            action.setChecked(name == self.ctx.display.value)
            action.triggered.connect(
                lambda checked=False, value=name: self.ctx.display.set_value(value)
            )
            self.display_actions[name] = action
        self.ctx.display.changed.connect(self._display_changed)
        self.help_shortcut = QShortcut(QKeySequence(self.ctx.shortcuts["help"]), self)
        self.help_shortcut.activated.connect(self._show_shortcuts_info)
        self.f1_shortcut = QShortcut(QKeySequence("F1"), self)
        self.f1_shortcut.activated.connect(self._show_shortcuts_info)
        self.ctx.shortcuts.changed.connect(self._shortcuts_changed)
        help_menu = self.menuBar().addMenu("ヘルプ(&H)")
        help_menu.addAction(
            "バージョン情報",
            lambda: QMessageBox.about(
                self, "バージョン情報", "気泡インスタンスセグメンテーション 0.1.0（モック）"
            ),
        )

    def _show_shortcuts_info(self) -> None:
        from .keymap_dialog import show_keymap_window

        show_keymap_window(self, self.ctx)

    def _display_changed(self, name: str) -> None:
        """共有表示設定の選択状態を更新する。"""
        for value, action in self.display_actions.items():
            action.setChecked(value == name)

    def _shortcuts_changed(self) -> None:
        """キー割り当て変更をホームのヘルプ操作へ反映する。"""
        self.help_shortcut.setKey(QKeySequence(self.ctx.shortcuts["help"]))

    def _open_mode(self, mode: ModeId) -> None:
        page_id = {
            ModeId.DATA_PREPARATION: PageId.DATA_PREPARATION,
            ModeId.TRAINING: PageId.TRAINING,
            ModeId.COMPARISON: PageId.CANDIDATES,
        }[ModeId(mode)]
        self.manager.navigate(page_id, {"_preserve_current_tab": True})

    def _jobs_changed(self, count: int) -> None:
        self.job_count.setText(running_jobs_label(count))
        for job in self.ctx.jobs.jobs():
            if id(job) not in self._observed_jobs:
                self._observed_jobs.add(id(job))
                job.progress.connect(lambda *_args: self.refresh_summary())
        self.refresh_summary()

    def _training_ended(self, outcome) -> None:
        """学習の終端状態を日本語で通知する。"""
        if outcome.reason == "conclusion_failed":
            self.ctx.status.show_message(
                f"{outcome.experiment_id} の終端状態を保存できず、キューを停止しました: "
                f"{outcome.message}"
            )
            self.refresh_summary()
            return
        status = {
            "completed": "完了しました",
            "failed": "失敗しました",
            "stopped": "中断しました",
        }.get(outcome.status, "終了しました")
        self.ctx.status.show_message(f"{outcome.experiment_id} の学習が{status}")
        self.refresh_summary()

    def _status_message(self, message: str) -> None:
        self.status_text.setText(message)
        self._recent.appendleft((datetime.now().astimezone(), message))
        self.refresh_summary()

    def refresh_summary(self) -> None:
        """ホーム表示用集計を作り、各部品へ渡す。"""
        self._summary = replace(
            build_home_summary(self.ctx.backend, self.ctx.jobs, self.ctx.queue_controller),
            recent=tuple(self._recent),
        )
        self.pipeline.set_summary(self._summary)
        while self.routing_layout.count():
            item = self.routing_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for classification, model_id in self._summary.routing.items():
            assignment = QWidget()
            assignment.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            row = QHBoxLayout(assignment)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            label = QLabel(f"{classification} →")
            label.setFont(body_font(9))
            model = QLabel(model_id or "未割り当て")
            model.setFont(numeric_font(9) if model_id else body_font(9))
            if not model_id:
                set_style(model, state="error")
            row.addWidget(label)
            row.addWidget(model)
            self.routing_layout.addWidget(assignment)
        self.routing_layout.addStretch(1)
        for index, slot in enumerate(self._recent_slots):
            if index < len(self._summary.recent):
                when, message = self._summary.recent[index]
                slot[0].setText(when.strftime("%H:%M"))
                slot[1].setText(message)
                slot[0].show()
                slot[1].show()
            else:
                slot[0].hide()
                slot[1].hide()

    def show_home(self) -> None:
        """ホームを前面へ出し、最新状態を再集計する。"""
        self.refresh_summary()
        self.showNormal()
        saved_geometry = getattr(self, "_saved_geometry", None)
        if saved_geometry:
            self.setGeometry(*saved_geometry)
            del self._saved_geometry
        self.raise_()
        self.activateWindow()

    def _interrupt_prompt(self) -> str | None:
        """学習・評価の実行中なら、中断して終了するかを尋ねる文を返す。"""
        training = self.ctx.training_runner.is_busy
        evaluation = bool(self.ctx.evaluation_runner and self.ctx.evaluation_runner.is_busy)
        if training and evaluation:
            return "学習と評価を中断して終了しますか？"
        if training:
            return "学習を中断して終了しますか？"
        if evaluation:
            return "評価を中断して終了しますか？"
        return None

    def confirm_exit(self) -> bool:
        """モード画面または学習中ジョブがある場合の終了確認。"""
        if (
            not self.manager.open_modes()
            and self.ctx.jobs.running_count == 0
            and self._interrupt_prompt() is None
        ):
            return True
        result = QMessageBox.question(
            self,
            "終了の確認",
            self._interrupt_prompt()
            or "開いているモード画面または実行中ジョブがあります。アプリを終了しますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return result == QMessageBox.StandardButton.Yes

    def closeEvent(self, event) -> None:
        rect = self.geometry()
        self.manager.settings.setValue(
            "home/geometry", [rect.x(), rect.y(), rect.width(), rect.height()]
        )
        self.manager.settings.sync()
        queue = self.ctx.queue_controller
        queue_state = queue.suspend_for_shutdown() if queue else None
        evaluation = self.ctx.evaluation_runner
        needs_confirmation = (
            bool(self.manager.open_modes())
            or self.ctx.training_runner.is_busy
            or bool(evaluation and evaluation.is_busy)
            or bool(queue and queue_state and queue_state[0])
        )
        answer = True
        if needs_confirmation:
            prompt = (
                self._interrupt_prompt()
                or "実行中の画面または学習があります。アプリを終了しますか？"
            )
            answer = (
                QMessageBox.question(
                    self,
                    "終了の確認",
                    prompt,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                == QMessageBox.StandardButton.Yes
            )
        if answer:
            if self.ctx.training_runner.is_busy:
                self.ctx.training_runner.request_stop("app_exit", timeout_ms=10_000)
            if evaluation is not None and evaluation.is_busy:
                # 学習と同じ手順（stop_request.json → kill → 最大 10 秒待つ。比較・推論設計 15.2）
                evaluation.shutdown()
            self.manager.save_all_windows()
            event.accept()
            from PySide6.QtWidgets import QApplication

            QApplication.instance().quit()
        else:
            if queue and queue_state is not None:
                queue.restore_after_shutdown(queue_state)
            event.ignore()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == event.Type.ActivationChange and self.isActiveWindow():
            self.refresh_summary()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        saved_geometry = getattr(self, "_saved_geometry", None)
        if saved_geometry:
            self.setGeometry(*saved_geometry)
            del self._saved_geometry
