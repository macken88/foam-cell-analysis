"""学習キューを画面から独立して進めるコントローラー。"""

from PySide6.QtCore import QObject, QTimer, Signal


class TrainingQueueController(QObject):
    changed = Signal()
    progressed = Signal()

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.executing = False
        self.stop_requested = False
        self.active_id = None
        self.active_queue_id = None
        self.waiting_for_training = False
        self.waiting_for_id = None
        self._job = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self.changed.emit)
        ctx.training_runner.ended.connect(self._training_ended)
        ctx.training_runner.progressed.connect(
            lambda _experiment_id: self._queue_progress_refresh()
        )

    def _jobs_changed(self, _count):
        self.changed.emit()

    def launch_training(self, experiment, on_finished=None):
        """互換入口を runner へ委譲する。"""
        self.ctx.training_runner.start(experiment.experiment_id)
        return self.ctx.training_runner.job

    def _queue_progress_refresh(self, *_args):
        self.progressed.emit()
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def sync_training_identifier(self):
        """学習設定フォームにバックエンドの最新 ID を反映する。"""
        from ...navigation import PageId

        manager = self.parent()
        if hasattr(manager, "page"):
            page = manager.page(PageId.TRAINING)
            page.refresh_next_identifier()

    def start(self):
        if self.executing:
            return
        self.executing = True
        self.stop_requested = False
        if self.ctx.training_runner.is_busy:
            self.waiting_for_training = True
            self.waiting_for_id = self.ctx.training_runner.experiment_id
            self.ctx.status.show_message(f"{self.waiting_for_id} の学習終了後にキューを開始します")
        else:
            self._next()
        self.changed.emit()

    def _connect_waiter(self):
        try:
            self.ctx.jobs.jobs_changed.connect(self._wait_for_single_job)
        except RuntimeError:
            pass

    def stop(self):
        if not self.executing:
            return
        self.stop_requested = True
        self.ctx.status.show_message(
            "今の学習が終わったら停止します。すぐに止めるには実験一覧の「中断」を使ってください。"
        )
        self.changed.emit()

    def _wait_for_single_job(self, _count=None):
        if self.executing and not self.ctx.jobs.has_training_job:
            try:
                self.ctx.jobs.jobs_changed.disconnect(self._wait_for_single_job)
            except (RuntimeError, TypeError):
                pass
            self.waiting_for_training = False
            self.waiting_for_id = None
            self._next()

    def _next(self):
        if not self.executing:
            return
        if self.stop_requested:
            self.executing = False
            self.stop_requested = False
            self.waiting_for_training = False
            self.waiting_for_id = None
            self.active_id = None
            self.changed.emit()
            return
        if self.ctx.training_runner.is_busy:
            self.waiting_for_training = True
            self._connect_waiter()
            self.changed.emit()
            return
        try:
            item = self.ctx.backend.take_next_training_queue_item()
        except Exception as error:
            self.ctx.status.show_message(f"キュー項目の取得に失敗しました: {error}")
            self.executing = False
            self.active_id = None
            self.changed.emit()
            return
        if item is None:
            self.executing = False
            self.stop_requested = False
            self.waiting_for_training = False
            self.waiting_for_id = None
            self.active_id = None
            self.changed.emit()
            return
        self.active_id = item.experiment_id
        self.active_queue_id = item.queue_id or item.experiment_id
        try:
            self.ctx.training_runner.start(
                item.experiment_id,
                self.active_queue_id,
                bool(getattr(item, "queue_is_retry", False)),
            )
        except Exception as error:
            self.ctx.backend.finish_training_queue_item(self.active_queue_id, "failed")
            self.active_id = None
            self.active_queue_id = None
            self.ctx.status.show_message(
                f"{item.experiment_id} の学習ジョブ開始に失敗しました: {error}"
            )
            QTimer.singleShot(0, self._next)
            return
        self.changed.emit()

    def suspend_for_shutdown(self):
        """終了確認の間、キュー進行状態を一時停止する。"""
        state = (self.executing, self.stop_requested)
        self.executing = False
        self.stop_requested = False
        self.changed.emit()
        return state

    def restore_after_shutdown(self, state):
        """終了を取り消したときにキュー進行を戻す。"""
        self.executing, self.stop_requested = state
        self.changed.emit()
        if self.executing and not self.ctx.training_runner.is_busy:
            self._next()

    def _training_ended(self, outcome):
        if self.waiting_for_training and self.executing and self.active_id is None:
            self.waiting_for_training = False
            QTimer.singleShot(0, self._next)
            return
        if not self.executing or outcome.experiment_id != self.active_id:
            return
        self.active_id = None
        self.active_queue_id = None
        self._job = None
        self.changed.emit()
        if outcome.reason == "conclusion_failed":
            self.executing = False
            self.stop_requested = False
            self.waiting_for_training = False
            self.waiting_for_id = None
            message = (
                f"{outcome.experiment_id} の終端状態を保存できないため、キューを停止しました: "
                f"{outcome.message}"
            )
            QTimer.singleShot(0, lambda: self.ctx.status.show_message(message))
            self.changed.emit()
            return
        QTimer.singleShot(0, self._next)
