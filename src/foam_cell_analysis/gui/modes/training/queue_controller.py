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
        self.waiting_for_training = False
        self.waiting_for_id = None
        self._job = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self.changed.emit)
        ctx.jobs.jobs_changed.connect(self._jobs_changed)

    def _jobs_changed(self, _count):
        self.changed.emit()
        if self.executing and self.waiting_for_training and not self.ctx.jobs.has_training_job:
            self._wait_for_single_job()

    def launch_training(self, experiment, on_finished=None):
        """唯一の学習ジョブ開始口。実行中の学習があれば開始を拒否する。"""
        if self.ctx.jobs.has_training_job:
            raise RuntimeError("別の学習が実行中です")
        from ...navigation import PageId

        manager = self.parent()
        page = manager.page(PageId.TRAINING) if hasattr(manager, "page") else None
        if page is None:
            raise RuntimeError("学習設定画面を取得できません")
        job = page.create_training_job(experiment)
        job.progress.connect(self._queue_progress_refresh)
        if on_finished is not None:
            job.finished.connect(on_finished)
        self.ctx.jobs.start(job)
        return job

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
        if self.ctx.jobs.has_training_job:
            self.waiting_for_training = True
            self.waiting_for_id = self.ctx.jobs.training_jobs[0].key.removeprefix("training:")
            self.ctx.status.show_message(f"{self.waiting_for_id} の学習終了後にキューを開始します")
            self._connect_waiter()
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
        if self.ctx.jobs.has_training_job:
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
        try:
            experiment = self.ctx.backend.start_training(item.config.values, item.experiment_id)
        except Exception as error:
            self.ctx.backend.finish_training(item.experiment_id, "failed")
            self.ctx.status.show_message(f"{item.experiment_id} の開始に失敗しました: {error}")
            self._next()
            return

        def finished(ok, _message, expid=item.experiment_id):
            status = (
                "failed"
                if expid in getattr(self.ctx.backend, "fail_training_ids", set())
                else "completed"
                if ok
                else "stopped"
            )
            self.ctx.backend.finish_training(expid, status)
            self.active_id = None
            self._job = None
            self.changed.emit()
            QTimer.singleShot(0, self._next)

        try:
            self._job = self.launch_training(experiment, finished)
        except Exception as error:
            self.ctx.backend.finish_training(item.experiment_id, "failed")
            self.active_id = None
            self.ctx.status.show_message(
                f"{item.experiment_id} の学習ジョブ開始に失敗しました: {error}"
            )
            QTimer.singleShot(0, self._next)
            return
        self.changed.emit()
