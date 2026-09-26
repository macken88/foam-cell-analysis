"""学習キューを画面から独立して進めるコントローラー。"""

from PySide6.QtCore import QObject, QTimer, Signal


class TrainingQueueController(QObject):
    changed = Signal()

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.executing = False
        self.stop_requested = False
        self.active_id = None
        self._job = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self.changed.emit)
        ctx.jobs.jobs_changed.connect(self._jobs_changed)

    def _jobs_changed(self, _count):
        self.changed.emit()

    def start(self):
        if self.executing:
            return
        self.executing = True
        self.stop_requested = False
        if self.ctx.jobs.running_count:
            self.ctx.status.show_message("単発の学習が終わってからキューを開始します")
            self.ctx.jobs.jobs_changed.connect(self._wait_for_single_job)
        else:
            self._next()
        self.changed.emit()

    def stop(self):
        if not self.executing:
            return
        self.stop_requested = True
        self.ctx.status.show_message(
            "今の学習が終わったら停止します。すぐに止めるには実験一覧の「中断」を使ってください。"
        )
        self.changed.emit()

    def _wait_for_single_job(self, count):
        if count == 0 and self.executing:
            try:
                self.ctx.jobs.jobs_changed.disconnect(self._wait_for_single_job)
            except (RuntimeError, TypeError):
                pass
            self._next()

    def _next(self):
        if not self.executing:
            return
        if self.stop_requested:
            self.executing = False
            self.active_id = None
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
        from ...navigation import PageId
        from .page import TrainingPage

        manager = self.parent()
        page = manager.page(PageId.TRAINING) if hasattr(manager, "page") else None
        try:
            job = page.create_training_job(experiment) if isinstance(page, TrainingPage) else None
        except Exception as error:
            self.ctx.backend.finish_training(item.experiment_id, "failed")
            self.ctx.status.show_message(
                f"{item.experiment_id} の学習ジョブ作成に失敗しました: {error}"
            )
            self._next()
            return
        if job is None:
            self.ctx.backend.finish_training(item.experiment_id, "failed")
            self._next()
            return
        self._job = job
        job.progress.connect(lambda *_args: self._refresh_timer.start())

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
            self._next()

        job.finished.connect(finished)
        self.ctx.jobs.start(job)
        self.changed.emit()
