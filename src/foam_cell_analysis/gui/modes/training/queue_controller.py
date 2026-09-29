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
        self.immediate_stop_requested = False
        self.active_id = None
        self.active_queue_id = None
        self.waiting_for_training = False
        self.waiting_for_id = None
        # 評価など別の処理が計算を占有している間、次の行の開始を待つ占有要求
        self._compute_ticket = None
        self._job = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self.changed.emit)
        ctx.training_runner.ended.connect(self._training_ended)
        ctx.training_runner.progressed.connect(
            lambda _experiment_id: self._queue_progress_refresh()
        )
        ctx.compute.changed.connect(self._compute_changed)

    @property
    def waiting_for_compute(self) -> bool:
        """評価など別の処理の終了を待っていて、次の行をまだ開始していないか。"""
        return self._compute_ticket is not None

    @property
    def compute_wait_text(self) -> str | None:
        """キュー表に出す待ち理由（「評価の終了を待っています」など）。"""
        if not self.waiting_for_compute:
            return None
        return self.ctx.compute.wait_message("training") or "計算処理の終了を待っています"

    def _compute_changed(self):
        if self.waiting_for_compute:
            self.changed.emit()

    def _cancel_compute_wait(self):
        ticket, self._compute_ticket = self._compute_ticket, None
        if ticket is not None:
            self.ctx.compute.cancel(ticket)

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
        if self.ctx.compute.is_blocked:
            # 終端状態の保存失敗で止めていた計算処理を、利用者の再実行で再開する
            self.ctx.compute.unblock()
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
        if self.waiting_for_compute and self.active_id is None:
            # まだ次の行を始めていないので、待つのをやめてキューを止める
            self._cancel_compute_wait()
            self.executing = False
            self.stop_requested = False
            self.changed.emit()
            return
        self.stop_requested = True
        self.ctx.status.show_message(
            "今の学習は最後まで続けます。終わったら次の行へ進まず、キューを止めます。"
            "今の学習の結果は「完了」として残ります。"
        )
        self.changed.emit()

    def stop_now(self):
        """キュー進行を止め、実行中の学習にも即時停止を依頼する。"""
        self.stop_requested = True
        self.immediate_stop_requested = True
        self.executing = False
        self.waiting_for_training = False
        self.waiting_for_id = None
        self._cancel_compute_wait()
        if self.ctx.training_runner.is_busy:
            if self.active_id is None:
                self.active_id = self.ctx.training_runner.experiment_id
            self.ctx.training_runner.request_stop("user_stop")
        else:
            self.stop_requested = False
            self.immediate_stop_requested = False
            self.active_id = None
            self.active_queue_id = None
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

    def _next(self, ticket=None):
        """次の待機行を開始する。ticket は占有済みの要求（占有を待った場合）。"""
        if not self.executing:
            self._release_unused(ticket)
            return
        if self.stop_requested:
            self._release_unused(ticket)
            self.executing = False
            self.stop_requested = False
            self.immediate_stop_requested = False
            self.waiting_for_training = False
            self.waiting_for_id = None
            self.active_id = None
            self.changed.emit()
            return
        if self.ctx.training_runner.is_busy:
            self._release_unused(ticket)
            self.waiting_for_training = True
            self._connect_waiter()
            self.changed.emit()
            return
        if ticket is None and self.ctx.compute.is_busy:
            # 評価などが実行中。先着順で占有を待ち、空いたら次の行を始める
            if self._compute_ticket is None:
                self._compute_ticket = self.ctx.compute.request(
                    "training", "学習キュー", self._compute_granted
                )
            self.changed.emit()
            return
        try:
            item = self.ctx.backend.take_next_training_queue_item()
        except Exception as error:
            self._release_unused(ticket)
            self.ctx.status.show_message(f"キュー項目の取得に失敗しました: {error}")
            self.executing = False
            self.active_id = None
            self.changed.emit()
            return
        if item is None:
            self._release_unused(ticket)
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
                ticket=ticket,
            )
        except Exception as error:
            self._release_unused(ticket)
            self.ctx.backend.finish_training_queue_item(self.active_queue_id, "failed")
            self.active_id = None
            self.active_queue_id = None
            self.ctx.status.show_message(
                f"{item.experiment_id} の学習ジョブ開始に失敗しました: {error}"
            )
            QTimer.singleShot(0, self._next)
            return
        self.changed.emit()

    def _compute_granted(self):
        """待っていた占有が回ってきたら、その占有で次の行を始める。"""
        ticket, self._compute_ticket = self._compute_ticket, None
        if ticket is None:
            ticket = self.ctx.compute.active
        self.changed.emit()
        self._next(ticket)

    def _release_unused(self, ticket):
        """学習に引き継がなかった占有を返す（返却済みなら何もしない）。"""
        if ticket is not None:
            self.ctx.compute.release(ticket)

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
        if self.executing and not self.ctx.training_runner.is_busy and not self.waiting_for_compute:
            self._next()

    def _training_ended(self, outcome):
        # 終わった行はキューの表から外れているため、どの経路でも表示を更新する
        self.changed.emit()
        if self.waiting_for_training and self.executing and self.active_id is None:
            self.waiting_for_training = False
            QTimer.singleShot(0, self._next)
            return
        if not self.executing or outcome.experiment_id != self.active_id:
            if self.immediate_stop_requested and outcome.experiment_id == self.active_id:
                self.active_id = None
                self.active_queue_id = None
                self.stop_requested = False
                self.immediate_stop_requested = False
                self.changed.emit()
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
        if outcome.reason == "prepare_failed":
            message = (
                f"{outcome.experiment_id} の学習を開始できませんでした"
                f"（キューの行は「失敗」として残しています）: {outcome.message}"
            )
            QTimer.singleShot(0, lambda: self.ctx.status.show_message(message))
        QTimer.singleShot(0, self._next)
