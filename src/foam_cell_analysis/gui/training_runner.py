"""学習ジョブの開始・停止・終端処理を一元管理する。"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QObject, Signal

from ..services.models import JobExit, TrainingOutcome
from .compute_coordinator import ComputeCoordinator, Ticket
from .error_messages import user_failure_message
from .training_jobs import FakeTrainingJob, ProcessTrainingJob
from .waiting import disconnect_runner_slots, job_exit_confirmed, wait_until

OWNER = "training"


class TrainingRunner(QObject):
    """試行の占有期間と終端処理を管理する。

    計算処理の占有（ComputeCoordinator）は準備 → 起動 → 実行 → 終端処理まで持ち、
    終端処理の後に返す。「プロセスが終わった」（busy の解除と ended）と
    「次を開始してよい」（占有の返却）を分け、終端状態の保存に失敗したときは
    占有を ok=False で返して次の要求を開始させない。
    """

    progressed = Signal(str)
    ended = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self, backend, parent=None, hello_timeout_ms=30_000, compute=None):
        super().__init__(parent)
        self.backend = backend
        self.hello_timeout_ms = hello_timeout_ms
        self.compute = compute if compute is not None else ComputeCoordinator(self)
        self._ticket: Ticket | None = None
        self._waiting_ticket: Ticket | None = None
        self.job = None
        self.experiment_id = None
        self.attempt = None
        self.queue_id = None
        self.is_busy = False
        self._concluded = False
        self._skip_conclude = False
        self._event_failure = None

    def start(self, experiment_id, queue_id=None, retry=False, *, ticket: Ticket | None = None):
        """Backend で試行を準備して学習を開始する。

        計算処理が空いていればこの呼び出しの中で開始し、評価などが使用中なら
        占有を待ってから開始する（waiting_for_compute が True の間）。
        ticket に占有済みの要求（owner="training"）を渡すと、それを引き継いで
        すぐ開始する（学習キューが先に占有を待った場合）。
        """
        if self.is_busy:
            raise RuntimeError("別の学習が実行中です")
        if ticket is not None and (ticket.state != "active" or ticket is not self.compute.active):
            raise RuntimeError("占有していない要求では学習を開始できません")
        self._set_busy(True)
        self._concluded = False
        self._skip_conclude = False
        self._event_failure = None
        self.experiment_id = experiment_id
        self.queue_id = queue_id
        # 前回の試行番号が残っていると、準備に失敗したとき前回の試行を終端処理してしまう
        self.attempt = None
        self.job = None
        if ticket is not None:
            self._ticket = ticket
            self._begin(retry)
            return
        waiting = self.compute.request(
            OWNER, f"学習 {experiment_id}", lambda: self._begin_with_active_ticket(retry)
        )
        if waiting.state == "waiting":
            self._waiting_ticket = waiting

    @property
    def waiting_for_compute(self) -> bool:
        """評価などの終了を待っていて、まだ準備を始めていないか。"""
        return self._waiting_ticket is not None

    def _begin_with_active_ticket(self, retry):
        self._waiting_ticket = None
        self._ticket = self.compute.active
        self._begin(retry)

    def _begin(self, retry):
        experiment_id = self.experiment_id
        try:
            prepared = self.backend.prepare_training_run(experiment_id, self.queue_id, retry)
            self.attempt = int(prepared.run_id.rsplit("/attempt_", 1)[1])
            if prepared.fake:
                job = FakeTrainingJob(
                    prepared, self.backend.get_experiment(experiment_id), parent=self
                )
            else:
                job = ProcessTrainingJob(
                    prepared,
                    self.backend,
                    self.hello_timeout_ms,
                    self,
                    experiment_id=experiment_id,
                    attempt=self.attempt,
                )
            self.job = job

            def event_slot(event):
                self._apply_event(event, job, experiment_id)

            def finished_slot(job_exit):
                self._job_finished(job_exit, job)

            job.event_received.connect(event_slot)
            job.finished.connect(finished_slot)
            job._runner_slots = (
                ("event_received", event_slot),
                ("finished", finished_slot),
            )
            job.start()
        except Exception as error:
            import logging

            logging.getLogger(__name__).exception("学習を開始できません")
            self._conclude(JobExit(start_failed=True, message=user_failure_message(error)))

    def _set_busy(self, busy):
        if self.is_busy == busy:
            return
        self.is_busy = busy
        self.busy_changed.emit(busy)

    def _apply_event(self, event, source_job=None, experiment_id=None):
        job = self.job if source_job is None else source_job
        experiment_id = self.experiment_id if experiment_id is None else experiment_id
        if self._concluded or job is None or job is not self.job or self._event_failure is not None:
            return
        try:
            self.backend.apply_training_event(experiment_id, event)
        except Exception as error:
            if self._event_failure is None:
                self._event_failure = str(error) or type(error).__name__
                job.kill()
        finally:
            self.progressed.emit(experiment_id)

    def _job_finished(self, job_exit, source_job=None):
        if source_job is not None and source_job is not self.job:
            return
        if not self._skip_conclude:
            if self._event_failure is not None:
                job_exit = replace(
                    job_exit,
                    message=f"学習イベントの保存に失敗しました: {self._event_failure}",
                    protocol_error=True,
                )
            self._conclude(job_exit, source_job)

    def _conclude(self, job_exit, old_job=None):
        if self._concluded:
            return
        self._concluded = True
        outcome = None
        try:
            if self.attempt is not None:
                outcome = self.backend.conclude_training_run(
                    self.experiment_id, self.attempt, job_exit
                )
                if self.queue_id:
                    self.backend.finish_training_queue_item(self.queue_id, outcome.status)
                    # 結果は実験一覧に残るため、終わった行はキューの表から外す
                    self.backend.clear_finished_training_queue_items()
            else:
                if hasattr(self.backend, "fail_training_preparation"):
                    self.backend.fail_training_preparation(self.experiment_id)
                outcome = TrainingOutcome(
                    self.experiment_id or "",
                    0,
                    self.queue_id,
                    "failed",
                    "prepare_failed",
                    job_exit.message,
                )
                if self.queue_id:
                    # 試行がなくても実験一覧に失敗として記録し、失敗行は片付け操作で除ける
                    self.backend.finish_training_queue_item(self.queue_id, "failed")
        except Exception as error:
            import logging

            logging.getLogger(__name__).exception("学習の終了状態を保存できません")
            outcome = TrainingOutcome(
                self.experiment_id or "",
                self.attempt or 0,
                self.queue_id,
                "failed",
                "conclusion_failed",
                user_failure_message(
                    error,
                    prepare_message="学習結果を保存できませんでした。ログを確認してください。",
                    fallback="学習結果を保存できませんでした。ログを確認してください。",
                ),
            )
        finally:
            self.job = None
            ticket, self._ticket = self._ticket, None
            self._set_busy(False)
            if outcome is not None:
                self.ended.emit(outcome)
            # 終端処理の後で占有を返す。保存に失敗したら次の要求を開始させない
            if ticket is not None:
                saved = outcome is None or outcome.reason != "conclusion_failed"
                self.compute.release(ticket, ok=saved, error=None if saved else outcome.message)
            if job_exit_confirmed(job_exit, old_job):
                disconnect_runner_slots(old_job)
                old_job.deleteLater()

    def request_stop(self, reason="user_stop", timeout_ms=None):
        """停止要求を保存し、子プロセスの終了を待って確定する。"""
        if not self.is_busy:
            return True
        if self._waiting_ticket is not None:
            # まだ準備していないので、待機中の要求を取り消すだけでよい
            self._cancel_waiting()
            return True
        if self.attempt is not None:
            self.backend.request_training_stop(self.experiment_id, self.attempt, reason)
        job = self.job
        if job is None:
            return True
        if timeout_ms is None:
            job.kill()
            return True
        completed = wait_until(lambda: self.job is not job, job.finished, timeout_ms, job.kill)
        if not completed and self.job is job:
            self._skip_conclude = True
            return False
        return completed

    def _cancel_waiting(self):
        ticket, self._waiting_ticket = self._waiting_ticket, None
        self.compute.cancel(ticket)
        self._concluded = True
        self._set_busy(False)
        self.ended.emit(
            TrainingOutcome(
                self.experiment_id or "",
                0,
                self.queue_id,
                "stopped",
                "cancelled",
                "開始前に取り消しました",
            )
        )
