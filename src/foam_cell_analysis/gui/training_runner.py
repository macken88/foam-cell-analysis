"""学習ジョブの開始・停止・終端処理を一元管理する。"""

from __future__ import annotations

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal

from ..services.models import JobExit, TrainingOutcome
from .training_jobs import FakeTrainingJob, ProcessTrainingJob


class TrainingRunner(QObject):
    """試行の占有期間と終端処理を管理する。"""

    progressed = Signal(str)
    ended = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self, backend, parent=None, hello_timeout_ms=30_000):
        super().__init__(parent)
        self.backend = backend
        self.hello_timeout_ms = hello_timeout_ms
        self.job = None
        self.experiment_id = None
        self.attempt = None
        self.queue_id = None
        self.is_busy = False
        self._concluded = False
        self._skip_conclude = False
        self._event_failure = None

    def start(self, experiment_id, queue_id=None, retry=False):
        """Backend で試行を準備して学習を開始する。"""
        if self.is_busy:
            raise RuntimeError("別の学習が実行中です")
        self._set_busy(True)
        self._concluded = False
        self._skip_conclude = False
        self._event_failure = None
        self.experiment_id = experiment_id
        self.queue_id = queue_id
        try:
            prepared = self.backend.prepare_training_run(experiment_id, queue_id, retry)
            self.attempt = int(prepared.run_id.rsplit("/attempt_", 1)[1])
            if prepared.fake:
                job = FakeTrainingJob(
                    prepared, self.backend.get_experiment(experiment_id), parent=self
                )
            else:
                job = ProcessTrainingJob(prepared, self.backend, self.hello_timeout_ms, self)
            self.job = job
            job.event_received.connect(self._apply_event)
            job.finished.connect(self._job_finished)
            job.start()
        except Exception as error:
            self._conclude(JobExit(start_failed=True, message=str(error)))

    def _set_busy(self, busy):
        if self.is_busy == busy:
            return
        self.is_busy = busy
        self.busy_changed.emit(busy)

    def _apply_event(self, event):
        try:
            self.backend.apply_training_event(self.experiment_id, event)
        except Exception as error:
            self._event_failure = str(error)
            if self.job is not None:
                self.job.kill()
        finally:
            self.progressed.emit(self.experiment_id)

    def _job_finished(self, job_exit):
        if not self._skip_conclude:
            if self._event_failure:
                job_exit = JobExit(
                    returncode=1,
                    message=f"学習イベントの保存に失敗しました: {self._event_failure}",
                )
            self._conclude(job_exit)

    def _conclude(self, job_exit):
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
                    self.backend.finish_training_queue_item(self.queue_id, "failed")
        except Exception as error:
            outcome = TrainingOutcome(
                self.experiment_id or "",
                self.attempt or 0,
                self.queue_id,
                "failed",
                "conclusion_failed",
                str(error),
            )
        finally:
            self.job = None
            self._set_busy(False)
            if outcome is not None:
                self.ended.emit(outcome)

    def request_stop(self, reason="user_stop", timeout_ms=None):
        """停止要求を保存し、子プロセスの終了を待って確定する。"""
        if not self.is_busy:
            return True
        if self.attempt is not None:
            self.backend.request_training_stop(self.experiment_id, self.attempt, reason)
        job = self.job
        if job is None:
            return True
        job.kill()
        if timeout_ms is None:
            return True
        loop = QEventLoop(self)
        timer = QTimer(self)
        timer.setSingleShot(True)
        self.ended.connect(loop.quit)
        timer.timeout.connect(loop.quit)
        timer.start(timeout_ms)
        if self.is_busy:
            loop.exec()
        try:
            self.ended.disconnect(loop.quit)
        except (RuntimeError, TypeError):
            pass
        if self.is_busy:
            self._skip_conclude = True
            return False
        return True
