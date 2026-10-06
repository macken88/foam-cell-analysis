"""検証用データセットでの評価を候補ごとに順に実行する（比較・推論設計 7 章・15.2）。

評価ジョブは 2 種類ある。
- ProcessEvaluationJob: 評価プロセス（inference/run.py）を共通の ProcessJob で動かす。
- FakeEvaluationJob: モックとテスト用。イベント列を時間差で送るだけでファイルは作らない。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal

from ..jobs.protocol import read_json
from ..services.models import EvaluationOutcome, JobExit, PreparedRun
from .compute_coordinator import ComputeCoordinator, Ticket
from .process_job import HELLO_TIMEOUT_MS, ProcessJob, ProcessJobInfo

logger = logging.getLogger(__name__)

PREPARE_FAILED_TEXT = "評価の準備に失敗しました。作業フォルダのパスが長すぎる可能性があります。"
UNEXPECTED_FAILED_TEXT = "評価を開始できませんでした。ログを確認してください。"


def user_failure_message(error: BaseException) -> str:
    """例外を、画面に出してよい日本語の理由へ置き換える（詳細はログにだけ残す）。

    ValueError は最初の「: 」より前（パスや内部の詳細は後ろに入る）、
    OSError はパス長などの準備失敗の文、それ以外は汎用の文にする。
    """
    if isinstance(error, OSError):
        return PREPARE_FAILED_TEXT
    if isinstance(error, ValueError):
        text = str(error).split(": ", 1)[0].strip()
        if text:
            return text
    return UNEXPECTED_FAILED_TEXT


OWNER = "evaluation"
APP_EXIT_TIMEOUT_MS = 10_000


def _parse_run_id(run_id: str) -> tuple[str, str, str]:
    """run_id（RC-001/val_v000/eval_001）を（候補, 検証版, 評価）へ分ける。"""
    parts = run_id.split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"評価の run_id が不正です: {run_id}")
    return parts[0], parts[1], parts[2]


class ProcessEvaluationJob(ProcessJob):
    """hello/go ハンドシェイク付きの評価子プロセス（共通 ProcessJob の評価用設定）。"""

    def __init__(
        self,
        prepared: PreparedRun,
        backend,
        hello_timeout_ms: int = HELLO_TIMEOUT_MS,
        parent=None,
    ):
        candidate_id, _version, evaluation_id = _parse_run_id(prepared.run_id)
        self.prepared = prepared
        self.backend = backend
        self.candidate_id = candidate_id
        self.evaluation_id = evaluation_id
        self.key = f"evaluation:{candidate_id}"
        info = ProcessJobInfo(
            run_id=prepared.run_id,
            run_dir=prepared.run_dir,
            program=prepared.program,
            args=list(prepared.args),
            env=dict(prepared.env),
            cwd=Path(prepared.run_dir),
            label="評価プロセス",
            record_process=lambda pid, created: backend.record_evaluation_process(
                candidate_id, evaluation_id, pid, created
            ),
            hello_timeout_ms=hello_timeout_ms,
        )
        super().__init__(info, parent)


class FakeEvaluationJob(QObject):
    """モック Backend とテストに評価のイベント列を送る時間差ジョブ。

    対象の item_id は run_spec.json があればそこから、なければ item_ids 引数から取る。
    ファイルは作らない（結果の確定は Backend の conclude_evaluation_run が決める）。
    """

    event_received = Signal(object)
    finished = Signal(object)

    def __init__(
        self,
        prepared: PreparedRun,
        item_ids: list[str] | None = None,
        interval_ms: int = 20,
        parent=None,
    ):
        super().__init__(parent)
        self.prepared = prepared
        candidate_id, _version, evaluation_id = _parse_run_id(prepared.run_id)
        self.candidate_id = candidate_id
        self.evaluation_id = evaluation_id
        self.key = f"evaluation:{candidate_id}"
        try:
            speed = max(0.001, float(os.getenv("FOAM_MOCK_SPEED", "1.0")))
        except ValueError:
            speed = 1.0
        self.interval_ms = max(0, int(interval_ms / speed))
        self.item_ids = list(item_ids or self._spec_item_ids() or ["item_001", "item_002"])
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._seq = 0
        self._events = self._make_events()
        self._done = False

    def _spec_item_ids(self) -> list[str] | None:
        try:
            spec = read_json(Path(self.prepared.run_dir) / "run_spec.json")
            item_ids = spec["validation"]["item_ids"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return list(item_ids) if isinstance(item_ids, list) else None

    def _make_events(self) -> list[dict]:
        total = len(self.item_ids)
        events = [
            {"type": "started", "device": "cpu", "versions": {}, "version_mismatches": []},
            {
                "type": "preflight",
                "n_images": total,
                "per_class_counts": {},
                "required_bytes": 0,
                "free_bytes": 0,
            },
        ]
        for index, item_id in enumerate(self.item_ids, start=1):
            events.append(
                {"type": "image_done", "item_id": item_id, "completed": index, "total": total}
            )
        events.append({"type": "completed", "path": "result.json"})
        return events

    def start(self) -> None:
        self._timer.start(self.interval_ms)

    def _advance(self) -> None:
        if self._done:
            return
        if not self._events:
            self._finish(JobExit(returncode=0))
            return
        self._seq += 1
        event = {
            "v": 1,
            "run_id": self.prepared.run_id,
            "time": time.time(),
            "seq": self._seq,
            **self._events.pop(0),
        }
        self.event_received.emit(event)

    def _finish(self, job_exit: JobExit) -> None:
        if self._done:
            return
        self._done = True
        self._timer.stop()
        self.finished.emit(job_exit)

    def kill(self) -> None:
        self._finish(JobExit(returncode=-1, message="モック評価を中断しました"))


class EvaluationRunner(QObject):
    """候補ごとの評価を順に実行し、占有期間と終端処理を管理する（15.2）。

    - 候補 1 件ごとに ComputeCoordinator の占有を取り、終端処理の後に 1 件ごとに返す
      （その間に学習の要求が来れば、先着順で学習が入る）。
    - 終端処理は runner だけが 1 回だけ行う。終端状態の保存に失敗したら占有を ok=False で返し、
      待機中の残りの候補も取り消す（次を開始しない。解除は次の評価の開始操作で行う）。
    - progressed(candidate_id) はイベントを反映するたび、ended(EvaluationOutcome) は
      候補 1 件の評価が確定する（または開始前に取り消す）たびに出る。
    """

    progressed = Signal(str)
    ended = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self, backend, parent=None, hello_timeout_ms=HELLO_TIMEOUT_MS, compute=None):
        super().__init__(parent)
        self.backend = backend
        self.hello_timeout_ms = hello_timeout_ms
        self.compute = compute if compute is not None else ComputeCoordinator(self)
        # 待ち行列は候補 ID だけ。評価先の検証用の版は候補に固定されている（準備で決まる）
        self._queue: list[str] = []
        self._current: str | None = None
        self._current_version = ""
        self._ticket: Ticket | None = None
        self._waiting_ticket: Ticket | None = None
        self.job = None
        self.evaluation_id: str | None = None
        self._concluded = True
        self._skip_conclude = False
        self._event_failure: str | None = None
        self._busy = False

    # ---- 状態 ----

    @property
    def is_busy(self) -> bool:
        """評価の実行中、占有の待機中、または待ち行列に候補があるか。"""
        return self._current is not None or bool(self._queue)

    @property
    def candidate_id(self) -> str | None:
        """実行中（または占有を待っている）候補。"""
        return self._current

    @property
    def validation_version(self) -> str | None:
        """実行中の評価の検証用の版（準備の後に決まる）。"""
        return self._current_version or None if self._current else None

    @property
    def waiting_for_compute(self) -> bool:
        """学習などの終了を待っていて、まだ準備を始めていないか。"""
        return self._waiting_ticket is not None

    @property
    def queued_candidate_ids(self) -> list[str]:
        """まだ占有を求めていない待機中の候補（先着順）。"""
        return list(self._queue)

    def is_evaluation_active(self, candidate_id: str) -> bool:
        """候補が評価中または評価待ちか（ComparisonService.is_evaluation_active に渡せる）。"""
        return self.candidate_id == candidate_id or candidate_id in self.queued_candidate_ids

    def is_running(self, candidate_id: str) -> bool:
        """候補の評価プロセスが準備・実行中か（占有待ち・待ち行列は含めない）。"""
        return self.candidate_id == candidate_id and self._waiting_ticket is None

    def _update_busy(self) -> None:
        busy = self.is_busy
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    # ---- 開始 ----

    def start(self, candidate_ids: list[str]) -> None:
        """候補を待ち行列に入れ、空いていれば先頭から評価を始める。

        評価先は候補ごとに固定された検証用の版（利用者は選ばない）。すでに評価中・評価待ちの
        候補は重ねて入れない。終端状態の保存失敗で計算処理が止まっているときは、この開始操作で
        解除する（15.1）。
        """
        for candidate_id in candidate_ids:
            if not self.is_evaluation_active(candidate_id) and candidate_id not in self._queue:
                self._queue.append(candidate_id)
        self._update_busy()
        if self.compute.is_blocked:
            self.compute.unblock()
        if self._current is None:
            self._request_next()

    def _request_next(self) -> None:
        if self._current is not None or not self._queue:
            self._update_busy()
            return
        self._current = self._queue.pop(0)
        self._current_version = ""
        candidate_id = self._current
        self._concluded = False
        self._skip_conclude = False
        self._event_failure = None
        self.evaluation_id = None
        self.job = None
        ticket = self.compute.request(
            OWNER, f"評価 {candidate_id}", lambda: self._begin_with_active_ticket(candidate_id)
        )
        if ticket.state == "waiting" and self._current == candidate_id:
            self._waiting_ticket = ticket
        self._update_busy()

    def _begin_with_active_ticket(self, candidate_id: str) -> None:
        if self._current is None or self._current != candidate_id:
            # 取り消し済みの要求。占有だけ返す
            self.compute.release(self.compute.active)
            return
        self._waiting_ticket = None
        self._ticket = self.compute.active
        self._begin()

    def _begin(self) -> None:
        candidate_id = self._current
        try:
            prepared = self.backend.prepare_evaluation_run(candidate_id)
            _candidate, self._current_version, self.evaluation_id = _parse_run_id(prepared.run_id)
            if prepared.fake:
                job = FakeEvaluationJob(prepared, parent=self)
            else:
                job = ProcessEvaluationJob(prepared, self.backend, self.hello_timeout_ms, self)
            self.job = job

            def apply_event(
                event, source=job, candidate=candidate_id, evaluation_id=self.evaluation_id
            ):
                self._apply_event(event, source, candidate, evaluation_id)

            job.event_received.connect(apply_event)
            job.finished.connect(lambda job_exit, source=job: self._job_finished(job_exit, source))
            job.start()
        except Exception as error:
            logger.warning("評価を開始できません (%s): %s", candidate_id, error, exc_info=True)
            self._conclude(JobExit(start_failed=True, message=user_failure_message(error)))

    # ---- 実行中 ----

    def _apply_event(self, event, source_job=None, candidate_id=None, evaluation_id=None) -> None:
        job = self.job if source_job is None else source_job
        candidate_id = self.candidate_id if candidate_id is None else candidate_id
        evaluation_id = self.evaluation_id if evaluation_id is None else evaluation_id
        if self._concluded or job is None or job is not self.job or self._event_failure is not None:
            return
        try:
            self.backend.apply_evaluation_event(candidate_id, evaluation_id, event)
        except Exception as error:
            if self._event_failure is None:
                self._event_failure = str(error) or type(error).__name__
                job.kill()
        finally:
            if candidate_id is not None:
                self.progressed.emit(candidate_id)

    def _job_finished(self, job_exit: JobExit, source_job=None) -> None:
        if source_job is not None and source_job is not self.job:
            return
        if self._skip_conclude:
            return
        if self._event_failure is not None:
            job_exit = replace(
                job_exit,
                message=f"評価イベントを反映できませんでした: {self._event_failure}",
                protocol_error=True,
            )
        self._conclude(job_exit)

    # ---- 終端処理 ----

    def _conclude(self, job_exit: JobExit) -> None:
        if self._concluded or self._current is None:
            return
        self._concluded = True
        candidate_id, version = self._current, self._current_version
        evaluation_id = self.evaluation_id
        saved = True
        try:
            if evaluation_id is not None:
                outcome = self.backend.conclude_evaluation_run(
                    candidate_id, evaluation_id, job_exit
                )
            else:
                outcome = EvaluationOutcome(
                    candidate_id,
                    None,
                    "failed",
                    job_exit.message,
                    "prepare_failed",
                    version,
                )
        except Exception as error:
            saved = False
            logger.error("評価の終了状態を保存できません (%s): %s", candidate_id, error)
            outcome = EvaluationOutcome(
                candidate_id,
                evaluation_id,
                "failed",
                user_failure_message(error),
                "conclusion_failed",
                version,
            )
        self.job = None
        ticket, self._ticket = self._ticket, None
        self._current = None
        cancelled, self._queue = ([], self._queue) if saved else (self._queue, [])
        self._update_busy()
        self.ended.emit(outcome)
        # 保存に失敗したら次を開始しない。待機中の残りの候補も取り消す
        self._emit_cancelled(cancelled)
        # 終端処理の後で占有を返す
        if ticket is not None:
            self.compute.release(ticket, ok=saved, error=None if saved else outcome.message)
        if saved:
            self._request_next()

    # ---- 中断 ----

    def _cancel_queue(self) -> None:
        queued, self._queue = self._queue, []
        self._emit_cancelled(queued)

    def _emit_cancelled(self, queued: list[str]) -> None:
        for candidate_id in queued:
            self.ended.emit(
                EvaluationOutcome(
                    candidate_id, None, "stopped", "開始前に取り消しました", "cancelled"
                )
            )

    def _cancel_waiting(self) -> None:
        ticket, self._waiting_ticket = self._waiting_ticket, None
        candidate_id = self._current
        self._current = None
        self._concluded = True
        self.compute.cancel(ticket)
        self.ended.emit(
            EvaluationOutcome(candidate_id, None, "stopped", "開始前に取り消しました", "cancelled")
        )

    def request_stop(self, reason: str = "user_stop", timeout_ms: int | None = None) -> bool:
        """実行中の 1 件を中断し、待機中の残りも取り消す（15.2）。

        stop_request.json を kill の前に保存する。timeout_ms を渡すと終端処理まで待ち、
        時間内に終わらなければ False を返す（アプリ終了時は終端処理を起動時の復旧に任せる）。
        """
        self._cancel_queue()
        if self._current is None:
            self._update_busy()
            return True
        if self._waiting_ticket is not None:
            self._cancel_waiting()
            self._update_busy()
            return True
        if self.evaluation_id is not None:
            self.backend.request_evaluation_stop(self.candidate_id, self.evaluation_id, reason)
        job = self.job
        if job is None:
            return not self.is_busy
        job.kill()
        if timeout_ms is None:
            return True
        if self._current is not None:
            loop = QEventLoop(self)
            timer = QTimer(self)
            timer.setSingleShot(True)
            self.ended.connect(loop.quit)
            timer.timeout.connect(loop.quit)
            timer.start(timeout_ms)
            loop.exec()
            timer.stop()
            try:
                self.ended.disconnect(loop.quit)
            except (RuntimeError, TypeError):
                pass
        if self._current is not None:
            self._skip_conclude = True
            return False
        return True

    def shutdown(self, timeout_ms: int = APP_EXIT_TIMEOUT_MS) -> bool:
        """アプリ終了時の手順（stop_request.json → kill → 最大 10 秒待つ）。

        確認ダイアログは呼び出し側（画面）が出す。時間内に確定しなければ False。
        """
        return self.request_stop("app_exit", timeout_ms=timeout_ms)
