"""GUI が所有する学習プロセスとモック学習ジョブ。"""

from __future__ import annotations

import os
import time
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from ..services.models import JobExit, PreparedRun
from .process_job import HELLO_TIMEOUT_MS, ProcessJob, ProcessJobInfo


class ProcessTrainingJob(ProcessJob):
    """hello/go ハンドシェイク付きの学習子プロセス（共通 ProcessJob の学習用設定）。

    experiment_id・attempt を省略したときだけ run_id（<実験>/attempt_<番号>）から求める。
    """

    def __init__(
        self,
        prepared: PreparedRun,
        backend,
        hello_timeout_ms: int = HELLO_TIMEOUT_MS,
        parent=None,
        *,
        experiment_id: str | None = None,
        attempt: int | None = None,
    ):
        if experiment_id is None or attempt is None:
            parsed_id, parsed_attempt = prepared.run_id.rsplit("/attempt_", 1)
            experiment_id = experiment_id or parsed_id
            attempt = attempt if attempt is not None else int(parsed_attempt)
        self.prepared = prepared
        self.backend = backend
        self.experiment_id = experiment_id
        self.attempt = attempt
        self.key = f"training:{experiment_id}"
        info = ProcessJobInfo(
            run_id=prepared.run_id,
            run_dir=prepared.run_dir,
            program=prepared.program,
            args=list(prepared.args),
            env=dict(prepared.env),
            cwd=Path(prepared.run_dir).resolve().parents[1],
            label="学習プロセス",
            record_process=lambda pid, created: backend.record_training_process(
                experiment_id, attempt, pid, created
            ),
            hello_timeout_ms=hello_timeout_ms,
        )
        super().__init__(info, parent)


class FakeTrainingJob(QObject):
    """モック Backend にイベント列を送る時間差ジョブ。"""

    event_received = Signal(object)
    finished = Signal(object)

    def __init__(self, prepared: PreparedRun, experiment, interval_ms=30, parent=None):
        super().__init__(parent)
        self.prepared = prepared
        self.experiment = experiment
        self.experiment_id = experiment.experiment_id
        self.key = f"training:{self.experiment_id}"
        self.interval_ms = interval_ms
        try:
            speed = max(0.001, float(os.getenv("FOAM_MOCK_SPEED", "1.0")))
        except ValueError:
            speed = 1.0
        self.interval_ms = max(0, int(self.interval_ms / speed))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._step = 0
        self.step = 0
        self._seq = 0
        self._events = self._make_events()
        self.total_steps = max(1, self.experiment.total_epochs) * (
            max(1, len(set(self.experiment.fold_assignments.values()))) + 1
        )

    def _make_events(self):
        epochs = max(1, self.experiment.total_epochs)
        folds = max(1, len(set(self.experiment.fold_assignments.values())))
        events = [{"type": "phase", "phase": "cv"}]
        for fold in range(1, folds + 1):
            for epoch in range(1, epochs + 1):
                loss = max(0.02, 1.2 * (1 - epoch / epochs))
                events.append(
                    {"type": "epoch", "phase": "cv", "fold": fold, "epoch": epoch, "loss": loss}
                )
                if epoch % 5 == 0 or epoch == epochs:
                    events.append(
                        {
                            "type": "val",
                            "fold": fold,
                            "epoch": epoch,
                            "ap": min(0.99, 0.35 + 0.6 * epoch / epochs),
                        }
                    )
        events.append(
            {
                "type": "oof",
                "epoch": epochs,
                "ap": 0.8,
                "per_fold": {str(fold): 0.8 for fold in range(1, folds + 1)},
            }
        )
        events.append({"type": "selected", "epoch": epochs, "ap": 0.8, "per_class": {}})
        events.append({"type": "checkpoint", "epoch": epochs, "path": "selected.pt"})
        events.append({"type": "phase", "phase": "final"})
        for epoch in range(1, epochs + 1):
            events.append(
                {
                    "type": "epoch",
                    "phase": "final",
                    "epoch": epoch,
                    "loss": max(0.02, 1.2 * (1 - epoch / epochs)),
                }
            )
        events.append({"type": "checkpoint", "epoch": epochs, "path": "final.pt"})
        return events

    def start(self):
        self._timer.start(self.interval_ms)

    def _advance(self):
        if self._step >= len(self._events):
            self._timer.stop()
            self.finished.emit(JobExit(returncode=0))
            return
        self._seq += 1
        event = {
            "v": 1,
            "run_id": self.prepared.run_id,
            "time": time.time(),
            "seq": self._seq,
            **self._events[self._step],
        }
        self._step += 1
        if event["type"] == "epoch":
            self.step += 1
        self.event_received.emit(event)

    def kill(self):
        self._timer.stop()
        self.finished.emit(JobExit(returncode=-1, message="モック学習を中断しました"))

    def cancel(self):
        """旧画面テストとの互換用に停止操作を別名で提供する。"""
        parent = self.parent()
        if hasattr(parent, "request_stop"):
            parent.request_stop("user_stop")
        else:
            self.kill()
