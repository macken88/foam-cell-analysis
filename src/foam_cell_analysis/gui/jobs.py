"""画面確認用の時間差ジョブ。"""

from __future__ import annotations

import os
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout


class FakeJob(QObject):
    """QTimer で進捗を模擬するジョブ。"""

    progress = Signal(int, int, str)
    finished = Signal(bool, str)

    def __init__(
        self,
        name: str,
        total_steps: int = 10,
        interval_ms: int = 200,
        on_step: Callable[[int], None] | None = None,
        parent: QObject | None = None,
        key: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.name = name
        self.key = key
        self.total_steps = max(1, total_steps)
        self.step = 0
        try:
            speed = max(0.001, float(os.getenv("FOAM_MOCK_SPEED", "1.0")))
        except ValueError:
            speed = 1.0
        self.interval_ms = max(1, int(interval_ms / speed))
        self.on_step = on_step
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._cancelled = False
        self._finished = False

    def start(self) -> FakeJob:
        """ジョブを開始する。"""
        if not self._finished and not self._cancelled:
            self._timer.start(self.interval_ms)
        return self

    def _advance(self) -> None:
        if self._cancelled or self._finished:
            return
        self.step += 1
        if self.on_step:
            self.on_step(self.step)
        self.progress.emit(
            self.step, self.total_steps, f"{self.name}: {self.step}/{self.total_steps}"
        )
        if self.step >= self.total_steps:
            self._timer.stop()
            self._finished = True
            self.finished.emit(True, f"{self.name}が完了しました")

    def cancel(self) -> None:
        """ジョブを中断する。"""
        if self._finished:
            return
        self._cancelled = True
        self._timer.stop()
        self._finished = True
        self.finished.emit(False, f"{self.name}を中断しました")


class JobManager(QObject):
    """実行中ジョブを保持し、件数変更を通知する。"""

    jobs_changed = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs: list[FakeJob] = []

    @property
    def running_count(self) -> int:
        return len(self._jobs)

    def start(self, job: FakeJob) -> FakeJob:
        """ジョブを保持して実行する。"""
        self._jobs.append(job)
        job.finished.connect(lambda _ok, _message, current=job: self._remove(current))
        self.jobs_changed.emit(self.running_count)
        job.start()
        return job

    def _remove(self, job: FakeJob) -> None:
        if job in self._jobs:
            self._jobs.remove(job)
            self.jobs_changed.emit(self.running_count)
            job.deleteLater()

    def jobs(self) -> list[FakeJob]:
        return self._jobs.copy()

    def find(self, key: str) -> FakeJob | None:
        """実行中ジョブをキーで検索する。"""
        return next((job for job in self._jobs if job.key == key), None)


class JobProgressDialog(QDialog):
    """ジョブ進捗を表示する非モーダル対応ダイアログ。"""

    def __init__(self, job: FakeJob, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(job.name)
        layout = QVBoxLayout(self)
        self.message = QLabel("準備中")
        self.bar = QProgressBar()
        self.cancel_button = QPushButton("中断")
        layout.addWidget(self.message)
        layout.addWidget(self.bar)
        layout.addWidget(self.cancel_button)
        self.bar.setRange(0, job.total_steps)
        job.progress.connect(self._progress)
        job.finished.connect(self._finished)
        self.cancel_button.clicked.connect(job.cancel)

    def _progress(self, step: int, total: int, message: str) -> None:
        self.bar.setRange(0, total)
        self.bar.setValue(step)
        self.message.setText(message)

    def _finished(self, _ok: bool, message: str) -> None:
        self.message.setText(message)
        self.cancel_button.setEnabled(False)
