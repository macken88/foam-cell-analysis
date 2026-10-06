"""Qt signal と状態述語を組み合わせた小さな同期待ち。"""

from collections.abc import Callable

from PySide6.QtCore import QEventLoop, QProcess, QTimer


def wait_until(
    predicate: Callable[[], bool], signal, timeout_ms: int, start: Callable[[], None]
) -> bool:
    """開始後に状態を即確認し、signalまたはtimeoutまでイベント処理しながら待つ。"""
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)

    def check(*_args) -> None:
        if predicate():
            loop.quit()

    signal.connect(check)
    try:
        start()
        if not predicate():
            timer.timeout.connect(loop.quit)
            timer.start(timeout_ms)
            loop.exec()
        return bool(predicate())
    finally:
        timer.stop()
        try:
            signal.disconnect(check)
        except (RuntimeError, TypeError):
            pass
        timer.deleteLater()
        loop.deleteLater()


def disconnect_runner_slots(job) -> None:
    """Runner自身が登録した2つのslotだけを外す。外部observerは維持する。"""
    try:
        slots = getattr(job, "_runner_slots", ())
        for signal_name, slot in slots:
            try:
                getattr(job, signal_name).disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        job._runner_slots = ()
    except (RuntimeError, AttributeError):
        pass


def job_exit_confirmed(job_exit, job) -> bool:
    """破棄可能な正常終了を確認する。未知状態や生存プロセスは保護する。"""
    try:
        if job is None or not callable(getattr(job, "deleteLater", None)):
            return False
        if getattr(job_exit, "process_alive", False):
            return False
        process = getattr(job, "process", None)
        if process is None:
            return True
        return process.state() == QProcess.ProcessState.NotRunning
    except (RuntimeError, AttributeError):
        return False
