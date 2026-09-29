"""段階 F の hybrid 学習操作を画面経由で確認する。"""

import shutil
import subprocess
import sys
from dataclasses import replace

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from foam_cell_analysis.gui.context import AppContext, StatusBus
from foam_cell_analysis.gui.home_window import HomeWindow
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.navigation import Navigator, PageId
from foam_cell_analysis.gui.shortcuts import ShortcutMap
from foam_cell_analysis.gui.window_manager import WindowManager
from foam_cell_analysis.services.hybrid_backend import HybridBackend
from foam_cell_analysis.training.protocol import read_run_spec, write_run_spec


def _make_shell(qapp, tmp_path, backend):
    context = AppContext(
        backend,
        Navigator(),
        JobManager(),
        StatusBus(),
        ShortcutMap(tmp_path / "keymap.json"),
    )
    manager = WindowManager(context)
    home = HomeWindow(context, manager)
    home.show()
    return context, manager, home


def _queue_experiment(backend, model_type="mask_rcnn"):
    config = backend.default_experiment_config(model_type)
    config["experiment"]["id"] = backend.next_experiment_id()
    config["data"]["cv"]["n_folds"] = 2
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 1
    config["training"]["early_stopping"] = {"enabled": False, "patience": 1}
    config["checkpoint"]["validation_interval"] = 1
    config["checkpoint"]["save_every"] = 1
    return backend.add_training_queue_item(config)


def _child_script(tmp_path, *, exit_code=None):
    path = tmp_path / ("exit_child.py" if exit_code is not None else "wait_child.py")
    error_line = (
        "open(os.path.join(sys.argv[2], 'error.json'), 'w', encoding='utf-8').write("
        "json.dumps({'exception_type':'RuntimeError','message':'意図した失敗'}))"
        if exit_code is not None
        else "time.sleep(0.05)"
    )
    exit_line = f"raise SystemExit({exit_code})" if exit_code is not None else "time.sleep(0.05)"
    repeat_line = "raise SystemExit(0)" if exit_code is not None else "time.sleep(1)"
    path.write_text(
        "import datetime, json, os, sys, time\n"
        "run_id = sys.argv[1]\n"
        "now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()\n"
        "print(json.dumps({'v':1,'run_id':run_id,'time':now(),'type':'hello',"
        "'pid':os.getpid()}), flush=True)\n"
        "if sys.stdin.readline().strip() != 'go': raise SystemExit(3)\n"
        "print(json.dumps({'v':1,'run_id':run_id,'seq':1,'time':now(),"
        "'type':'started','device':'cpu','versions':{},'version_mismatches':[]}), "
        "flush=True)\n"
        f"{error_line}\n"
        f"{exit_line}\n"
        f"{repeat_line}\n",
        encoding="utf-8",
    )
    return path


def _patch_preparer(backend, tmp_path, *, fail_experiment=None, wait_all=True):
    original = backend.prepare_training_run
    wait_script = _child_script(tmp_path)
    fail_script = _child_script(tmp_path, exit_code=9) if fail_experiment else None

    def prepare(experiment_id, queue_id=None, retry=False):
        prepared = original(experiment_id, queue_id, retry)
        spec = read_run_spec(prepared.run_dir)
        spec["config"]["model"]["type"] = "fake"
        write_run_spec(prepared.run_dir, spec)
        if not wait_all and experiment_id != fail_experiment:
            return prepared
        script = fail_script if experiment_id == fail_experiment else wait_script
        return replace(
            prepared,
            program=sys.executable,
            args=[str(script), prepared.run_id, prepared.run_dir],
        )

    backend.prepare_training_run = prepare


def _click_stop_action(qapp, page, monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    page.refresh()
    cell = page.table.item(0, 1)
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.table.visualItemRect(cell).center(),
    )
    QTest.mouseClick(page.more_button, Qt.MouseButton.LeftButton)
    menu = page.more_menu
    action = page.action_map["stop"]
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())


def _wait_for(qapp, predicate, timeout=20_000):
    deadline = 0
    while deadline < timeout:
        QApplication.processEvents()
        if predicate():
            return
        QTest.qWait(50)
        deadline += 50
    assert predicate(), "制限時間内に学習状態が変わりませんでした"


def _accept_yes_when_shown():
    for dialog in QApplication.topLevelWidgets():
        if not isinstance(dialog, QMessageBox):
            continue
        button = dialog.button(QMessageBox.StandardButton.Yes)
        if button is not None:
            button.click()
            return
    QTimer.singleShot(25, _accept_yes_when_shown)


def _setup_hybrid(tmp_path, monkeypatch):
    from tests.training.test_training_process import _workspace

    _workspace(tmp_path, epochs=1, n_items=4)
    shutil.rmtree(tmp_path / "experiments")
    monkeypatch.setenv("FOAM_BACKEND", "mock")
    backend = HybridBackend(tmp_path)
    backend.recover()
    return backend


@pytest.mark.slow
@pytest.mark.parametrize("stop_kind", ["user", "app_exit", "taskkill"])
def test_hybrid_interruption_survives_restart_via_gui(qapp, tmp_path, monkeypatch, stop_kind):
    backend = _setup_hybrid(tmp_path, monkeypatch)
    queued = _queue_experiment(backend)
    _patch_preparer(backend, tmp_path)
    context, manager, home = _make_shell(qapp, tmp_path, backend)
    context.navigator.navigate(PageId.TRAINING_QUEUE)
    queue = manager.page(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    _wait_for(qapp, lambda: context.training_runner.is_busy)
    _wait_for(qapp, lambda: backend.get_experiment(queued.experiment_id).status == "running")

    if stop_kind == "user":
        context.navigator.navigate(PageId.EXPERIMENTS)
        _click_stop_action(qapp, manager.page(PageId.EXPERIMENTS), monkeypatch)
    elif stop_kind == "app_exit":
        QTimer.singleShot(25, _accept_yes_when_shown)
        QTest.keyClick(home, Qt.Key.Key_F4, Qt.KeyboardModifier.AltModifier)
    else:
        _wait_for(qapp, lambda: context.training_runner.job._hello_pid is not None)
        pid = context.training_runner.job._hello_pid
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    _wait_for(qapp, lambda: not context.training_runner.is_busy)
    assert backend.get_experiment(queued.experiment_id).status == "stopped"
    context.navigator.navigate(PageId.EXPERIMENTS)
    old_page = manager.page(PageId.EXPERIMENTS)
    old_page.refresh()
    first_display = old_page.table.item(0, 6).text()
    restarted = HybridBackend(tmp_path)
    restarted.recover()
    new_context, new_manager, new_home = _make_shell(qapp, tmp_path, restarted)
    new_context.navigator.navigate(PageId.EXPERIMENTS)
    new_page = new_manager.page(PageId.EXPERIMENTS)
    new_page.refresh()
    assert new_page.table.item(0, 6).text() == first_display
    assert (tmp_path / "experiments" / queued.experiment_id / "runs" / "attempt_001").exists()
    home.hide()
    for window in manager._windows.values():
        window.close()
    new_home.hide()
    for window in new_manager._windows.values():
        window.close()


@pytest.mark.slow
def test_hybrid_queue_runs_three_jobs_and_continues_after_failure(qapp, tmp_path, monkeypatch):
    backend = _setup_hybrid(tmp_path, monkeypatch)
    queued = [_queue_experiment(backend) for _ in range(3)]
    _patch_preparer(backend, tmp_path, fail_experiment=queued[1].experiment_id, wait_all=False)
    context, manager, home = _make_shell(qapp, tmp_path, backend)
    context.navigator.navigate(PageId.TRAINING_QUEUE)
    queue = manager.page(PageId.TRAINING_QUEUE)
    QTest.mouseClick(queue.run_button, Qt.MouseButton.LeftButton)
    _wait_for(
        qapp,
        lambda: (
            not context.queue_controller.executing
            and all(
                backend.get_experiment(item.experiment_id).status
                in {"completed", "failed", "stopped"}
                for item in queued
            )
        ),
        timeout=60_000,
    )
    states = [backend.get_experiment(item.experiment_id).status for item in queued]
    assert states == ["completed", "failed", "completed"]
    assert not context.training_runner.is_busy
    home.hide()
    for window in manager._windows.values():
        window.close()
