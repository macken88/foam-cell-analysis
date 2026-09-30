"""比較・リリース画面の動作確認（比較・推論設計 16.2）。"""

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox

from foam_cell_analysis.gui.context import AppContext
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.modes.comparison.candidates_page import CandidatesPage
from foam_cell_analysis.gui.modes.comparison.dialogs import (
    CandidateDialog,
    EvaluationDialog,
    MaskExportDialog,
    MaskExportDoneDialog,
    ReleaseDialog,
)
from foam_cell_analysis.gui.modes.comparison.mask_compare import MaskComparisonPage
from foam_cell_analysis.gui.navigation import ModeId, Navigator, PageId
from foam_cell_analysis.services.mock.backend import MockBackend
from foam_cell_analysis.services.models import RunAttempt


def make_context() -> AppContext:
    return AppContext(MockBackend(), Navigator(), JobManager())


def _row(page, candidate_id):
    return next(
        row
        for row in range(page.table.rowCount())
        if page.table.item(row, 1).text() == candidate_id
    )


def _click_checkbox(page, candidate_id):
    rect = page.table.visualItemRect(page.table.item(_row(page, candidate_id), 0))
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=rect.topLeft() + QPoint(12, rect.height() // 2),
    )


def _click_menu_item(window, top_label, item_label):
    """メニューバーのメニューを開き、項目をマウスでクリックする。"""
    menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text().startswith(top_label)
    )
    action = next(item for item in menu.actions() if item.text().split("\t", 1)[0] == item_label)
    assert action.isEnabled(), action.toolTip()
    menu.popup(window.mapToGlobal(QPoint(0, 0)))
    QTest.qWaitForWindowExposed(menu)
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())


def _candidates(shell):
    shell.navigate(PageId.CANDIDATES)
    window = shell.manager.window(ModeId.COMPARISON)
    window.resize(1400, 900)
    page = shell.page(PageId.CANDIDATES)
    page.refresh()
    return window, page


def test_duplicate_candidate_raises():
    ctx = make_context()
    config = ctx.backend.create_inference_config("mask_rcnn", {"box_score_thresh": 0.3})
    ctx.backend.add_candidate("exp_0042", 1, config.config_id)
    with pytest.raises(ValueError, match="登録済み"):
        ctx.backend.add_candidate("exp_0042", 1, config.config_id)


def test_mask_comparison_slots_and_navigation(qtbot):
    ctx = make_context()
    page = MaskComparisonPage(ctx)
    qtbot.addWidget(page)
    page.on_enter({"validation_version": "val_v003", "candidate_ids": ["RC-001", "RC-002"]})
    assert len(page.views) == 3
    assert len(page.items) > 1
    original = page.index
    QTest.mouseClick(page.next, Qt.MouseButton.LeftButton)
    assert page.index == (original + 1) % len(page.items)
    assert not page.views[0].scene().items() == []
    assert all(view.horizontalScrollBarPolicy().name == "ScrollBarAlwaysOff" for view in page.views)
    # 原画像スロットと各候補の予測スロットを描画し、画像移動後も表示を更新する。
    assert len(page.views) == 1 + len(page.candidate_ids)
    assert page.views[0].scene().items()
    assert all(view.scene().items() for view in page.views[1:])
    page._move(1)
    assert all(view.scene().items() for view in page.views)


def test_mask_comparison_refuses_five_candidates_without_truncation(qtbot):
    ctx = make_context()
    page = MaskComparisonPage(ctx)
    qtbot.addWidget(page)
    ids = ["RC-001", "RC-002", "RC-003", "RC-001", "RC-002"]
    page.on_enter({"validation_version": "val_v003", "candidate_ids": ids})
    assert page.candidate_ids == []
    assert "2〜4 件" in page.block_reason


def test_comparison_entry_resolves_candidate_fixed_version_when_not_supplied(qtbot, monkeypatch):
    ctx = make_context()
    monkeypatch.setattr(
        ctx.backend,
        "list_validation_versions",
        lambda: [type("Version", (), {"version": "val_v004"})()],
    )
    page = MaskComparisonPage(ctx)
    qtbot.addWidget(page)
    page.on_enter({"candidate_ids": ["RC-001", "RC-002"]})
    assert page.validation == "val_v003"
    assert not page.block_reason


def test_mask_comparison_shortcuts_update_every_slot_and_status(qtbot):
    ctx = make_context()
    page = MaskComparisonPage(ctx)
    qtbot.addWidget(page)
    page.on_enter({"candidate_ids": ["RC-001", "RC-002"]})
    messages = []
    ctx.status.message.connect(messages.append)
    page.views[0].setFocus()
    QTest.keyClick(page.views[0], Qt.Key.Key_M)
    assert not page.display_toggle.is_alternate
    start = page.index
    QTest.keyClick(page.views[0], Qt.Key.Key_Right)
    assert page.index == (start + 1) % len(page.items)
    assert all(not view.scene().items() == [] for view in page.views)


def test_candidate_table_defaults_to_all_versions_and_uses_checkboxes(qtbot):
    ctx = make_context()
    page = CandidatesPage(ctx)
    qtbot.addWidget(page)
    assert page.validation.currentText() == "すべて"
    assert page.table.columnCount() == 12
    assert page.table.horizontalHeaderItem(0).text() == "選択"
    assert page.table.horizontalHeaderItem(7).text() == "検証 AP"
    assert page.table.horizontalHeaderItem(8).text() == "OOF AP"
    page.resize(1200, 700)
    page.show()
    QTest.qWait(50)
    _click_checkbox(page, page.table.item(0, 1).text())
    assert len(page._selected()) == 1


def test_validation_filter_hides_candidates_fixed_to_other_version(qtbot):
    ctx = make_context()
    ctx.backend.get_candidate("RC-003").validation_version = "val_v002"
    page = CandidatesPage(ctx)
    qtbot.addWidget(page)
    page.validation.setCurrentIndex(page.validation.findData("val_v003"))
    assert page.table.rowCount() == 2
    assert {page.table.item(row, 6).text() for row in range(page.table.rowCount())} == {"val_v003"}


def test_add_config_action_requires_exactly_one_selected_candidate(qtbot):
    ctx = make_context()
    page = CandidatesPage(ctx)
    qtbot.addWidget(page)
    action = page.candidate_actions["add_config"]
    assert not action.isEnabled()
    assert "1 件" in action.toolTip()
    _click_checkbox(page, "RC-003")
    assert action.isEnabled()
    _click_checkbox(page, "RC-001")
    assert not action.isEnabled()


def test_add_config_presets_values_and_keeps_source_candidate(shell, qtbot, monkeypatch):
    window, page = _candidates(shell)
    source = shell.ctx.backend.get_candidate("RC-001")
    source_config = next(
        item
        for item in shell.ctx.backend.list_inference_configs()
        if item.config_id == source.inference_config_id
    )
    original_ids = {item.candidate_id for item in shell.ctx.backend.list_candidates()}
    _click_checkbox(page, "RC-001")

    def add_changed(dialog):
        dialog.show()
        assert dialog.use_new.isChecked()
        assert dialog.experiment.currentText() == source.experiment_id
        assert dialog.attempt.currentData() == source.source_attempt_number
        for key, value in source_config.params.items():
            if key in dialog.fields:
                assert dialog.fields[key].value() == pytest.approx(value)
        field = dialog.fields["box_score_thresh"]
        field.setValue(field.value() + 0.05)
        QTest.mouseClick(dialog.ok_button, Qt.MouseButton.LeftButton)
        return dialog.result()

    monkeypatch.setattr(CandidateDialog, "exec", add_changed)
    QTest.mouseClick(page.buttons["add_config"], Qt.MouseButton.LeftButton)
    candidates = shell.ctx.backend.list_candidates()
    added = [item for item in candidates if item.candidate_id not in original_ids]
    assert len(added) == 1
    assert source.candidate_id in {item.candidate_id for item in candidates}
    assert added[0].experiment_id == source.experiment_id
    assert added[0].source_attempt_number == source.source_attempt_number
    assert added[0].inference_config_id != source.inference_config_id


def test_oof_ap_is_shown_only_when_inference_matches_training(qtbot):
    ctx = make_context()
    backend = ctx.backend
    config = backend.create_inference_config(
        "mask_rcnn", backend.default_inference_params("mask_rcnn")
    )
    matching = backend.add_candidate("exp_0042", 1, config.config_id)
    backend.get_candidate("RC-003").oof_reason = "推論設定が学習時と異なります"
    page = CandidatesPage(ctx)
    qtbot.addWidget(page)
    seeded = page.table.item(_row(page, "RC-003"), 8)
    assert seeded.text() == "対象外"
    assert seeded.toolTip() == "推論設定が学習時と異なります"
    value = page.table.item(_row(page, matching.candidate_id), 8).text()
    assert value not in {"対象外", "—"}
    float(value)


def test_evaluate_from_menu_completes_and_shows_validation_ap(shell, qtbot):
    window, page = _candidates(shell)
    assert page.table.item(_row(page, "RC-003"), 7).text() == "未評価"
    _click_checkbox(page, "RC-003")
    _click_menu_item(window, "候補", "評価実行")
    qtbot.waitUntil(lambda: not shell.ctx.evaluation_runner.is_busy, timeout=5000)
    qtbot.waitUntil(lambda: page.table.item(_row(page, "RC-003"), 7).text() != "未評価")
    row = _row(page, "RC-003")
    float(page.table.item(row, 7).text())
    assert page.table.item(row, 9).text() == "候補"
    assert page.table.item(row, 0).checkState() == Qt.CheckState.Checked


def test_same_event_loop_refresh_requests_are_coalesced(qtbot):
    ctx = make_context()
    page = CandidatesPage(ctx)
    qtbot.addWidget(page)
    list_candidates = ctx.backend.list_candidates
    refresh_reads = []

    def counted_list_candidates():
        refresh_reads.append(None)
        return list_candidates()

    ctx.backend.list_candidates = counted_list_candidates
    page.runner.busy_changed.emit(True)
    page.runner.busy_changed.emit(False)
    qtbot.waitUntil(lambda: len(refresh_reads) == 1, timeout=1000)
    assert len(refresh_reads) == 1


def test_stop_evaluation_from_menu(shell, qtbot, monkeypatch):
    monkeypatch.setenv("FOAM_MOCK_SPEED", "0.2")
    window, page = _candidates(shell)
    runner = shell.ctx.evaluation_runner
    outcomes = []
    runner.ended.connect(outcomes.append)
    _click_checkbox(page, "RC-003")
    _click_menu_item(window, "候補", "評価実行")
    assert runner.is_busy
    qtbot.waitUntil(lambda: page.table.item(_row(page, "RC-003"), 9).text().startswith("評価中"))
    _click_menu_item(window, "候補", "評価中止")
    qtbot.waitUntil(lambda: not runner.is_busy, timeout=5000)
    assert [outcome.status for outcome in outcomes] == ["stopped"]
    row = _row(page, "RC-003")
    assert page.table.item(row, 7).text() == "未評価"
    assert page.table.item(row, 9).text() == "候補"
    assert not page.candidate_actions["stop"].isEnabled()


def test_add_dialog_requires_attempt_and_creates_candidate(shell, qtbot, monkeypatch):
    backend = shell.ctx.backend
    experiment = backend.get_experiment("exp_0042")
    experiment.runs.append(RunAttempt(2, experiment.runs[0].started_at, result="completed"))
    window, page = _candidates(shell)
    observed = {}

    def interact(dialog):
        dialog.show()
        dialog.experiment.setCurrentText("exp_0042")
        observed.setdefault("disabled_before_choice", not dialog.ok_button.isEnabled())
        assert dialog.attempt.itemText(0) == "試行 1 / final.pt"
        dialog.attempt.setFocus()
        QTest.keyClick(dialog.attempt, Qt.Key.Key_Down)
        QTest.keyClick(dialog.attempt, Qt.Key.Key_Down)
        assert dialog.attempt.currentData() == 2
        assert dialog.ok_button.isEnabled()
        assert dialog.fields["box_detections_per_img"].maximum() == 1000
        QTest.mouseClick(dialog.use_new, Qt.MouseButton.LeftButton)
        QTest.mouseClick(dialog.ok_button, Qt.MouseButton.LeftButton)
        observed.setdefault("results", []).append(dialog.result())
        observed.setdefault("errors", []).append(
            dialog.error.text() if dialog.error.isVisible() else ""
        )
        return dialog.result()

    monkeypatch.setattr(CandidateDialog, "exec", interact)
    _click_menu_item(window, "候補", "候補追加")
    created = [c for c in backend.list_candidates() if c.source_attempt_number == 2]
    assert observed["disabled_before_choice"]
    assert len(created) == 1
    assert observed["results"] == [QDialog.DialogCode.Accepted]
    page.refresh()
    assert page.table.item(_row(page, created[0].candidate_id), 4).text() == "試行 2 / final.pt"

    # 同じ試行・同じ推論設定なら、登録済み候補を一覧で案内する
    _click_menu_item(window, "候補", "候補追加")
    assert observed["results"][-1] == QDialog.DialogCode.Accepted
    assert len(page._selected()) == 1
    assert page._selected()[0].source_attempt_number == 2
    assert len([c for c in backend.list_candidates() if c.source_attempt_number == 2]) == 1


def _evaluate_open_candidate(backend, page, candidate_id="RC-003"):
    """リリース前の候補（RC-003）の評価を模擬イベントで完了させ、表を読み直す。"""
    version = backend.get_candidate(candidate_id).validation_version
    prepared = backend.prepare_evaluation_run(candidate_id)
    evaluation_id = prepared.run_id.rsplit("/", 1)[1]
    event = {"v": 1, "run_id": prepared.run_id, "seq": 1, "time": 0, "type": "completed"}
    backend.apply_evaluation_event(candidate_id, evaluation_id, event)
    backend.conclude_evaluation_run(candidate_id, evaluation_id)
    page.refresh()
    return backend.get_candidate_evaluation(candidate_id, version)


def test_detail_dialog_opens_and_saves_external_result(shell, qtbot, monkeypatch):
    window, page = _candidates(shell)
    backend = shell.ctx.backend
    record = _evaluate_open_candidate(backend, page)
    _click_checkbox(page, "RC-003")
    observed = {}

    def save(dialog):
        dialog.show()
        observed["headers"] = [
            dialog.metrics.horizontalHeaderItem(column).text()
            for column in range(dialog.metrics.columnCount())
        ]
        observed["rows"] = [dialog.metrics.item(row, 0).text() for row in range(4)]
        dialog.results.item(0, 2).setText("12.3")
        item_id = dialog.results.item(0, 0).text()
        dialog.paste.setPlainText(f"{item_id}\tnan")
        monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
        dialog._apply_external_paste()
        assert dialog.results.item(0, 2).text() == "12.3"
        QTest.keyClicks(dialog.software, "ImageJ")
        QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
        return dialog.result()

    monkeypatch.setattr(EvaluationDialog, "exec", save)
    _click_menu_item(window, "候補", "評価詳細")
    assert observed["headers"][:2] == ["評価項目", "全体"]
    classes = list(record.evaluation.per_class)
    assert observed["headers"][2 : 2 + len(classes)] == classes
    assert observed["rows"][0] == "検証 AP"
    assert observed["rows"][2] == "学習時 OOF AP（参考）"
    saved = backend.list_external_results("RC-003")[-1]
    assert saved["evaluation_id"] == record.evaluation_id
    assert saved["software"] == "ImageJ"
    assert (
        saved["values"][backend.list_validation_items(record.validation_version)[0].item_id] == 12.3
    )
    assert page.table.item(_row(page, "RC-003"), 10).text().startswith("12.3")


def test_mask_export_dialog_completes_and_shows_summary(shell, qtbot, monkeypatch, tmp_path):
    window, page = _candidates(shell)
    _click_checkbox(page, "RC-001")
    _click_checkbox(page, "RC-002")
    summaries = []

    def export(dialog):
        dialog.show()
        assert not dialog.start_button.isEnabled()
        QTest.keyClicks(dialog.output_dir, str(tmp_path))
        QTest.mouseClick(dialog.start_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: dialog.summary is not None, timeout=5000)
        return dialog.result()

    def done(dialog):
        summaries.append(dialog.summary)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(MaskExportDialog, "exec", export)
    monkeypatch.setattr(MaskExportDoneDialog, "exec", done)
    _click_menu_item(window, "ファイル", "抽出結果出力")
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["n_candidates"] == 2
    assert summary["n_images"] > 0
    dialog = MaskExportDoneDialog(summary)
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "粒子解析用抽出結果を出力しました"
    assert dialog.open_button.text() == "フォルダを開く"


def test_release_flow_via_dialog(shell, qtbot, monkeypatch):
    window, page = _candidates(shell)
    backend = shell.ctx.backend
    record = _evaluate_open_candidate(backend, page)
    _click_checkbox(page, "RC-003")
    observed = {}

    def release(dialog):
        dialog.show()
        observed["evaluation_id"] = dialog.evaluation_id
        QTest.mouseClick(dialog.ok_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: dialog.model is not None, timeout=5000)
        return dialog.result()

    monkeypatch.setattr(ReleaseDialog, "exec", release)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.No
    )
    QTest.mouseClick(page.buttons["release"], Qt.MouseButton.LeftButton)
    assert observed["evaluation_id"] == record.evaluation_id
    assert backend.get_candidate("RC-003").status == "released"
    assert page.table.item(_row(page, "RC-003"), 9).text() == "リリース済み"


def test_release_dialog_refuses_found_contamination(qtbot):
    ctx = make_context()
    record = ctx.backend.get_candidate_evaluation("RC-001", "val_v003")
    record.contamination = {"status": "found", "pairs": [["a", "b"]], "reason": None}
    record.schema = 1
    dialog = ReleaseDialog(ctx, ctx.backend.get_candidate("RC-001"), record)
    qtbot.addWidget(dialog)
    assert not dialog.ok_button.isEnabled()
    assert "リリースできません" in dialog.ok_button.toolTip()


def test_blocked_compute_disables_evaluation(shell):
    _window, page = _candidates(shell)
    _click_checkbox(page, "RC-003")
    assert page.candidate_actions["evaluate"].isEnabled()
    shell.ctx.compute.block("前回のプロセスが残っています")
    assert not page.candidate_actions["evaluate"].isEnabled()
    assert page.candidate_actions["evaluate"].toolTip() == "前回のプロセスが残っています"


def test_release_button_explains_missing_evaluation(qtbot):
    ctx = make_context()
    page = CandidatesPage(ctx)
    qtbot.addWidget(page)
    page.table.item(_row(page, "RC-003"), 0).setCheckState(Qt.CheckState.Checked)

    assert not page.buttons["release"].isEnabled()
    assert page.buttons["release"].toolTip() == "評価済みの候補を 1 つ選ぶとリリースできます"


def test_candidate_dialog_builds_model_specific_fields(qtbot):
    ctx = make_context()
    dialog = CandidateDialog(ctx)
    qtbot.addWidget(dialog)
    assert dialog.model_type.text() == "Mask R-CNN"
    assert set(dialog.fields) == {
        "box_score_thresh",
        "box_nms_thresh",
        "box_detections_per_img",
    }
    assert dialog.fields["box_detections_per_img"].value() == 300
    assert not dialog.fields["box_score_thresh"].isEnabled()
    dialog.use_new.setChecked(True)
    assert dialog.fields["box_score_thresh"].isEnabled()
    dialog.experiment.setCurrentText("exp_0043")
    assert dialog.model_type.text() == "Cellpose"
    assert set(dialog.fields) == {"cellprob_threshold", "flow_threshold"}
    assert dialog.fields["cellprob_threshold"].minimum() == -6.0


def test_candidate_page_refreshes_validation_versions_from_home(shell, qapp):
    shell.navigate(PageId.CANDIDATES)
    page = shell.page(PageId.CANDIDATES)
    for candidate in shell.ctx.backend.get_working_items():
        if candidate.usage in {"train", "val"}:
            candidate.classification = candidate.classification or "分類A"
            candidate.quality = candidate.quality or "良"
            if not candidate.mask_revisions:
                candidate.mask_revisions = ["rev_001"]
                candidate.selected_mask_revision = "rev_001"
    item = next(item for item in shell.ctx.backend.get_working_items() if item.usage == "train")
    shell.ctx.backend.update_item("all", item.item_id, usage="val")
    shell.ctx.backend.finalize_working_dataset("検証用版を追加")
    window = shell.manager.window(ModeId.COMPARISON)
    assert window.tabs.currentIndex() == 0
    QTest.mouseClick(shell.home.pipeline._stages[2], Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert window.tabs.currentIndex() == 0
    assert page.validation.findText("val_v004") >= 0


def test_candidate_menu_actions_disable_without_selection(qapp, qtbot):
    page = CandidatesPage(AppContext(MockBackend(), Navigator(), JobManager()))
    qtbot.addWidget(page)
    page.show()
    assert not page.candidate_actions["export"].isEnabled()
    assert not page.candidate_actions["reject"].isEnabled()
    assert not page.candidate_actions["stop"].isEnabled()
    page.candidate_actions["export"].trigger()
    assert not page.candidate_actions["export"].isEnabled()


def test_detail_dialog_is_read_only_for_released_candidate(qtbot):
    ctx = make_context()
    candidate = ctx.backend.get_candidate("RC-001")
    candidate.status = "released"
    record = ctx.backend.get_candidate_evaluation("RC-001", "val_v003")
    dialog = EvaluationDialog(ctx, candidate, record)
    qtbot.addWidget(dialog)
    assert not dialog.save_button.isEnabled()
    assert dialog.controls.button(QDialogButtonBox.StandardButton.Close).isEnabled()


def test_candidate_dialog_without_existing_config_defaults_to_new(qtbot):
    ctx = make_context()
    ctx.backend.inference_configs.clear()
    dialog = CandidateDialog(ctx)
    qtbot.addWidget(dialog)

    assert dialog.use_new.isChecked()
    assert not dialog.use_existing.isEnabled()
    assert dialog.ok_button.isEnabled()
    dialog.use_existing.setEnabled(True)
    dialog.use_existing.setChecked(True)
    assert not dialog.ok_button.isEnabled()


def test_candidate_dialog_skips_experiment_whose_final_model_was_pruned(qtbot):
    ctx = make_context()
    for run in ctx.backend.get_experiment("exp_0042").runs:
        ctx.backend._pruned.setdefault(("exp_0042", run.attempt), set()).add("final")
    dialog = CandidateDialog(ctx)
    qtbot.addWidget(dialog)

    assert "exp_0042" not in dialog.experiments


def test_failed_add_does_not_leave_a_new_inference_config(qtbot):
    ctx = make_context()
    dialog = CandidateDialog(ctx, preset={"experiment_id": "exp_0042"})
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.use_new.setChecked(True)
    dialog.fields["box_score_thresh"].setValue(0.11)
    before = len(ctx.backend.list_inference_configs())
    for run in ctx.backend.get_experiment("exp_0042").runs:
        ctx.backend._pruned.setdefault(("exp_0042", run.attempt), set()).add("final")

    QTest.mouseClick(dialog.ok_button, Qt.MouseButton.LeftButton)

    assert len(ctx.backend.list_inference_configs()) == before
    assert dialog.error.isVisible()


def test_evaluation_failure_is_shown_with_japanese_reason(qtbot):
    from foam_cell_analysis.services.models import EvaluationOutcome

    ctx = make_context()
    page = CandidatesPage(ctx)
    qtbot.addWidget(page)
    page.show()

    ctx.evaluation_runner.ended.emit(
        EvaluationOutcome(
            "RC-003",
            None,
            "failed",
            "評価の準備に失敗しました。作業フォルダのパスが長すぎる可能性があります。",
            "prepare_failed",
        )
    )
    assert page.failure_note.isVisible()
    assert "作業フォルダのパスが長すぎる" in page.failure_note.text()

    ctx.evaluation_runner.ended.emit(
        EvaluationOutcome("RC-003", None, "failed", r"C:\work\x\y raw error", "error")
    )
    assert "C:" not in page.failure_note.text()
    assert "raw error" not in page.failure_note.text()


def test_mask_export_is_in_candidate_menu_and_context_menu(qtbot):
    page = CandidatesPage(make_context())
    qtbot.addWidget(page)
    export = page.candidate_actions["export"]

    assert export in page.menu_action_groups()["candidate"]
    assert export in page.menu_action_groups()["file"]
    assert export in page.context_menu.actions()


def test_running_evaluation_is_counted_as_a_running_job(qtbot):
    from PySide6.QtCore import QObject, Signal

    class _Runner(QObject):
        busy_changed = Signal(bool)
        is_busy = False

    runner = _Runner()
    jobs = JobManager()
    jobs.set_evaluation_runner(runner)
    counts = []
    jobs.jobs_changed.connect(counts.append)

    runner.is_busy = True
    runner.busy_changed.emit(True)

    assert jobs.running_count == 1
    assert counts[-1] == 1


def test_export_done_dialog_elides_long_paths(qtbot):
    from foam_cell_analysis.gui.modes.comparison.dialogs import MaskExportDoneDialog

    folder = "C:/" + "/".join(["very_long_folder_name"] * 12)
    dialog = MaskExportDoneDialog(
        {
            "folder": folder,
            "n_candidates": 1,
            "n_images": 2,
            "vanished_count": 0,
            "vanished_images": 0,
            "split_count": 0,
            "split_images": 0,
        }
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.folder_label.toolTip() == folder
    assert "…" in dialog.folder_label.text()
    assert dialog.width() < 900


def test_progress_updates_do_not_reread_candidate_files(qapp, qtbot, tmp_path, monkeypatch):
    """hybrid の実ファイル: 進捗通知では候補・データセット・評価のファイルを読み直さない。

    完了評価を持つ 2 候補を評価し、対象行の進捗・評価待ちを正しく出し、終了後は AP・壊れ表示・
    操作可否を新しい状態にする。
    """
    import builtins
    import io
    import json

    from foam_cell_analysis.gui.modes.comparison.candidates_page import BROKEN_TEXT
    from foam_cell_analysis.services.hybrid_backend import HybridBackend
    from foam_cell_analysis.services.models import (
        EvaluationOutcome,
        EvaluationProgress,
        PreparedRun,
    )
    from tests.gui.test_stage_f_acceptance import _make_shell
    from tests.services.test_comparison_service import FakeTraining, _dataset, _write_evaluation

    _dataset(tmp_path, "train_v000", "train", ["分類A", "分類B"], "val_v000")
    _dataset(tmp_path, "val_v000", "val", ["分類A", "分類C"])
    fake = FakeTraining(tmp_path)
    fake.add_attempt("exp_0001", 1)
    fake.add_attempt("exp_0001", 2, weights=b"weights-2")
    backend = HybridBackend(tmp_path, process_alive=lambda _record: False)
    backend.training.create_candidate_snapshot = fake.create_candidate_snapshot
    config = backend.create_inference_config(
        "mask_rcnn", backend.default_inference_params("mask_rcnn")
    )
    first, second = (
        backend.add_candidate("exp_0001", attempt, config.config_id, "").candidate_id
        for attempt in (1, 2)
    )
    for candidate_id in (first, second):
        _write_evaluation(backend.comparison, candidate_id, "eval_001")

    # 評価プロセスの代わりに偽の評価ジョブを使う（進捗はメモリ上に持つ）
    progress: dict[str, EvaluationProgress] = {}

    def prepare(candidate_id):
        version = backend.get_candidate(candidate_id).validation_version
        run_dir = tmp_path / "fake_runs" / candidate_id
        run_dir.mkdir(parents=True)
        return PreparedRun(f"{candidate_id}/{version}/eval_002", str(run_dir), "", [], {}, True)

    def apply_event(candidate_id, evaluation_id, event):
        if event["type"] == "image_done":
            progress[candidate_id] = EvaluationProgress(
                candidate_id, evaluation_id, "val_v000", event["completed"], event["total"]
            )

    def conclude(candidate_id, evaluation_id, _job_exit):
        progress.pop(candidate_id, None)
        run_dir = _write_evaluation(backend.comparison, candidate_id, evaluation_id)
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if candidate_id == first:
            result["overall"]["ap"] = 1.5  # 壊れた評価結果
        else:
            result["overall"]["ap"] = 0.9
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return EvaluationOutcome(candidate_id, evaluation_id, "completed")

    monkeypatch.setattr(backend, "prepare_evaluation_run", prepare)
    monkeypatch.setattr(backend, "apply_evaluation_event", apply_event)
    monkeypatch.setattr(backend, "conclude_evaluation_run", conclude)
    monkeypatch.setattr(backend, "get_evaluation_progress", progress.get)

    context, manager, home = _make_shell(qapp, tmp_path, backend)
    runner = context.evaluation_runner
    # 画面より前と後に接続し、画面の progressed 処理の間だけファイルのオープンを記録する
    recording = []
    opened: list[str] = []
    snapshots: list[dict[str, str]] = []
    runner.progressed.connect(lambda _cid: recording.append(True))
    context.navigator.navigate(PageId.CANDIDATES)
    window = manager.window(ModeId.COMPARISON)
    window.resize(1400, 900)
    page = manager.page(PageId.CANDIDATES)
    page.validation.setCurrentText("val_v000")
    assert [page.table.item(row, 7).text() for row in range(2)] == ["0.600", "0.600"]

    def stop_recording(_cid):
        recording.clear()
        snapshots.append(
            {cid: page.table.item(_row(page, cid), 9).text() for cid in (first, second)}
        )

    runner.progressed.connect(stop_recording)
    original_builtin_open, original_io_open = builtins.open, io.open

    def recorder(original):
        def record(file, *args, **kwargs):
            if recording:
                opened.append(str(file))
            return original(file, *args, **kwargs)

        return record

    monkeypatch.setattr(builtins, "open", recorder(original_builtin_open))
    monkeypatch.setattr(io, "open", recorder(original_io_open))

    _click_checkbox(page, first)
    _click_checkbox(page, second)
    _click_menu_item(window, "候補", "評価実行")
    qtbot.waitUntil(lambda: not runner.is_busy, timeout=5000)
    qtbot.waitUntil(lambda: page.table.item(_row(page, first), 7).text() == BROKEN_TEXT)

    assert snapshots
    assert opened == []
    assert {first: "評価中 1 / 2", second: "評価待ち"} in snapshots
    assert {first: "候補", second: "評価中 2 / 2"} in snapshots
    assert page.table.item(_row(page, second), 7).text() == "0.900"
    assert page.table.item(_row(page, second), 9).text() == "候補"
    assert not page.candidate_actions["stop"].isEnabled()
    _click_checkbox(page, second)
    assert page.table.item(_row(page, first), 0).checkState() == Qt.CheckState.Checked
    assert not page.candidate_actions["release"].isEnabled()
    home.hide()
    for mode_window in manager._windows.values():
        mode_window.close()


def test_release_dialog_tells_when_previous_release_was_reused(qtbot, monkeypatch):
    ctx = make_context()
    record = ctx.backend.get_candidate_evaluation("RC-001", "val_v003")
    model = ctx.backend.list_released_models()[0]
    model.recovered_release = True
    monkeypatch.setattr(ctx.backend, "release_candidate", lambda *_args: model)
    messages = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda _parent, _title, text: messages.append(text)
    )
    dialog = ReleaseDialog(ctx, ctx.backend.get_candidate("RC-001"), record)
    qtbot.addWidget(dialog)
    dialog.show()
    QTest.mouseClick(dialog.ok_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog.model is not None, timeout=5000)
    assert messages == [
        f"前回の登録で {model.model_id} として公開済みでした。"
        "候補の状態を修復しました。コメントは前回の登録内容のままです。"
    ]
    assert dialog.result() == QDialog.DialogCode.Accepted
