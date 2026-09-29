"""データ準備 v2 の一括作業フローを検証する。"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QFileDialog

from foam_cell_analysis.gui.context import AppContext
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.modes.data_preparation.dialogs import (
    ContinuousTriageDialog,
    ImportSettingsDialog,
)
from foam_cell_analysis.gui.modes.data_preparation.page import DataPreparationPage
from foam_cell_analysis.gui.navigation import Navigator
from foam_cell_analysis.gui.widgets.image_convert import DisplayMode
from foam_cell_analysis.services.mock.backend import MockBackend
from foam_cell_analysis.services.models import DataItem


def _page(qapp, backend=None):
    backend = backend or MockBackend()
    return DataPreparationPage(AppContext(backend, Navigator(), JobManager()))


def _item(backend, item_id: str, folder: str, usage: str = "unassigned", classification="分類A"):
    return DataItem(
        item_id,
        f"{item_id}.tif",
        f"{folder}/{item_id}.tif",
        ["A"],
        classification,
        "良",
        ["rev_001"],
        "rev_001",
        usage=usage,
        sha256=backend._digest(item_id),
        seed=int(backend._digest(item_id)[:6], 16),
    )


def test_import_uniform_values_are_applied_per_item_without_linking_edits(qapp):
    backend = MockBackend()
    candidates = backend.scan_import_source({"A": "C:/batch-a"}, "C:/masks")[:3]
    groups = {candidates[0].source_relpath.rsplit("/", 1)[0]: candidates}
    dialog = ImportSettingsDialog(None, groups)
    dialog.rows[0][1].setCurrentIndex(dialog.rows[0][1].findData("train"))
    dialog.rows[0][2].setCurrentIndex(dialog.rows[0][2].findData("分類B"))
    dialog.rows[0][3].setCurrentIndex(dialog.rows[0][3].findData("良"))
    dialog._accept()
    values = dialog.values[next(iter(groups))]
    items = backend.import_items("all", candidates, values["classification"])
    backend.bulk_update_items("all", [item.item_id for item in items], **values)
    first, second = items[:2]
    backend.update_item("all", first.item_id, classification="分類C")
    assert first.classification == "分類C"
    assert second.classification == "分類B"
    assert all(item.usage == "train" and item.quality == "良" for item in items)


def test_table_edits_multiple_items_and_undo_restores_them(qapp):
    backend = MockBackend()
    page = _page(qapp, backend)
    selected = [item.item_id for item in backend.get_working_items()[:2]]
    page._selected_ids = selected
    page._change(selected, usage="val")
    assert all(
        next(item for item in backend.get_working_items() if item.item_id == key).usage == "val"
        for key in selected
    )
    page.undo_stack.undo()
    assert all(
        next(item for item in backend.get_working_items() if item.item_id == key).usage == "train"
        for key in selected
    )
    row = page.model.visible_items().index(backend.get_working_items()[0])
    assert page.model.setData(page.model.index(row, 4), "可", Qt.ItemDataRole.EditRole)
    assert backend.get_working_items()[0].quality == "可"
    page.undo_stack.undo()
    assert backend.get_working_items()[0].quality == "良"
    page.close()


def _visible_ids(page):
    return [item.item_id for item in page.model.visible_items()]


def _click_combo_row(qtbot, combo, row):
    combo.showPopup()
    qtbot.wait(20)
    index = combo.model().index(row, 0)
    rect = combo.view().visualRect(index)
    QTest.mouseClick(
        combo.view().viewport(),
        Qt.MouseButton.LeftButton,
        pos=rect.topLeft() + QPoint(10, rect.height() // 2),
    )
    qtbot.wait(20)


def test_filter_controls_combine_and_match_visible_rows(shell, qtbot):
    from foam_cell_analysis.gui.navigation import PageId

    shell.navigate(PageId.DATA_PREPARATION)
    page = shell.page(PageId.DATA_PREPARATION)
    window = shell.manager.window(
        __import__("foam_cell_analysis.gui.navigation", fromlist=["ModeId"]).ModeId.DATA_PREPARATION
    )
    window.show()
    window.activateWindow()
    window.raise_()
    qtbot.waitUntil(window.isActiveWindow)
    qtbot.waitUntil(lambda: page.isVisible())

    first_usage = page.items[0].usage
    if first_usage != "train":
        target_usage = next((item for item in page.items if item.usage == "train"), None)
    else:
        target_usage = page.items[0]
    assert target_usage is not None
    target_usage.change = "updated"
    target_usage.classification = "分類A"
    target_usage.source_relpath = "test-source/needle.tif"
    target_usage.source_filename = "needle.tif"
    second_target = next(
        item
        for item in page.items
        if item.item_id != target_usage.item_id and item.usage == "train"
    )
    second_target.change = "updated"
    second_target.classification = "分類B"
    second_target.source_relpath = "test-source/other.tif"
    second_target.source_filename = "other.tif"
    shell.ctx.backend.validate_items = lambda *_args, **_kwargs: type(
        "Report",
        (),
        {
            "errors": [
                type("Issue", (), {"item_id": item.item_id, "message": "test error"})()
                for item in (target_usage, second_target)
            ]
        },
    )()
    page.refresh()

    QTest.mouseClick(page.chips["train"], Qt.MouseButton.LeftButton)
    QTest.mouseClick(page.error_filter, Qt.MouseButton.LeftButton)
    QTest.mouseClick(page.changed_filter, Qt.MouseButton.LeftButton)
    expected = {
        item.item_id
        for item in page.items
        if item.usage == "train" and item.item_id in page.model.errors and item.change is not None
    }
    assert set(_visible_ids(page)) == expected
    _click_combo_row(qtbot, page.class_filter, 1)
    _click_combo_row(qtbot, page.class_filter, 2)
    assert page.model.classifications == {"分類A", "分類B"}
    assert set(_visible_ids(page)) == {target_usage.item_id, second_target.item_id}
    _click_combo_row(qtbot, page.class_filter, 0)
    assert page.model.classifications == set()
    before = (target_usage.usage, target_usage.classification, target_usage.quality)
    page.class_filter.setFocus()
    for key in (
        Qt.Key.Key_Q,
        Qt.Key.Key_W,
        Qt.Key.Key_E,
        Qt.Key.Key_R,
        Qt.Key.Key_1,
        Qt.Key.Key_Z,
        Qt.Key.Key_X,
        Qt.Key.Key_C,
    ):
        QTest.keyClick(page.class_filter, key)
    assert (target_usage.usage, target_usage.classification, target_usage.quality) == before
    page.source_combo.showPopup()
    qtbot.wait(20)
    source_index = page.source_combo.findData("test-source")
    QTest.mouseClick(
        page.source_combo.view().viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.source_combo.view()
        .visualRect(page.source_combo.model().index(source_index, 0))
        .center(),
    )
    QTest.keyClicks(page.search, "needle")
    assert _visible_ids(page) == [target_usage.item_id]
    assert page.visible_count.text() == f"{len(page.items)} 件中 1 件を表示"

    page.search.clear()
    QTest.mouseClick(page.changed_filter, Qt.MouseButton.LeftButton)
    assert not page.model.changed_only
    page.table.setFocus()
    QTest.keyClick(page.table, Qt.Key.Key_6, Qt.KeyboardModifier.AltModifier)
    assert not page.model.errors_only
    QTest.keyClick(page.table, Qt.Key.Key_6, Qt.KeyboardModifier.AltModifier)
    assert page.model.errors_only


def test_auto_triage_is_deterministic_stratified_groupwise_and_preserves_assigned_items(qapp):
    backend = MockBackend()
    master = backend.get_working_items()
    assigned = next(item for item in master if item.usage == "train")
    existing = assigned.usage
    candidates = [
        _item(
            backend,
            f"triage_{index:03d}",
            f"lot-{index // 2}",
            classification=("分類A" if index % 2 == 0 else "分類B"),
        )
        for index in range(12)
    ]
    backend.add_imported_items("all", candidates)
    result = backend.apply_auto_triage(
        {"validation_ratio": 50, "by_folder": True, "stratify": True, "seed": 42},
        [item.item_id for item in candidates],
    )
    assert result["train"] + result["val"] == 12
    assert assigned.usage == existing
    assert all(
        len({item.usage for item in candidates if item.source_folder == folder}) == 1
        for folder in {item.source_folder for item in candidates}
    )
    after = [item.usage for item in candidates]
    for item in candidates:
        item.usage = "unassigned"
    backend.apply_auto_triage(
        {"validation_ratio": 50, "by_folder": True, "stratify": True, "seed": 42},
        [item.item_id for item in candidates],
    )
    assert [item.usage for item in candidates] == after


def test_finalization_publishes_only_changed_train_and_val_and_omits_unassigned(qapp):
    backend = MockBackend()
    master = backend.get_working_items()
    for item in master:
        if item.usage in {"train", "val"}:
            item.classification = item.classification or "分類A"
            item.quality = item.quality or "良"
            if not item.mask_revisions:
                item.mask_revisions = ["rev_001"]
                item.selected_mask_revision = "rev_001"
    candidates = backend.scan_import_source({"A": "C:/new-batch"}, "C:/masks")[:2]
    imported = backend.import_items("all", candidates, "分類A")
    extra_unassigned = _item(backend, "extra_unassigned", "lot-unassigned")
    backend.add_imported_items("all", [extra_unassigned])
    backend.bulk_update_items("all", [imported[0].item_id], usage="train", quality="良")
    backend.bulk_update_items("all", [imported[1].item_id], usage="val", quality="良")
    unassigned = next(item for item in master if item.usage == "unassigned")
    versions = backend.finalize_working_dataset("同時確定")
    assert {item.purpose for item in versions} == {"train", "val"}
    assert all(unassigned.item_id not in version.item_ids for version in versions)
    train = next(item for item in versions if item.purpose == "train")
    val = next(item for item in versions if item.purpose == "val")
    assert train.base_validation_version == val.version
    assert train.version == "train_v004" and val.version == "val_v004"


def test_excel_export_and_import_round_trip(qapp, tmp_path, monkeypatch):
    pytest = __import__("pytest")
    pytest.importorskip("openpyxl")
    backend = MockBackend()
    page = _page(qapp, backend)
    output = str(tmp_path / "working.xlsx")
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (output, "Excel (*.xlsx)")
    )
    page.export_excel()
    before = [
        (item.item_id, item.usage, item.classification, item.quality)
        for item in backend.get_working_items()
    ]
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: (output, "Excel (*.xlsx)")
    )
    page.import_excel()
    after = [
        (item.item_id, item.usage, item.classification, item.quality)
        for item in backend.get_working_items()
    ]
    assert after == before
    page.undo_stack.undo()
    assert [
        (item.item_id, item.usage, item.classification, item.quality)
        for item in backend.get_working_items()
    ] == before
    page.close()


def test_working_table_uses_no_thumbnail_toggle_and_updates_only_edited_items(qapp):
    backend = MockBackend()
    page = _page(qapp, backend)
    assert not hasattr(page, "thumbnail")
    assert "toggle_view" not in page.shortcuts.mapping
    calls = []
    validate = backend.validate_items

    def record(item_ids=None):
        calls.append(item_ids)
        return validate(item_ids)

    backend.validate_items = record
    resets = QSignalSpy(page.model.modelReset)
    layouts = QSignalSpy(page.model.layoutChanged)
    item = next(item for item in backend.get_working_items() if item.usage == "unassigned")
    page._change([item.item_id], quality="可")
    assert calls == [None]
    assert resets.count() == 0
    assert layouts.count() == 0
    page.close()


def test_preview_shortcuts_work_from_table_but_not_search_input(qapp):
    backend = MockBackend()
    page = _page(qapp, backend)
    messages = []
    page.ctx.status.message.connect(messages.append)
    page.table.setFocus()
    QTest.keyClick(page.table, Qt.Key.Key_M)
    assert not page.display_toggle.is_alternate
    QTest.keyClick(page.table, Qt.Key.Key_M)
    assert page.display_toggle.is_alternate
    page.search.setFocus()
    QTest.keyClick(page.search, Qt.Key.Key_M)
    assert page.search.text().casefold() == "m"
    assert page.display_toggle.is_alternate
    page.close()


def test_continuous_triage_shortcuts_toggle_mode_and_move_image(qapp, qtbot):
    backend = MockBackend()
    page = _page(qapp, backend)
    items = [item for item in backend.get_working_items() if item.usage == "unassigned"][:2]
    dialog = ContinuousTriageDialog(
        page, items, lambda *_args, **_kwargs: None, backend, page.shortcuts
    )
    qtbot.addWidget(dialog)
    messages = []
    page.ctx.status.message.connect(messages.append)
    dialog.show()
    dialog.setFocus()
    QTest.keyClick(dialog, Qt.Key.Key_M)
    assert dialog.display_mode == DisplayMode.IMAGE
    QTest.keyClick(dialog, Qt.Key.Key_M)
    assert dialog.display_mode == DisplayMode.OVERLAY
    QTest.keyClick(dialog, Qt.Key.Key_Right)
    assert dialog.index == 1
    dialog.close()
    page.close()
