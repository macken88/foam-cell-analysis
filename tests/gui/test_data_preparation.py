"""データ準備 v2 の一括作業フローを検証する。"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QFileDialog

from foam_cell_analysis.gui.context import AppContext
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.modes.data_preparation import finalize_thumbnails
from foam_cell_analysis.gui.modes.data_preparation.dialogs import (
    ContinuousTriageDialog,
    DatasetFinalizeDialog,
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
    assert page.model.setData(page.model.index(row, 5), "可", Qt.ItemDataRole.EditRole)
    assert backend.get_working_items()[0].quality == "可"
    page.undo_stack.undo()
    assert backend.get_working_items()[0].quality == "良"
    page.close()


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


def test_finalize_thumbnail_dialog_opens_fast_and_loads_visible_range(qapp, qtbot):
    backend = MockBackend()
    base = backend.get_working_items()
    additions = [
        replace(
            base[index % len(base)],
            item_id=f"perf_item_{index:04d}",
            source_filename=f"perf_{index:04d}.tif",
            source_relpath=f"perf/perf_{index:04d}.tif",
            seed=index + 5000,
            change="added",
            usage="train" if index % 2 == 0 else "val",
        )
        for index in range(911)
    ]
    backend.add_imported_items("all", additions)
    started = perf_counter()
    dialog = DatasetFinalizeDialog(None, backend)
    qtbot.addWidget(dialog)
    dialog.show()
    qapp.processEvents()
    elapsed = perf_counter() - started
    assert elapsed < 0.5
    assert dialog.thumbnail_filter.currentText() == "追加・変更のみ"
    assert sum(model.rowCount() for model in dialog.thumbnail_models.values()) >= 900
    for purpose, model in dialog.thumbnail_models.items():
        model.filter_name = "all"
        model.set_items(dialog.thumbnail_source_items[purpose], model.errors)
    dialog.finalize_tabs.setCurrentIndex(1)
    qapp.processEvents()
    view = dialog.thumbnail_views["train"]
    model = dialog.thumbnail_models["train"]
    scroll_started = perf_counter()
    view.verticalScrollBar().setValue(view.verticalScrollBar().maximum())
    model.request_visible()
    scroll_elapsed = perf_counter() - scroll_started
    assert any(item_id.startswith("perf_item_") for item_id in model._wanted)
    assert len(model._wanted) <= 30
    assert scroll_elapsed < 0.25
    qapp.processEvents()
    qtbot.waitUntil(
        lambda: any(
            key[0].startswith("perf_item_") for key in finalize_thumbnails._thumbnail_cache
        ),
        timeout=5000,
    )
    assert model.rowCount() >= 400
    assert len(finalize_thumbnails._thumbnail_cache) <= 240
    dialog.close()
