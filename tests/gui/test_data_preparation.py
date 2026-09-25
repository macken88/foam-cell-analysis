"""データ準備画面の操作テスト。"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from foam_cell_analysis.gui.context import AppContext, StatusBus
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.modes.data_preparation.dialogs import (
    ArchiveDialog,
    DatasetFinalizeDialog,
    ExcelExportDialog,
    ExcelImportDialog,
    ImportDialog,
    MaskRevisionDialog,
)
from foam_cell_analysis.gui.modes.data_preparation.page import DataPreparationPage
from foam_cell_analysis.gui.navigation import Navigator
from foam_cell_analysis.services.mock.backend import MockBackend


@pytest.fixture
def mock_backend():
    """新しいモック Backend を用意する。"""
    return MockBackend()


@pytest.fixture
def data_page(qapp, mock_backend):
    """データ準備ページだけを生成する。"""
    context = AppContext(mock_backend, Navigator(), JobManager(), StatusBus())
    page = DataPreparationPage(context)
    yield page
    page.close()


def test_data_preparation_validation_and_finalize_flow(data_page, mock_backend, qtbot) -> None:
    page = data_page
    dataset = mock_backend.get_working_dataset("train")
    for item in dataset.items:
        if item.included and (not item.classification or not item.quality):
            mock_backend.update_item(
                "train",
                item.item_id,
                classification=item.classification or "分類A",
                quality=item.quality or "良",
            )
    page.refresh()
    page.start_validation()
    qtbot.waitUntil(lambda: mock_backend.get_working_dataset("train").state == "VALIDATED")
    dialog = DatasetFinalizeDialog(page, mock_backend, "train")
    version = dialog.apply()
    page.refresh()
    assert version.version == "train_v004"
    assert mock_backend.get_working_dataset("train").state == "WORKING"
    assert any(
        item.version == version.version for item in mock_backend.list_dataset_versions("train")
    )


def test_exclude_and_restore_selection(data_page, mock_backend) -> None:
    page = data_page
    item = mock_backend.get_working_dataset("train").items[0]
    index = page.proxy.mapFromSource(page.item_model.index(0, 0))
    page.table.selectRow(index.row())
    page._bulk_update(included=False)
    assert not next(
        row
        for row in mock_backend.get_working_dataset("train").items
        if row.item_id == item.item_id
    ).included
    page.table.selectRow(0)
    page._bulk_update(included=True)
    assert next(
        row
        for row in mock_backend.get_working_dataset("train").items
        if row.item_id == item.item_id
    ).included


def test_initial_row_is_selected_and_preview_is_loaded(data_page, mock_backend) -> None:
    first = mock_backend.get_working_dataset("train").items[0]
    assert data_page.selected_item_id == first.item_id
    assert data_page.image_view._pixmap_item is not None
    assert data_page.changed_filter.isCheckable()
    assert data_page.errors_filter.isCheckable()
    assert data_page.changed_filter.text() == "変更ありのみ"
    assert data_page.errors_filter.text() == "エラーありのみ"
    assert data_page.display_combo.findText("インスタンスラベル") >= 0
    assert data_page.mask_combo.count() == len(first.mask_revisions)
    assert data_page.table.verticalHeader().isHidden()
    assert data_page.history_table.verticalHeader().isHidden()


def test_excel_import_error_does_not_apply_changes(data_page, mock_backend) -> None:
    page = data_page
    before = mock_backend.get_working_dataset("train").items[0].quality
    dialog = ExcelImportDialog(page, mock_backend, "train")
    dialog.file_row.path_edit.setText("C:/tmp/error.xlsx")
    dialog.preview()
    assert "エラー" in dialog.result_label.text()
    assert not dialog.apply_button.isEnabled()
    assert dialog.approved_changes == []
    assert not dialog._changes
    assert mock_backend.get_working_dataset("train").items[0].quality == before


def test_finalize_cancel_rejects_and_archive_accepts_path(data_page) -> None:
    finalize = DatasetFinalizeDialog(data_page, data_page.ctx.backend, "train")
    assert finalize.windowTitle() == "新しいデータセットを作成"
    assert finalize.minimumWidth() >= 480
    buttons = finalize.findChildren(QPushButton)
    cancel = next(button for button in buttons if button.text() == "キャンセル")
    cancel.click()
    assert finalize.result() == finalize.DialogCode.Rejected

    archive = ArchiveDialog(data_page, "train_v003")
    assert archive.windowTitle() == "アーカイブ作成"
    assert archive.minimumWidth() >= 560
    archive.destination.path_edit.setText("C:/archive")
    archive_buttons = archive.findChildren(QPushButton)
    next(button for button in archive_buttons if button.text() == "作成").click()
    assert archive.result() == archive.DialogCode.Accepted
    assert archive.output_path == "C:/archive"


def test_finalize_duplicate_check_disables_creation_and_lists_conflicts(data_page) -> None:
    dialog = DatasetFinalizeDialog(data_page, data_page.ctx.backend, "train")
    dialog.backend.check_dataset_duplicates = lambda *_args: [
        ("item_000007", "ハッシュ"),
        ("item_000009", "識別子"),
    ]

    duplicates = dialog.check_duplicates()

    assert len(duplicates) == 2
    assert "item_000007" in dialog.duplicate_label.text()
    assert "item_000009" in dialog.duplicate_label.text()
    assert dialog.duplicate_label.property("state") == "error"
    assert not dialog.create_button.isEnabled()


def test_operation_dialogs_have_titles_sizes_labels_and_primary_actions(data_page) -> None:
    backend = data_page.ctx.backend
    dialogs = [
        (ImportDialog(data_page), "データ取り込み", "選択分を取り込む"),
        (ExcelExportDialog(data_page), "Excel出力", "出力"),
        (ExcelImportDialog(data_page, backend, "train"), "Excel取込", "変更を取り込む"),
        (MaskRevisionDialog(data_page, "sample.tif"), "新しいマスク版を取り込む", "取り込む"),
    ]
    for dialog, title, action in dialogs:
        assert dialog.windowTitle() == title
        assert dialog.minimumWidth() > 0
        button = next(
            button for button in dialog.findChildren(QPushButton) if button.text() == action
        )
        assert button.property("primary") is True


def test_editing_validated_item_returns_to_working(data_page, mock_backend) -> None:
    page = data_page
    dataset = mock_backend.get_working_dataset("train")
    for item in dataset.items:
        if item.included and (not item.classification or not item.quality):
            mock_backend.update_item(
                "train",
                item.item_id,
                classification=item.classification or "分類A",
                quality=item.quality or "良",
            )
    mock_backend.validate_working_dataset("train")
    assert dataset.state == "VALIDATED"
    page.refresh()
    row = page.item_model.row_for_id(dataset.items[0].item_id)
    old_quality = dataset.items[0].quality
    new_quality = "可" if old_quality != "可" else "良"
    assert page.item_model.setData(
        page.item_model.index(row, 4), new_quality, Qt.ItemDataRole.EditRole
    )
    assert dataset.state == "WORKING"
    assert dataset.items[0].quality == new_quality
