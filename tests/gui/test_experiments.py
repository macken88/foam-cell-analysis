"""実験一覧の選択条件とモデル比較への遷移テスト。"""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

from foam_cell_analysis.gui.modes.training.dialogs import (
    ExperimentCompareDialog,
    SendToCandidatesDialog,
)
from foam_cell_analysis.gui.navigation import PageId
from foam_cell_analysis.gui.theme import SERIES


def test_send_completed_experiment_transitions_with_candidate_parameters(shell, monkeypatch):
    page = shell.page(PageId.EXPERIMENTS)
    target_row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0042"
    )
    page.table.setCurrentCell(target_row, 1)
    received = []
    monkeypatch.setattr(
        shell.ctx.navigator,
        "navigate",
        lambda page_id, **params: received.append((page_id, params)),
    )
    monkeypatch.setattr(SendToCandidatesDialog, "exec", lambda _self: QDialog.DialogCode.Accepted)

    page.send_selected()

    assert received[-1][0] == PageId.CANDIDATES
    assert received[-1][1]["action"] == "add_candidate"
    assert received[-1][1]["experiment_id"] == "exp_0042"
    assert received[-1][1]["checkpoint"] == "final.pt"


def test_compare_dialog_has_one_value_column_per_experiment(mock_backend):
    experiments = [
        mock_backend.get_experiment("exp_0042"),
        mock_backend.get_experiment("exp_0043"),
    ]

    dialog = ExperimentCompareDialog(experiments)

    assert dialog.table.columnCount() == 3
    assert dialog.table.horizontalHeaderItem(1).text() == "exp_0042"
    assert dialog.table.horizontalHeaderItem(2).text() == "exp_0043"
    row = next(
        index
        for index in range(dialog.table.rowCount())
        if dialog.table.item(index, 0).toolTip() == "data.used_item_ids"
    )
    assert dialog.table.item(row, 1).text() == "59 件"
    assert dialog.table.item(row, 2).text() == "59 件"


def test_learning_curves_use_separate_map_and_loss_charts(shell):
    page = shell.page(PageId.EXPERIMENTS)
    page.resize(1400, 900)
    page.show()
    row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0042"
    )
    cell = page.table.visualItemRect(page.table.item(row, 1))
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=cell.topLeft() + QPoint(15, cell.height() // 2),
    )

    assert [series[0] for series in page.chart_map.series][0] == "OOF mAP"
    assert len(page.chart_map.series) == 6
    oof_values = page.chart_map.series[0][3]
    assert any(series[3][-1] != oof_values[-1] for series in page.chart_map.series[1:])
    assert page.chart_loss.series[0][0] == "交差検証 loss"
    assert page.chart_loss.series[1][0] == "最終学習 loss"
    assert page.chart_map.best is not None
    assert [page.details.tabText(i) for i in range(page.details.count())] == [
        "概要",
        "学習曲線",
        "交差検証",
        "途中保存モデル",
        "実行試行",
        "実使用データ",
    ]
    assert page.details.currentIndex() == 0
    assert page.cv_table.item(0, 3).text() != page.cv_table.item(1, 3).text()
    folds = {
        page.checkpoint_table.item(i, 0).text()
        for i in range(page.checkpoint_table.rowCount())
        if page.checkpoint_table.item(i, 0).text() != "最終"
    }
    assert folds == {"1", "2", "3", "4", "5"}
    assert all(
        not page.checkpoint_table.item(i, 1).text().startswith("fold_")
        for i in range(page.checkpoint_table.rowCount())
    )
    assert (
        sum(
            page.checkpoint_table.item(i, 0).text() != "最終"
            for i in range(page.checkpoint_table.rowCount())
        )
        == 55
    )
    assert any(
        page.checkpoint_table.item(i, 1).text() == "final.pt"
        for i in range(page.checkpoint_table.rowCount())
    )


def test_overview_has_empty_hint_and_two_column_selected_details(shell):
    page = shell.page(PageId.EXPERIMENTS)
    assert page.overview_placeholder.text() == "実験を選ぶと詳細が表示されます"
    row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0042"
    )

    page.table.setCurrentCell(row, 1)

    assert page.overview_table.columnCount() == 2
    assert page.overview_table.rowCount() > 0
    assert page.overview_table.item(0, 0).text() == "実験群"


def test_overview_translates_all_cv_config_keys_and_model_details_distinguish_sections(shell):
    """概要の交差検証キーを日本語にし、Mask R-CNN の詳細区分を区別する。"""
    from foam_cell_analysis.gui.widgets.form import CollapsibleSection

    page = shell.page(PageId.EXPERIMENTS)
    row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0042"
    )
    rect = page.table.visualItemRect(page.table.item(row, 1))
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    experiment = shell.ctx.backend.get_experiment("exp_0042")
    experiment.oof_evaluation = shell.ctx.backend._oof_evaluation(experiment, 0.91)
    page._current_changed()
    oof_headers = [
        page.oof_table.horizontalHeaderItem(column).text()
        for column in range(page.oof_table.columnCount())
    ]
    assert "未分類" in oof_headers
    assert int(page.oof_table.item(1, 1).text()) == len(experiment.used_item_ids)
    labels = {
        page.overview_table.item(row, 0).text() for row in range(page.overview_table.rowCount())
    }
    assert "画像分類で層別" in labels
    assert "取り込み元フォルダ単位で分割" in labels
    assert all("_" not in label and "." not in label for label in labels)
    assert all(
        page.overview_table.item(row, 0).text()
        not in {"stratify_by_classification", "group_by_source_folder"}
        for row in range(page.overview_table.rowCount())
    )

    training = shell.page(PageId.TRAINING)
    sections = {
        section.button.text()
        for section in training.findChildren(CollapsibleSection)
        if "詳細設定" in section.button.text()
    }
    assert "RPN 詳細設定 ▶" in sections
    assert "ROI 詳細設定 ▶" in sections
    assert "詳細設定 ▶" not in sections


def test_seeded_selected_checkpoints_follow_training_date_order(shell):
    """選択エポックのチェックポイントを CV 保存期間の時間軸に置く。"""
    experiment = shell.ctx.backend.get_experiment("exp_0042")
    selected = [item for item in experiment.checkpoints if item.name == "selected.pt"]
    final = next(item for item in experiment.checkpoints if item.name == "final.pt")
    fold_saves = [item for item in experiment.checkpoints if item.name.startswith("epoch_")]
    assert selected and fold_saves
    # 選択エポックは全フォールドの交差検証が終わってから決まるので、
    # CV の保存より後・最終学習より前に並ぶ（日付の境目には左右されない比較にする）
    latest_cv_save = max(item.saved_at for item in fold_saves)
    assert all(latest_cv_save <= item.saved_at for item in selected)
    assert max(item.saved_at for item in selected) < final.saved_at


def test_experiment_tables_fit_contents_and_keep_manually_resized_columns(shell, qapp):
    """実験クリックで各表を内容幅にし、手動幅を選択・更新後も保つ。"""
    page = shell.page(PageId.EXPERIMENTS)
    page.resize(1500, 1000)
    page.show()
    row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0042"
    )
    cell = page.table.visualItemRect(page.table.item(row, 1))
    QTest.mouseClick(
        page.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=cell.center(),
    )
    qapp.processEvents()

    for table in (
        page.table,
        page.overview_table,
        page.cv_table,
        page.oof_table,
        page.checkpoint_table,
        page.run_table,
    ):
        for column in range(table.columnCount()):
            header_width = table.horizontalHeader().sectionSizeHint(column)
            content_width = max((table.sizeHintForColumn(column) for _ in [0]), default=0)
            assert table.columnWidth(column) >= min(max(header_width, content_width), 360)

    header = page.overview_table.horizontalHeader()
    before = page.overview_table.columnWidth(0)
    edge = header.sectionViewportPosition(0) + before - 1
    QTest.mousePress(
        header.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(edge, header.height() // 2),
    )
    QTest.mouseMove(header.viewport(), QPoint(edge + 100, header.height() // 2), delay=20)
    QTest.mouseRelease(
        header.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(edge + 100, header.height() // 2),
    )
    qapp.processEvents()
    manual_width = page.overview_table.columnWidth(0)
    assert manual_width >= before + 80

    other_row = next(
        row for row in range(page.table.rowCount()) if page.table.item(row, 1).text() == "exp_0043"
    )
    rect = page.table.visualItemRect(page.table.item(other_row, 1))
    QTest.mouseClick(page.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    page.refresh()
    qapp.processEvents()
    assert page.overview_table.columnWidth(0) == manual_width
    page.refresh_on_activate()
    qapp.processEvents()
    assert page.overview_table.columnWidth(0) == manual_width


def test_comparison_series_colors_keep_theme_order_after_refresh(mock_backend):
    experiments = [mock_backend.get_experiment(key) for key in ("exp_0042", "exp_0043", "exp_0044")]
    dialog = ExperimentCompareDialog(experiments)
    expected = [QColor(color) for color in SERIES]

    dialog.differences_only.setChecked(True)

    assert [series[1] for series in dialog.chart.series] == expected[:3]
