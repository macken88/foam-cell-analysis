"""リリース済みモデル画面の振り分け動作を検証する。"""

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QMessageBox

from foam_cell_analysis.gui.context import AppContext, StatusBus
from foam_cell_analysis.gui.jobs import JobManager
from foam_cell_analysis.gui.modes.release.page import ReleasedModelsPage
from foam_cell_analysis.gui.navigation import Navigator


def make_page(mock_backend):
    """画面を初期化して返す。"""
    context = AppContext(mock_backend, Navigator(), JobManager(), StatusBus())
    return ReleasedModelsPage(context)


def test_routing_change_apply_updates_backend_and_history(qapp, mock_backend):
    page = make_page(mock_backend)
    page.on_enter({})
    page._routing_controls["分類A"].setCurrentIndex(
        page._routing_controls["分類A"].findData("model_012")
    )

    assert page.build_change_rows() == [("分類A", "model_007", "model_012")]
    applied = page.apply_pending_changes()

    assert mock_backend.get_routing()["分類A"] == "model_012"
    assert applied == 1
    assert len(mock_backend.list_routing_history()) == 1
    assert page.history_table.rowCount() == 1


def test_discard_routing_change_restores_original_value(qapp, mock_backend):
    page = make_page(mock_backend)
    page.on_enter({})
    control = page._routing_controls["分類A"]
    control.setCurrentIndex(control.findData("model_012"))

    page.discard_changes()

    assert page._routing_controls["分類A"].currentData() == "model_007"
    assert page.build_change_rows() == []
    assert mock_backend.get_routing()["分類A"] == "model_007"


def test_pending_routing_target_immediately_blocks_release_lifecycle_actions(qtbot, mock_backend):
    state = mock_backend.get_routing_state()
    mock_backend.apply_routing(
        {classification: None for classification in state.assignments},
        expected_revision=state.revision,
    )
    page = make_page(mock_backend)
    qtbot.addWidget(page)
    page.show()
    page.on_enter({"select": "model_012"})
    control = page._routing_controls["分類A"]
    control.setFocus()
    QTest.keyClick(control, Qt.Key.Key_End)
    QTest.keyClick(control, Qt.Key.Key_Enter)
    assert control.currentData() == "model_012"
    assert not page.archive_button.isEnabled()
    assert "未適用" in page.archive_button.toolTip()
    assert not page.delete_button.isEnabled()
    assert "未適用" in page.delete_button.toolTip()

    QTest.keyClick(control, Qt.Key.Key_Home)
    QTest.keyClick(control, Qt.Key.Key_Enter)
    assert control.currentData() is None
    assert page.archive_button.isEnabled()
    assert page.delete_button.isEnabled()


def test_on_enter_selects_requested_model(qapp, mock_backend):
    page = make_page(mock_backend)

    page.on_enter({"select": "model_012"})

    assert page.model_table.selectionModel().selectedRows()[0].row() == 1
    assert page.detail_values["実験・途中保存モデル"].text().startswith("exp_0043")


def test_archive_and_delete_release_from_history_with_capacity_confirmation(
    qapp, qtbot, mock_backend, monkeypatch
):
    page = make_page(mock_backend)
    qtbot.addWidget(page)
    page.show()
    page.on_enter({"select": "model_007"})
    qtbot.wait(20)
    assert page.model_table.rowCount() == 2
    assert not page.archive_button.isEnabled()
    assert "振り分けを解除" in page.archive_button.toolTip()

    mock_backend.apply_routing(
        {classification: None for classification in mock_backend.get_routing_state().assignments},
        expected_revision=mock_backend.get_routing_state().revision,
    )
    page._read_routing_state()
    page._load_routing()
    page._update_lifecycle_actions()
    assert page._selected_release().model_id == "model_007"
    assert page.archive_button.isEnabled()
    assert not page.show_history.isChecked()
    page._routing_controls["分類B"].setCurrentIndex(
        page._routing_controls["分類B"].findData("model_012")
    )
    model_012_row = next(
        row
        for row in range(page.model_table.rowCount())
        if page.model_table.item(row, 0).text() == "model_012"
    )
    QTest.mouseClick(
        page.model_table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.model_table.visualItemRect(page.model_table.item(model_012_row, 0)).center(),
    )
    assert not page.archive_button.isEnabled()
    assert "未適用" in page.archive_button.toolTip()
    model_007_row = next(
        row
        for row in range(page.model_table.rowCount())
        if page.model_table.item(row, 0).text() == "model_007"
    )
    QTest.mouseClick(
        page.model_table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.model_table.visualItemRect(page.model_table.item(model_007_row, 0)).center(),
    )
    assert page.archive_button.isEnabled()
    QTest.mouseClick(page.archive_button, Qt.MouseButton.LeftButton)
    assert page.model_table.rowCount() == 1
    assert mock_backend.released["model_007"].lifecycle_status == "archived"
    assert page._routing_controls["分類B"].currentData() == "model_012"
    assert page.build_change_rows() == [("分類B", None, "model_012")]

    QTest.mouseClick(page.show_history, Qt.MouseButton.LeftButton)
    assert page.model_table.rowCount() == 2
    row = next(
        row
        for row in range(page.model_table.rowCount())
        if page.model_table.item(row, 0).text() == "model_007"
    )
    QTest.mouseClick(
        page.model_table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.model_table.visualItemRect(page.model_table.item(row, 0)).center(),
    )
    assert page._selected_release().model_id == "model_007"
    assert page.model_table.item(0, 11).text() == "採用（保管）"
    QTest.mouseClick(page.archive_button, Qt.MouseButton.LeftButton)
    assert mock_backend.released["model_007"].lifecycle_status == "active"
    assert page._routing_controls["分類A"].findData("model_007") >= 0
    assert page.model_table.rowCount() == 2
    assert page.model_table.item(page.model_table.currentRow(), 11).text() == "採用"
    observed = []

    def confirm(_parent, _title, message, *_args):
        observed.append(message)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)
    QTest.mouseClick(page.delete_button, Qt.MouseButton.LeftButton)
    assert "空く容量: 約 0 B" in observed[0]
    assert mock_backend.released["model_007"].lifecycle_status == "deleted"
    deleted_row = next(
        row
        for row in range(page.model_table.rowCount())
        if page.model_table.item(row, 0).text() == "model_007"
    )
    assert page.model_table.item(deleted_row, 11).text() == "採用（削除済み）"


def test_detailed_model_settings_use_japanese_labels_instead_of_dict_repr(qapp, mock_backend):
    page = make_page(mock_backend)

    rows = page._flatten_detail("前処理設定", {"input": {"min_size": 800, "max_size": 1333}})

    assert rows == [
        ("前処理設定 / 入力画像の短辺サイズ", "800"),
        ("前処理設定 / 入力画像の最大辺サイズ", "1333"),
    ]
    assert "{'min_size'" not in str(rows)


def _change_routing(page, classification, model_id):
    control = page._routing_controls[classification]
    control.setCurrentIndex(control.findData(model_id))


def test_apply_button_passes_loaded_revision(qtbot, mock_backend, monkeypatch):
    page = make_page(mock_backend)
    qtbot.addWidget(page)
    page.on_enter({})
    revision = mock_backend.get_routing_state().revision
    calls = []
    original = mock_backend.apply_routing

    def spy(changes, *, expected_revision):
        calls.append(expected_revision)
        return original(changes, expected_revision=expected_revision)

    monkeypatch.setattr(mock_backend, "apply_routing", spy)
    monkeypatch.setattr(
        "foam_cell_analysis.gui.modes.release.page.RoutingChangesDialog.exec",
        lambda dialog: QDialog.DialogCode.Accepted,
    )
    _change_routing(page, "分類A", "model_012")
    page.show()
    QTest.mouseClick(page.apply_button, Qt.MouseButton.LeftButton)

    assert calls == [revision]
    assert mock_backend.get_routing()["分類A"] == "model_012"
    assert page._revision == mock_backend.get_routing_state().revision
    assert page.apply_button.isEnabled() is False


def test_routing_conflict_shows_message_and_reloads(qtbot, mock_backend, monkeypatch):
    page = make_page(mock_backend)
    qtbot.addWidget(page)
    page.on_enter({})
    # 画面を開いた後に、別の操作で振り分けが変わる
    other = mock_backend.get_routing_state()
    mock_backend.apply_routing({"分類B": "model_007"}, expected_revision=other.revision)
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda _parent, _title, text, *args: warnings.append(text)
    )
    monkeypatch.setattr(
        "foam_cell_analysis.gui.modes.release.page.RoutingChangesDialog.exec",
        lambda dialog: QDialog.DialogCode.Accepted,
    )
    _change_routing(page, "分類A", "model_012")
    page.show()
    QTest.mouseClick(page.apply_button, Qt.MouseButton.LeftButton)

    assert warnings == ["振り分けが別の操作で変更されました。画面を開き直してください"]
    assert mock_backend.get_routing()["分類A"] == "model_007"
    assert page._routing_controls["分類B"].currentData() == "model_007"
    assert page.build_change_rows() == []


def test_release_page_does_not_read_experiments(qapp, mock_backend, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("リリース画面は実験を引かない")

    monkeypatch.setattr(mock_backend, "get_experiment", forbidden)
    page = make_page(mock_backend)
    page.on_enter({"select": "model_007"})

    headers = [
        page.model_table.horizontalHeaderItem(column).text()
        for column in range(page.model_table.columnCount())
    ]
    assert "検証 AP" in headers and "学習時 OOF AP" in headers
    assert not any("mAP" in header for header in headers)
    assert page.model_table.item(0, 1).text() == "Mask R-CNN"
    assert page.detail_values["推論設定"].text() != "—"


def test_inference_summary_uses_specific_names_and_hides_internal_keys(qapp, mock_backend):
    page = make_page(mock_backend)

    rows = page._inference_rows(
        {
            "channel_axis": 2,
            "normalize": False,
            "bsize": 256,
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "min_size": 15,
            "max_size_fraction": 0.4,
        }
    )
    labels = [label for label, _value in rows]

    assert labels == [
        "セル確率閾値",
        "フロー閾値",
        "最小サイズ（画素）",
        "最大サイズの割合",
    ]
    summary = page._inference_summary({"min_size": 15, "bsize": 256, "unknown_key": 1})
    assert summary == "最小サイズ（画素） 15"


def _oof_texts(page, model_id):
    row = next(
        row
        for row in range(page.model_table.rowCount())
        if page.model_table.item(row, 0).text() == model_id
    )
    return page.model_table.item(row, 7).text(), page.model_table.item(row, 7).toolTip(), row


def test_release_list_shows_saved_oof_as_reference_for_any_condition(qapp, mock_backend):
    """学習時 OOF AP は比較設定と関係なく参考値として保持する。"""
    released = mock_backend.list_released_models()
    released[0].oof_applicability = "matching"
    released[1].oof_applicability = "different"
    released[1].oof_reason = "推論設定が学習時と異なります"
    page = make_page(mock_backend)
    page.on_enter({})

    text, _tip, _row = _oof_texts(page, released[0].model_id)
    assert text != "対象外" and text != "—"
    text, tip, row = _oof_texts(page, released[1].model_id)
    assert text != "対象外" and text != "—"
    assert tip == "推論設定が学習時と異なります"

    QTest.mouseClick(
        page.model_table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.model_table.visualItemRect(page.model_table.item(row, 0)).center(),
    )
    detail = page.detail_values["学習時 OOF AP（参考）"].text()
    assert "推論設定が学習時と異なります" in detail
    assert "final.pt 自身の評価ではありません" in detail


def test_release_list_treats_unknown_and_missing_applicability_as_not_applicable(
    qapp, mock_backend
):
    released = mock_backend.list_released_models()
    released[0].oof_applicability = "unknown"
    released[0].oof_reason = "学習時の評価条件を確認できません"
    released[1].oof_applicability = ""
    released[1].oof_reason = ""
    page = make_page(mock_backend)
    page.on_enter({})

    for model in released[:2]:
        text, tip, _row = _oof_texts(page, model.model_id)
        assert text != "対象外" and text != "—"
        assert tip == "学習時の評価条件を確認できません"


def test_lifecycle_file_errors_reload_and_keep_actions_usable(
    qapp, qtbot, mock_backend, monkeypatch
):
    page = make_page(mock_backend)
    qtbot.addWidget(page)
    page.show()
    page.on_enter({"select": "model_012"})
    state = mock_backend.get_routing_state()
    mock_backend.apply_routing(
        {classification: None for classification in state.assignments},
        expected_revision=state.revision,
    )
    page._read_routing_state()
    page._load_routing()
    row = next(
        row
        for row in range(page.model_table.rowCount())
        if page.model_table.item(row, 0).text() == "model_012"
    )
    QTest.mouseClick(
        page.model_table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=page.model_table.visualItemRect(page.model_table.item(row, 0)).center(),
    )
    page._update_lifecycle_actions()
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda _parent, _title, message, *args: warnings.append(message)
    )
    monkeypatch.setattr(
        mock_backend,
        "set_release_archived",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("振り分けに使用中のモデルは保管できません")
        ),
    )
    assert page.archive_button.isEnabled(), (page._selected_release(), page._assignments)
    QTest.mouseClick(page.archive_button, Qt.MouseButton.LeftButton)
    assert warnings[-1] == "振り分けに使用中のモデルは保管できません"

    monkeypatch.setattr(
        mock_backend,
        "set_release_archived",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("private path")),
    )
    assert page.archive_button.isEnabled(), (page._selected_release(), page._assignments)
    QTest.mouseClick(page.archive_button, Qt.MouseButton.LeftButton)
    assert warnings[-1] == "保管状態を保存できませんでした。状態を再読み込みしました。"
    assert mock_backend.released["model_012"].lifecycle_status == "active"
    assert page.archive_button.isEnabled()
    assert not page.show_history.isChecked()

    monkeypatch.setattr(
        mock_backend,
        "delete_released_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private path")),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    QTest.mouseClick(page.delete_button, Qt.MouseButton.LeftButton)
    assert warnings[-1] == "公開モデルのファイルを削除できませんでした"
    assert mock_backend.released["model_012"].lifecycle_status == "active"
    assert page.delete_button.isEnabled()
    assert not page.show_history.isChecked()

    monkeypatch.setattr(
        mock_backend,
        "delete_released_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("リリース保存先に reparse point があるため操作できません")
        ),
    )
    QTest.mouseClick(page.delete_button, Qt.MouseButton.LeftButton)
    assert warnings[-1] == "リリース保存先に通常のフォルダではない項目があるため操作できません"
    assert "reparse point" not in warnings[-1]
    assert not page.show_history.isChecked()
