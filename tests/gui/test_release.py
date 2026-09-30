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


def test_on_enter_selects_requested_model(qapp, mock_backend):
    page = make_page(mock_backend)

    page.on_enter({"select": "model_012"})

    assert page.model_table.selectionModel().selectedRows()[0].row() == 1
    assert page.detail_values["実験・途中保存モデル"].text().startswith("exp_0043")


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
    assert "検証 AP" in headers and "OOF AP" in headers
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


def test_release_list_shows_oof_only_when_it_applies(qapp, mock_backend):
    """適用できない学習時 OOF AP を、リリース一覧で通常の数値として見せない。"""
    released = mock_backend.list_released_models()
    released[0].oof_applicability = "matching"
    released[1].oof_applicability = "different"
    released[1].oof_reason = "推論設定が学習時と異なります"
    page = make_page(mock_backend)
    page.on_enter({})

    text, _tip, _row = _oof_texts(page, released[0].model_id)
    assert text != "対象外" and text != "—"
    text, tip, row = _oof_texts(page, released[1].model_id)
    assert text == "対象外" and tip == "推論設定が学習時と異なります"

    page.select_model(released[1].model_id)
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
        assert text == "対象外"
        assert tip == "学習時の評価条件を確認できません"
