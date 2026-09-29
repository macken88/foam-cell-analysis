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
