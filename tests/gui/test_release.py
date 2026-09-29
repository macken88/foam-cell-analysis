"""リリース済みモデル画面の振り分け動作を検証する。"""

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
    records = page.apply_pending_changes()

    assert mock_backend.get_routing()["分類A"] == "model_012"
    assert len(records) == 1
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
