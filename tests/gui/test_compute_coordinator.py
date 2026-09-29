"""計算処理の排他制御（比較・推論設計 15.1）。"""

import pytest

from foam_cell_analysis.gui.compute_coordinator import ComputeCoordinator


@pytest.fixture
def coordinator(qapp):
    return ComputeCoordinator()


def test_requests_start_in_fifo_order(coordinator):
    started = []
    first = coordinator.request("training", "学習 exp_001", lambda: started.append("a"))
    second = coordinator.request("evaluation", "評価 cand_001", lambda: started.append("b"))
    third = coordinator.request("training", "学習 exp_002", lambda: started.append("c"))

    assert started == ["a"]
    assert coordinator.is_busy
    assert coordinator.active_label == "学習 exp_001"
    assert coordinator.waiting == [second, third]
    assert coordinator.wait_message("training") is None

    coordinator.release(first)
    assert started == ["a", "b"]
    assert coordinator.wait_message("training") == "評価の終了を待っています"
    coordinator.release(second)
    assert started == ["a", "b", "c"]
    coordinator.release(third)
    assert not coordinator.is_busy
    assert coordinator.active_label is None


def test_cancel_removes_only_the_waiting_request(coordinator):
    started = []
    first = coordinator.request("evaluation", "評価 cand_001", lambda: started.append("a"))
    second = coordinator.request("training", "学習 exp_001", lambda: started.append("b"))
    third = coordinator.request("evaluation", "評価 cand_002", lambda: started.append("c"))

    coordinator.cancel(second)
    coordinator.cancel(first)  # 実行中には割り込まない
    assert coordinator.active is first
    assert coordinator.waiting == [third]

    coordinator.release(first)
    assert started == ["a", "c"]


def test_failed_terminal_save_blocks_next_request_until_unblock(coordinator):
    started = []
    changes = []
    coordinator.changed.connect(lambda: changes.append(coordinator.is_busy))
    first = coordinator.request("training", "学習 exp_001", lambda: started.append("a"))
    coordinator.request("evaluation", "評価 cand_001", lambda: started.append("b"))

    coordinator.release(first, ok=False, error="status.json の保存に失敗")
    assert started == ["a"]
    assert coordinator.is_blocked and coordinator.is_busy
    assert "status.json の保存に失敗" in coordinator.wait_message("evaluation")

    # 停止中の新しい要求も開始しない
    coordinator.request("training", "学習 exp_002", lambda: started.append("c"))
    assert started == ["a"]

    coordinator.unblock()
    assert started == ["a", "b"]
    assert not coordinator.is_blocked
    assert changes


def test_release_inside_start_moves_to_next_request(coordinator):
    started = []
    tickets = {}

    def fail_immediately():
        started.append("a")
        coordinator.release(tickets["a"])

    coordinator.request("training", "占有", lambda: started.append("x"))
    holder = coordinator.active
    tickets["a"] = coordinator.request("training", "準備に失敗", fail_immediately)
    coordinator.request("evaluation", "評価", lambda: started.append("b"))

    coordinator.release(holder)
    assert started == ["x", "a", "b"]
    assert coordinator.active_label == "評価"
