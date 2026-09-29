import numpy as np

from foam_cell_analysis.inference.particle_split import (
    PARTICLE_SPLIT_ID,
    binary_mask,
    split_labels,
    split_report,
)


def _diagonal():
    labels = np.zeros((6, 6), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    labels[3:5, 3:5] = 2  # (2,2) と (3,3) が斜めに接する
    return labels


def test_diagonal_contact_is_separated_on_both_sides():
    labels = _diagonal()
    split = split_labels(labels)
    assert split[2, 2] == 0 and split[3, 3] == 0
    assert split[1, 1] == 1 and split[4, 4] == 2
    assert labels[2, 2] == 1  # 入力は変更しない
    assert split_report(labels)["remaining_contacts"] == 0
    assert binary_mask(labels).dtype == np.uint8
    assert set(np.unique(binary_mask(labels))) == {0, 255}
    assert PARTICLE_SPLIT_ID == "particle_split_symmetric8_v1"


def test_thin_bridge_splits_particle_and_reports_it():
    labels = np.zeros((5, 9), dtype=np.int32)
    labels[1:4, 0:3] = 5
    labels[2, 3:6] = 5  # 幅 1 の橋
    labels[1:4, 6:9] = 5
    labels[1, 4] = 7  # 橋の上に別ラベルを接して置き、橋を削る
    report = split_report(labels)
    assert report["remaining_contacts"] == 0
    assert 5 in report["split_labels"]
    assert report["removed_pixels"] > 0
    assert 0 < report["removed_fraction"] < 1


def test_single_pixel_particle_vanishes():
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[1:3, 0:2] = 1
    labels[1, 2] = 2  # 1 画素の粒子が接する
    report = split_report(labels)
    assert report["vanished_labels"] == [2]
    assert report["remaining_contacts"] == 0


def test_label_permutation_invariance():
    labels = _diagonal()
    permuted = labels.copy()
    permuted[labels == 1], permuted[labels == 2] = 9, 3
    assert np.array_equal(split_labels(labels) > 0, split_labels(permuted) > 0)
    assert split_report(labels)["removed_pixels"] == split_report(permuted)["removed_pixels"]


def test_empty_image_report():
    report = split_report(np.zeros((3, 3), dtype=np.uint16))
    assert report["removed_pixels"] == 0 and report["removed_fraction"] == 0.0
    assert report["vanished_labels"] == [] and report["split_labels"] == []


def test_label_with_two_components_before_is_not_counted_as_split():
    labels = np.zeros((5, 13), dtype=np.int32)
    labels[1:4, 0:3] = 5
    labels[2, 3:6] = 5  # 幅 1 の橋
    labels[1:4, 6:9] = 5
    labels[1, 4] = 7
    labels[1:4, 10:13] = 5  # 分離前から別の連結成分（合計 2 成分 -> 分離後 3 成分）
    report = split_report(labels)
    assert 5 not in report["split_labels"]
