from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from foam_cell_analysis.training.adapters import mask_rcnn as mask_module
from foam_cell_analysis.training.adapters.base import Sample
from foam_cell_analysis.training.adapters.mask_rcnn import (
    MaskRCNNAdapter,
    _labels_to_target,
    _model_kwargs,
    _predictions_to_labels,
    build_for_inference,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.backbone = SimpleNamespace(out_channels=64)
        self.rpn = SimpleNamespace(anchor_generator=None, head="default")
        self.roi_heads = SimpleNamespace(
            box_predictor=SimpleNamespace(cls_score=SimpleNamespace(in_features=8)),
            mask_predictor=SimpleNamespace(
                conv5_mask=SimpleNamespace(in_channels=4, out_channels=8)
            ),
        )
        self.loaded_state = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.loaded_state = state_dict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def _config(backbone="resnet50_fpn_v2", pretrained="none", aspect_ratios=None):
    return {
        "type": "mask_rcnn",
        "backbone": backbone,
        "pretrained_weights": pretrained,
        "trainable_backbone_layers": 2,
        "optimizer": "SGD",
        "input": {
            "min_size": 256,
            "max_size": 256,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "anchors": {
            "sizes": [16, 32, 64, 128, 256],
            "aspect_ratios": aspect_ratios or [0.5, 1.0, 2.0],
        },
        "rpn": {
            "fg_iou_thresh": 0.65,
            "bg_iou_thresh": 0.25,
            "batch_size_per_image": 192,
            "positive_fraction": 0.4,
            "pre_nms_top_n": 1100,
            "post_nms_top_n": 700,
            "nms_thresh": 0.6,
        },
        "roi": {
            "fg_iou_thresh": 0.55,
            "bg_iou_thresh": 0.45,
            "batch_size_per_image": 384,
            "positive_fraction": 0.3,
        },
    }


def _stub_torchvision(monkeypatch):
    parts = mask_module._torchvision_components()
    calls = {}

    def build_v2(**kwargs):
        calls["v2"] = kwargs
        return TinyModel()

    def build_backbone(**kwargs):
        calls["backbone"] = kwargs
        return SimpleNamespace(out_channels=64)

    def build_resnet101(backbone, **kwargs):
        calls["resnet101"] = (backbone, kwargs)
        return TinyModel()

    def build_rpn_head(channels, anchors, conv_depth=1):
        calls["rpn_head"] = (channels, anchors, conv_depth)
        return SimpleNamespace(num_anchors=anchors, conv_depth=conv_depth)

    parts.update(
        maskrcnn_resnet50_fpn_v2=build_v2,
        resnet_fpn_backbone=build_backbone,
        MaskRCNN=build_resnet101,
        AnchorGenerator=lambda **kwargs: kwargs,
        FastRCNNPredictor=lambda channels, classes: ("box", channels, classes),
        MaskRCNNPredictor=lambda channels, reduced, classes: (
            "mask",
            channels,
            reduced,
            classes,
        ),
        RPNHead=build_rpn_head,
    )
    monkeypatch.setattr(mask_module, "_torchvision_components", lambda: parts)
    return parts, calls


def test_model_kwargs_map_settings_and_share_pre_post_nms_limits():
    kwargs = _model_kwargs(_config(), mask_module.DEFAULT_EVAL_PARAMS)

    assert kwargs["min_size"] == 256
    assert kwargs["max_size"] == 256
    assert kwargs["rpn_pre_nms_top_n_train"] == kwargs["rpn_pre_nms_top_n_test"] == 1100
    assert kwargs["rpn_post_nms_top_n_train"] == kwargs["rpn_post_nms_top_n_test"] == 700
    assert kwargs["box_score_thresh"] == 0.5
    assert kwargs["box_nms_thresh"] == 0.5
    assert kwargs["box_detections_per_img"] == 300


@pytest.mark.parametrize(
    ("config", "expected_branch"),
    [
        (_config(pretrained="coco"), "v2"),
        (_config(pretrained="imagenet"), "v2"),
        (_config(backbone="resnet101_fpn", pretrained="imagenet"), "resnet101"),
    ],
)
def test_three_supported_backbone_weight_combinations(monkeypatch, config, expected_branch):
    parts, calls = _stub_torchvision(monkeypatch)
    adapter = MaskRCNNAdapter()
    adapter.build(config, "cpu")

    if expected_branch == "v2":
        kwargs = calls["v2"]
        assert kwargs["trainable_backbone_layers"] == 2
        assert kwargs["min_size"] == kwargs["max_size"] == 256
        if config["pretrained_weights"] == "coco":
            assert kwargs["weights"] is parts["MaskRCNN_ResNet50_FPN_V2_Weights"].COCO_V1
            assert not {"num_classes", "weights_backbone", "box_predictor", "mask_predictor"} & (
                set(kwargs)
            )
            assert adapter.model.roi_heads.box_predictor == ("box", 8, 2)
            assert adapter.model.roi_heads.mask_predictor == ("mask", 4, 8, 2)
        else:
            assert kwargs["weights"] is None
            assert kwargs["weights_backbone"] is parts["ResNet50_Weights"].IMAGENET1K_V1
            assert kwargs["num_classes"] == 2
    else:
        assert calls["backbone"]["backbone_name"] == "resnet101"
        assert calls["backbone"]["weights"] is parts["ResNet101_Weights"].IMAGENET1K_V1
        assert calls["backbone"]["trainable_layers"] == 2
        _backbone, kwargs = calls["resnet101"]
        assert kwargs["num_classes"] == 2
    assert len(adapter.model.rpn.anchor_generator["sizes"]) == 5
    assert adapter.optimizer is None


def test_non_three_anchor_count_rebuilds_rpn_head_and_warns(monkeypatch):
    _parts, calls = _stub_torchvision(monkeypatch)
    adapter = MaskRCNNAdapter()
    config = _config(aspect_ratios=[0.5, 1.0])

    with pytest.warns(RuntimeWarning, match="RPN ヘッドを再生成"):
        adapter.build(config, "cpu")

    assert calls["rpn_head"] == (64, 2, 2)
    assert adapter.model.rpn.head.num_anchors == 2


@pytest.mark.parametrize(
    ("optimizer_name", "expected_type"),
    [("SGD", torch.optim.SGD), ("AdamW", torch.optim.AdamW)],
)
def test_optimizer_choices_and_constant_learning_rate(optimizer_name, expected_type):
    adapter = MaskRCNNAdapter()
    adapter.model = torch.nn.Linear(2, 2)
    adapter.model_config = {"optimizer": optimizer_name}
    optimizer = adapter.make_optimizer(
        {"learning_rate": 0.01, "weight_decay": 0.002, "batch_size": 2}
    )

    assert isinstance(optimizer, expected_type)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert adapter.lr(1, 5, 0.01) == adapter.lr(5, 5, 0.01) == 0.01
    if optimizer_name == "SGD":
        assert optimizer.param_groups[0]["momentum"] == 0.9


def test_labels_become_instance_masks_boxes_and_empty_tensor_shapes():
    empty = _labels_to_target(np.zeros((4, 5), dtype=np.uint16))
    assert tuple(empty["boxes"].shape) == (0, 4)
    assert tuple(empty["masks"].shape) == (0, 4, 5)

    labels = np.zeros((5, 7), dtype=np.uint16)
    labels[1:3, 2:5] = 4
    labels[3:5, 5:7] = 12
    target = _labels_to_target(labels)
    assert tuple(target["masks"].shape) == (2, 5, 7)
    assert target["boxes"].tolist() == [[2.0, 1.0, 5.0, 3.0], [5.0, 3.0, 7.0, 5.0]]
    assert target["labels"].tolist() == [1, 1]


def test_input_is_replicated_to_rgb_and_labels_are_kept():
    labels = np.array([[0, 1], [2, 0]], dtype=np.uint16)
    sample = Sample("item", np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32), labels)

    prepared = MaskRCNNAdapter().prepare_sample(sample, training=True, rng=None)

    assert prepared.image.shape == (2, 2, 3)
    assert np.array_equal(prepared.image[:, :, 0], prepared.image[:, :, 2])
    assert prepared.mask.dtype == np.uint16
    assert prepared.mask is labels


def test_prediction_masks_paint_low_score_first_and_drop_covered_instances():
    masks = torch.zeros((3, 1, 3, 3), dtype=torch.float32)
    masks[0, 0, :, :] = 1
    masks[1, 0, :2, :2] = 1
    masks[2, 0, 1:, 1:] = 1
    prediction = {
        "masks": masks,
        "scores": torch.tensor([0.2, 0.9, 0.7]),
        "labels": torch.tensor([1, 1, 1]),
    }

    labels = _predictions_to_labels(prediction, {"box_score_thresh": 0.1})

    assert labels.dtype == np.uint32
    assert labels[0, 0] == 3
    assert labels[2, 0] == 1
    assert labels[1, 1] == 3
    assert labels[2, 2] == 2
    assert set(np.unique(labels)) == {1, 2, 3}

    completely_covered = {
        "masks": torch.stack([masks[0], torch.ones_like(masks[0])]),
        "scores": torch.tensor([0.2, 0.9]),
        "labels": torch.tensor([1, 1]),
    }
    final = _predictions_to_labels(completely_covered, {"box_score_thresh": 0.1})
    assert set(np.unique(final)) == {1}


def test_initial_weights_file_uses_torch_home_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    path = MaskRCNNAdapter().initial_weights_file(_config(pretrained="coco"))
    assert path == tmp_path / "hub" / "checkpoints" / "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"


def test_trained_weights_rebuild_does_not_request_pretrained_download(tmp_path, monkeypatch):
    _stub_torchvision(monkeypatch)
    adapter = MaskRCNNAdapter()
    adapter.build(_config(pretrained="none"), "cpu")
    checkpoint = tmp_path / "final.pt"
    torch.save(adapter.model.state_dict(), checkpoint)
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *_args, **_kwargs: pytest.fail("unexpected network download"),
    )

    restored = build_for_inference(_config(pretrained="coco"), checkpoint, "cpu")

    assert restored.model.loaded_state is not None
    assert restored.model.training is False
    assert restored.weights_nbytes() > 0


@pytest.mark.ml
def test_real_coco_model_attributes_and_offline_trained_weight_rebuild(tmp_path, monkeypatch):
    import gc

    cache_root = mask_module.Path.cwd() / "workspace" / "pretrained" / "torch"
    monkeypatch.setenv("TORCH_HOME", str(cache_root))
    coco_weights = MaskRCNNAdapter().initial_weights_file(_config(pretrained="coco"))
    if coco_weights is None or not coco_weights.is_file():
        pytest.skip("COCO 重みが TORCH_HOME にありません")

    config = _config(pretrained="coco")
    config["eval_params"] = {
        "box_score_thresh": 0.37,
        "box_nms_thresh": 0.42,
        "box_detections_per_img": 123,
    }
    adapter = MaskRCNNAdapter()
    adapter.build(config, "cpu")
    model = adapter.model

    model.train()
    assert model.rpn.pre_nms_top_n() == 1100
    assert model.rpn.post_nms_top_n() == 700
    model.eval()
    assert model.rpn.pre_nms_top_n() == 1100
    assert model.rpn.post_nms_top_n() == 700
    assert model.roi_heads.score_thresh == pytest.approx(0.37)
    assert model.roi_heads.nms_thresh == pytest.approx(0.42)
    assert model.roi_heads.detections_per_img == 123
    assert model.transform.min_size == (256,)
    assert model.transform.max_size == 256
    assert model.transform.image_mean == [0.485, 0.456, 0.406]
    assert model.transform.image_std == [0.229, 0.224, 0.225]
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2
    assert model.rpn.anchor_generator.sizes == ((16,), (32,), (64,), (128,), (256,))
    assert model.rpn.anchor_generator.aspect_ratios == ((0.5, 1.0, 2.0),) * 5

    final_checkpoint = tmp_path / "final.pt"
    torch.save(model.state_dict(), final_checkpoint)
    del adapter, model
    gc.collect()
    empty_torch_home = tmp_path / "empty_torch_home"
    empty_torch_home.mkdir()
    monkeypatch.setenv("TORCH_HOME", str(empty_torch_home))
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *_args, **_kwargs: pytest.fail("空 TORCH_HOME から重みをダウンロードしました"),
    )

    restored = build_for_inference(config, final_checkpoint, "cpu")

    assert restored.model.roi_heads.box_predictor.cls_score.out_features == 2
    assert restored.model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2


def test_resnet101_coco_combination_is_rejected(monkeypatch):
    _stub_torchvision(monkeypatch)
    with pytest.raises(ValueError, match="ResNet101 と COCO"):
        MaskRCNNAdapter().build(_config(backbone="resnet101_fpn", pretrained="coco"), "cpu")


def test_eval_params_reach_model_kwargs_and_labelization(monkeypatch, tmp_path):
    _parts, calls = _stub_torchvision(monkeypatch)
    config = _config(pretrained="coco")
    config["eval_params"] = {
        "box_score_thresh": 0.31,
        "box_nms_thresh": 0.42,
        "box_detections_per_img": 77,
        "mask_thresh": 0.8,
    }
    checkpoint = tmp_path / "final.pt"
    torch.save({"layer.weight": torch.zeros(2, 2)}, checkpoint)

    restored = build_for_inference(config, checkpoint, "cpu")

    kwargs = calls["v2"]
    assert kwargs["box_score_thresh"] == pytest.approx(0.31)
    assert kwargs["box_nms_thresh"] == pytest.approx(0.42)
    assert kwargs["box_detections_per_img"] == 77
    # 事前学習済み重みは取得しない
    assert kwargs["weights"] is None
    assert kwargs["weights_backbone"] is None
    assert restored.eval_params["mask_thresh"] == 0.8

    masks = torch.tensor([[[[0.6, 0.9]]]], dtype=torch.float32)
    prediction = {"masks": masks, "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}
    labels = _predictions_to_labels(prediction, restored.eval_params)
    assert labels.tolist() == [[0, 1]]
