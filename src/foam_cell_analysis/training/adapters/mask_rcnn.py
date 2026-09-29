"""Mask R-CNN のモデル構築、学習、推論アダプタ。"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from foam_cell_analysis.training.adapters.base import Sample
from foam_cell_analysis.training.preprocessing import to_model_channels

DEFAULT_EVAL_PARAMS = {
    "box_score_thresh": 0.5,
    "box_nms_thresh": 0.5,
    "box_detections_per_img": 300,
    "mask_thresh": 0.5,
}


def _torchvision_components():
    """遅延 import で torchvision の構築関数を返す。"""
    from torchvision.models import ResNet50_Weights, ResNet101_Weights
    from torchvision.models.detection import (
        MaskRCNN,
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models.detection.rpn import RPNHead

    return {
        "MaskRCNN": MaskRCNN,
        "MaskRCNN_ResNet50_FPN_V2_Weights": MaskRCNN_ResNet50_FPN_V2_Weights,
        "ResNet50_Weights": ResNet50_Weights,
        "ResNet101_Weights": ResNet101_Weights,
        "maskrcnn_resnet50_fpn_v2": maskrcnn_resnet50_fpn_v2,
        "resnet_fpn_backbone": resnet_fpn_backbone,
        "AnchorGenerator": AnchorGenerator,
        "FastRCNNPredictor": FastRCNNPredictor,
        "MaskRCNNPredictor": MaskRCNNPredictor,
        "RPNHead": RPNHead,
    }


def _model_kwargs(model_config: dict[str, Any], eval_params: dict[str, Any]) -> dict[str, Any]:
    """設定値を torchvision の Mask R-CNN 引数へ写す。"""
    input_config = model_config.get("input", {})
    rpn = model_config.get("rpn", {})
    roi = model_config.get("roi", {})

    def shared_nms_limit(value: Any, default: int) -> int:
        if not isinstance(value, dict):
            return int(value)
        train = int(value.get("train", default))
        test = int(value.get("test", train))
        if train != test:
            raise ValueError("Mask R-CNN の NMS 件数は train/test で同じ値にしてください")
        return train

    pre_nms = shared_nms_limit(rpn.get("pre_nms_top_n", 2000), 2000)
    post_nms = shared_nms_limit(rpn.get("post_nms_top_n", 1000), 1000)
    eval_config = DEFAULT_EVAL_PARAMS | dict(eval_params)
    return {
        "min_size": int(input_config.get("min_size", 800)),
        "max_size": int(input_config.get("max_size", 1333)),
        "image_mean": input_config.get("image_mean", [0.485, 0.456, 0.406]),
        "image_std": input_config.get("image_std", [0.229, 0.224, 0.225]),
        "rpn_fg_iou_thresh": float(rpn.get("fg_iou_thresh", 0.7)),
        "rpn_bg_iou_thresh": float(rpn.get("bg_iou_thresh", 0.3)),
        "rpn_batch_size_per_image": int(rpn.get("batch_size_per_image", 256)),
        "rpn_positive_fraction": float(rpn.get("positive_fraction", 0.5)),
        "rpn_pre_nms_top_n_train": pre_nms,
        "rpn_pre_nms_top_n_test": pre_nms,
        "rpn_post_nms_top_n_train": post_nms,
        "rpn_post_nms_top_n_test": post_nms,
        "rpn_nms_thresh": float(rpn.get("nms_thresh", 0.7)),
        "box_fg_iou_thresh": float(roi.get("fg_iou_thresh", 0.5)),
        "box_bg_iou_thresh": float(roi.get("bg_iou_thresh", 0.5)),
        "box_batch_size_per_image": int(roi.get("batch_size_per_image", 512)),
        "box_positive_fraction": float(roi.get("positive_fraction", 0.25)),
        "box_score_thresh": float(eval_config["box_score_thresh"]),
        "box_nms_thresh": float(eval_config["box_nms_thresh"]),
        "box_detections_per_img": int(eval_config["box_detections_per_img"]),
    }


def _weights_cache_path(url: str) -> Path:
    """TORCH_HOME に対応する torchvision チェックポイントの位置を返す。"""
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        root = Path(torch_home)
    else:
        import torch

        root = Path(torch.hub.get_dir()).parent
    return root / "hub" / "checkpoints" / Path(urlparse(url).path).name


def _labels_to_target(labels: np.ndarray) -> dict[str, Any]:
    """整数インスタンスラベルを Mask R-CNN のマスク・箱へ変換する。"""
    import torch

    label_image = np.asarray(labels)
    if label_image.ndim != 2 or not np.issubdtype(label_image.dtype, np.integer):
        raise ValueError("Mask R-CNN のラベルは 2 次元整数画像である必要があります")
    height, width = label_image.shape
    masks = []
    boxes = []
    for instance_id in np.unique(label_image):
        if instance_id == 0:
            continue
        mask = label_image == instance_id
        rows, columns = np.nonzero(mask)
        if not len(rows):
            continue
        masks.append(mask.astype(np.uint8))
        boxes.append(
            [
                float(columns.min()),
                float(rows.min()),
                float(columns.max() + 1),
                float(rows.max() + 1),
            ]
        )
    if masks:
        mask_tensor = torch.from_numpy(np.stack(masks))
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
    else:
        mask_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
        box_tensor = torch.zeros((0, 4), dtype=torch.float32)
    return {
        "boxes": box_tensor,
        "labels": torch.ones((len(boxes),), dtype=torch.int64),
        "masks": mask_tensor,
        "image_id": torch.tensor([0], dtype=torch.int64),
        "area": (
            mask_tensor.flatten(1).sum(1).to(torch.float32)
            if boxes
            else torch.zeros((0,), dtype=torch.float32)
        ),
        "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
    }


def _predictions_to_labels(
    prediction: dict[str, Any], eval_params: dict[str, Any] | None = None
) -> np.ndarray:
    """低スコアからマスクを塗り、後から来る高スコアで上書きする。"""
    params = DEFAULT_EVAL_PARAMS | dict(eval_params or {})
    masks = prediction["masks"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    labels = prediction.get("labels")
    class_ids = labels.detach().cpu().numpy() if labels is not None else None
    shape = tuple(int(size) for size in masks.shape[-2:])
    canvas = np.zeros(shape, dtype=np.uint32)
    threshold = float(params["box_score_thresh"])
    mask_threshold = float(params["mask_thresh"])
    accepted = [
        index
        for index, score in enumerate(scores)
        if float(score) >= threshold and (class_ids is None or int(class_ids[index]) == 1)
    ]
    accepted.sort(key=lambda index: (float(scores[index]), index))
    next_label = 1
    for index in accepted:
        mask = masks[index]
        if mask.ndim == 3:
            mask = mask[0]
        foreground = mask >= mask_threshold
        if foreground.any():
            canvas[foreground] = next_label
            next_label += 1
    present = np.unique(canvas)
    present = present[present != 0]
    if len(present) == next_label - 1:
        return canvas
    compact = np.zeros_like(canvas)
    for new_label, old_label in enumerate(present, start=1):
        compact[canvas == old_label] = new_label
    return compact


class MaskRCNNAdapter:
    """torchvision Mask R-CNN を共通学習ループへ接続する。"""

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        self.device = None
        self.model_config: dict[str, Any] = {}
        self.eval_params = DEFAULT_EVAL_PARAMS.copy()
        self.batch_size = 1
        self.backbone = "resnet50_fpn_v2"

    def initial_weights_file(self, model_config: dict[str, Any]) -> Path | None:
        """選択した torchvision 重みがキャッシュされるパスを返す。"""
        parts = _torchvision_components()
        backbone = model_config.get("backbone", "resnet50_fpn_v2")
        pretrained = model_config.get("pretrained_weights")
        if pretrained in {None, "none"}:
            return None
        if backbone == "resnet50_fpn_v2" and pretrained == "coco":
            weights = parts["MaskRCNN_ResNet50_FPN_V2_Weights"].COCO_V1
        elif backbone == "resnet50_fpn_v2" and pretrained == "imagenet":
            weights = parts["ResNet50_Weights"].IMAGENET1K_V1
        elif backbone == "resnet101_fpn" and pretrained == "imagenet":
            weights = parts["ResNet101_Weights"].IMAGENET1K_V1
        else:
            return None
        return _weights_cache_path(weights.url)

    def build(self, model_config: dict[str, Any], device: Any) -> None:
        """設定に応じて backbone と 2 クラス検出ヘッドを構築する。"""
        parts = _torchvision_components()
        backbone_name = model_config.get("backbone", "resnet50_fpn_v2")
        pretrained = model_config.get("pretrained_weights")
        trainable_layers = int(model_config.get("trainable_backbone_layers", 3))
        kwargs = _model_kwargs(model_config, model_config.get("eval_params", {}))

        if backbone_name == "resnet50_fpn_v2":
            builder_kwargs: dict[str, Any] = {
                "trainable_backbone_layers": trainable_layers,
                **kwargs,
            }
            if pretrained == "coco":
                weights = parts["MaskRCNN_ResNet50_FPN_V2_Weights"].COCO_V1
                builder_kwargs["weights"] = weights
            elif pretrained == "imagenet":
                weights = parts["ResNet50_Weights"].IMAGENET1K_V1
                builder_kwargs.update(
                    weights=None,
                    weights_backbone=weights,
                    num_classes=2,
                )
            elif pretrained in {None, "none"}:
                builder_kwargs.update(
                    weights=None,
                    weights_backbone=None,
                    num_classes=2,
                )
            else:
                raise ValueError(f"ResNet50 FPN v2 の重み指定が不正です: {pretrained}")
            model = parts["maskrcnn_resnet50_fpn_v2"](**builder_kwargs)
            if pretrained == "coco":
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = parts["FastRCNNPredictor"](in_features, 2)
                mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
                mask_channels = model.roi_heads.mask_predictor.conv5_mask.out_channels
                model.roi_heads.mask_predictor = parts["MaskRCNNPredictor"](
                    mask_features, mask_channels, 2
                )
            conv_depth = 2
        elif backbone_name == "resnet101_fpn":
            if pretrained == "coco":
                raise ValueError("ResNet101 と COCO 重みは組み合わせできません")
            if pretrained == "imagenet":
                backbone_weights = parts["ResNet101_Weights"].IMAGENET1K_V1
            elif pretrained in {None, "none"}:
                backbone_weights = None
            else:
                raise ValueError(f"ResNet101 FPN の重み指定が不正です: {pretrained}")
            backbone = parts["resnet_fpn_backbone"](
                backbone_name="resnet101",
                weights=backbone_weights,
                trainable_layers=trainable_layers,
            )
            model = parts["MaskRCNN"](backbone, num_classes=2, **kwargs)
            conv_depth = 1
        else:
            raise ValueError(f"未対応の Mask R-CNN backbone です: {backbone_name}")

        anchor_config = model_config.get("anchors", {})
        sizes = tuple(int(value) for value in anchor_config.get("sizes", (32, 64, 128, 256, 512)))
        ratios = tuple(
            float(value) for value in anchor_config.get("aspect_ratios", (0.5, 1.0, 2.0))
        )
        if len(sizes) != 5:
            raise ValueError("Mask R-CNN のアンカーサイズは 5 個必要です")
        model.rpn.anchor_generator = parts["AnchorGenerator"](
            sizes=tuple((size,) for size in sizes),
            aspect_ratios=tuple(ratios for _ in sizes),
        )
        if len(ratios) != 3:
            model.rpn.head = parts["RPNHead"](
                model.backbone.out_channels, len(ratios), conv_depth=conv_depth
            )
            warnings.warn(
                "アンカー数が既定値から変わったため RPN ヘッドを再生成しました",
                RuntimeWarning,
                stacklevel=2,
            )
        self.model = model.to(device)
        self.device = device
        self.model_config = dict(model_config)
        self.eval_params = DEFAULT_EVAL_PARAMS | dict(model_config.get("eval_params", {}))
        self.backbone = backbone_name
        self.optimizer = None

    def make_optimizer(self, training: dict[str, Any]) -> Any:
        """Mask R-CNN 用の SGD または AdamW を作る。"""
        import torch

        if self.model is None:
            raise RuntimeError("モデルが構築されていません")
        self.batch_size = int(training.get("batch_size", 1))
        learning_rate = float(training.get("learning_rate", 1e-3))
        weight_decay = float(training.get("weight_decay", 0.0))
        optimizer_name = self.model_config.get("optimizer", "SGD")
        if optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"未対応の最適化手法です: {optimizer_name}")
        return self.optimizer

    def lr(self, epoch: int, total_epochs: int, base_lr: float) -> float:
        """Mask R-CNN の学習率は全エポックで一定にする。"""
        del epoch, total_epochs
        return float(base_lr)

    def should_exclude_training_item(self, mask: np.ndarray, model_config: dict[str, Any]) -> bool:
        del mask, model_config
        return False

    def training_exclusion_reason(
        self, mask: np.ndarray, model_config: dict[str, Any]
    ) -> str | None:
        del mask, model_config
        return None

    def epoch_item_ids(
        self, item_ids: list[str], model_config: dict[str, Any], rng: Any
    ) -> list[str]:
        del model_config, rng
        return list(item_ids)

    def prepare_sample(self, sample: Sample, *, training: bool, rng: Any) -> Sample:
        del training, rng
        image = np.asarray(sample.image, dtype=np.float32)
        if image.ndim == 2:
            image = to_model_channels(image, "mask_rcnn")
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Mask R-CNN の入力画像は 1 チャンネル画像である必要があります")
        return Sample(sample.item_id, np.ascontiguousarray(image), sample.mask)

    @staticmethod
    def _image_tensor(image: np.ndarray, device: Any):
        import torch

        return torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).to(device)

    def _target(self, mask: np.ndarray) -> dict[str, Any]:
        target = _labels_to_target(mask)
        return {key: value.to(self.device) for key, value in target.items()}

    def train_one_epoch(self, samples: Iterable[Sample], lr: float) -> float:
        """画像を batch_size 枚ずつ渡して学習する。"""
        import torch

        if self.model is None or self.optimizer is None:
            raise RuntimeError("モデルと optimizer が必要です")
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)
        self.model.train()
        batch: list[Sample] = []
        batch_losses: list[float] = []

        def train_batch(rows: list[Sample]) -> None:
            images = [self._image_tensor(row.image, self.device) for row in rows]
            targets = [self._target(row.mask) for row in rows]
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.model(images, targets)
            total_loss = sum(losses.values())
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Mask R-CNN の loss が有限値ではありません")
            total_loss.backward()
            self.optimizer.step()
            batch_losses.append(float(total_loss.detach().cpu()))

        for sample in samples:
            batch.append(sample)
            if len(batch) >= self.batch_size:
                train_batch(batch)
                batch = []
        if batch:
            train_batch(batch)
        return float(np.mean(batch_losses)) if batch_losses else 0.0

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """評価パラメータで推論し、インスタンスラベル画像へ変換する。"""
        import torch

        if self.model is None:
            raise RuntimeError("モデルが構築されていません")
        self.model.eval()
        tensors = [self._image_tensor(image, self.device) for image in images]
        with torch.no_grad():
            predictions = self.model(tensors)
        return [_predictions_to_labels(prediction, self.eval_params) for prediction in predictions]

    def state_dict(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("モデルが構築されていません")
        return self.model.state_dict()

    def weights_nbytes(self) -> int:
        """現在のモデル重みを fp32 として保存した場合のサイズを返す。"""
        if self.model is None:
            raise RuntimeError("モデルが構築されていません")
        return sum(
            tensor.numel() * tensor.element_size() for tensor in self.model.state_dict().values()
        )


def build_for_inference(
    model_config: dict[str, Any], weights_path: str | Path, device: Any = "cpu"
) -> MaskRCNNAdapter:
    """初期重みを取得せず、学習済み state_dict のみで復元する。"""
    import torch

    config = dict(model_config)
    config["pretrained_weights"] = None
    adapter = MaskRCNNAdapter()
    adapter.build(config, device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    adapter.model.load_state_dict(state)
    adapter.model.eval()
    return adapter
