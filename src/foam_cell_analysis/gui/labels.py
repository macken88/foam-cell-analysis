"""内部キー・状態を画面表示用の日本語へ変換する。"""

from datetime import datetime

_MODEL_TYPES = {"mask_rcnn": "Mask R-CNN", "cellpose": "Cellpose"}
_CLASSIFICATIONS = {"all": "全分類"}
_QUALITY_FILTERS = {
    "all": "すべて",
    "good": "良のみ",
    "good_only": "良のみ",
    "good_and_acceptable": "良・可",
}
_EXPERIMENT_STATUSES = {
    "draft": "下書き",
    "running": "実行中",
    "completed": "完了",
    "failed": "失敗",
    "stopped": "中断",
}
_CANDIDATE_STATUSES = {
    "candidate": "候補",
    "evaluating": "評価中",
    "released": "リリース済み",
    "rejected": "非採用",
}

_CONFIG_LABELS = {
    "experiment.id": "実験識別子",
    "experiment.study_id": "実験群",
    "experiment.description": "説明",
    "data.dataset_version": "データセット版",
    "dataset_version": "データセット版",
    "data.split_id": "交差検証分割",
    "split_id": "交差検証分割",
    "data.seed": "乱数シード",
    "seed": "乱数シード",
    "data.classification": "画像分類",
    "classification": "画像分類",
    "data.quality_filter": "品質条件",
    "quality_filter": "品質条件",
    "data.input_channels": "入力チャンネル",
    "input_channels": "入力チャンネル",
    "data.used_item_ids": "実使用データ一覧",
    "used_item_ids": "実使用データ一覧",
    "training.epochs": "エポック数",
    "epochs": "エポック数",
    "training.batch_size": "バッチサイズ",
    "batch_size": "バッチサイズ",
    "training.learning_rate": "学習率",
    "learning_rate": "学習率",
    "training.weight_decay": "重み減衰",
    "weight_decay": "重み減衰",
    "training.early_stopping.enabled": "早期終了",
    "early_stopping.enabled": "早期終了",
    "training.early_stopping.patience": "待機エポック数",
    "early_stopping.patience": "待機エポック数",
    "augmentation.profile": "データ拡張プロファイル",
    "validation_interval": "検証間隔",
    "checkpoint.validation_interval": "検証間隔",
    "save_every": "途中保存モデル保存間隔",
    "checkpoint.save_every": "途中保存モデル保存間隔",
    "checkpoint.save_last": "最終モデルを保存",
    "checkpoint.save_best": "最良モデルを保存",
    "checkpoint.best_metric": "最良モデル判定指標",
    "checkpoint.best_mode": "最良モデル判定方向",
    "model.type": "モデル種類",
    "model.pretrained_weights": "事前学習済み重み",
    "pretrained_weights": "事前学習済み重み",
    "model.backbone": "バックボーン",
    "backbone": "バックボーン",
    "model.trainable_backbone_layers": "学習するバックボーン層数",
    "trainable_backbone_layers": "学習するバックボーン層数",
    "model.num_classes": "クラス数",
    "num_classes": "クラス数",
    "model.input.min_size": "入力画像の短辺サイズ",
    "min_size": "入力画像の短辺サイズ",
    "model.input.max_size": "入力画像の最大辺サイズ",
    "max_size": "入力画像の最大辺サイズ",
    "model.input.image_mean": "画像平均",
    "image_mean": "画像平均",
    "model.input.image_std": "画像標準偏差",
    "image_std": "画像標準偏差",
    "model.anchors.sizes": "アンカーサイズ",
    "anchor_sizes": "アンカーサイズ",
    "model.anchors.aspect_ratios": "アンカー縦横比",
    "anchor_aspect_ratios": "アンカー縦横比",
    "model.rpn.fg_iou_thresh": "RPN前景IoU閾値",
    "rpn_fg_iou_thresh": "RPN前景IoU閾値",
    "model.rpn.bg_iou_thresh": "RPN背景IoU閾値",
    "rpn_bg_iou_thresh": "RPN背景IoU閾値",
    "model.rpn.batch_size_per_image": "RPNバッチ数 / 画像",
    "rpn_batch_size_per_image": "RPNバッチ数 / 画像",
    "model.rpn.positive_fraction": "RPN正例割合",
    "rpn_positive_fraction": "RPN正例割合",
    "model.rpn.pre_nms_top_n": "RPN NMS前 top N",
    "rpn_pre_nms_top_n": "RPN NMS前 top N",
    "model.rpn.post_nms_top_n": "RPN NMS後 top N",
    "rpn_post_nms_top_n": "RPN NMS後 top N",
    "model.rpn.nms_thresh": "RPN NMS閾値",
    "rpn_nms_thresh": "RPN NMS閾値",
    "model.roi.fg_iou_thresh": "Box前景IoU閾値",
    "box_fg_iou_thresh": "Box前景IoU閾値",
    "model.roi.bg_iou_thresh": "Box背景IoU閾値",
    "box_bg_iou_thresh": "Box背景IoU閾値",
    "model.roi.batch_size_per_image": "ROIバッチ数 / 画像",
    "box_batch_size_per_image": "ROIバッチ数 / 画像",
    "model.roi.positive_fraction": "ROI正例割合",
    "box_positive_fraction": "ROI正例割合",
    "model.pretrained_model": "事前学習済みモデル",
    "pretrained_model": "事前学習済みモデル",
    "model.optimizer": "最適化手法",
    "optimizer": "最適化手法",
    "model.normalize": "正規化",
    "normalize": "正規化",
    "model.scale_range": "スケール変動幅",
    "scale_range": "スケール変動幅",
    "model.rescale": "リスケール",
    "rescale": "リスケール",
    "model.bsize": "学習パッチサイズ",
    "bsize": "学習パッチサイズ",
    "model.nimg_per_epoch": "1エポック当たり画像数",
    "nimg_per_epoch": "1エポック当たり画像数",
    "model.min_train_masks": "最小マスク数",
    "min_train_masks": "最小マスク数",
    "model.class_weights": "クラス重み",
    "class_weights": "クラス重み",
    "box_score_thresh": "検出スコア閾値",
    "box_nms_thresh": "Box NMS閾値",
    "box_detections_per_img": "最大検出数",
    "cellprob_threshold": "セル確率閾値",
    "flow_threshold": "フロー閾値",
    "enabled": "有効 / 無効",
    "probability": "適用確率",
    "range": "変換範囲",
    "horizontal_flip": "左右反転",
    "vertical_flip": "上下反転",
    "rotation": "回転",
    "scale": "拡大縮小",
    "translate": "平行移動",
    "crop": "切り出し",
    "elastic": "弾性変形",
    "brightness": "明るさ変化",
    "contrast": "コントラスト変化",
    "gamma": "ガンマ",
    "blur": "ぼかし",
    "noise": "ノイズ付与",
    "channel_dropout": "チャンネルドロップ",
    "channel_intensity": "チャンネル強度変動",
    "pipeline_order": "適用順序",
}


def model_type_label(value: str) -> str:
    """モデル種別を画面表示名へ変換する。"""
    return _MODEL_TYPES.get(value, value)


def classification_label(value: str) -> str:
    """画像分類キーを画面表示名へ変換する。"""
    return _CLASSIFICATIONS.get(value, value)


def quality_filter_label(value: str) -> str:
    """品質条件キーを画面表示名へ変換する。"""
    return _QUALITY_FILTERS.get(value, value)


def experiment_status_label(value: str) -> str:
    """実験状態を画面表示名へ変換する。"""
    return _EXPERIMENT_STATUSES.get(value, value)


def candidate_status_label(value: str) -> str:
    """候補状態を画面表示名へ変換する。"""
    return _CANDIDATE_STATUSES.get(value, value)


def format_score(value: float | None, digits: int = 3) -> str:
    """スコアを指定桁数で表示する。値が無い場合はダッシュにする。"""
    return "—" if value is None else f"{value:.{digits}f}"


def format_datetime(value: datetime) -> str:
    """日時をローカル時刻の年月日時分で表示する。"""
    return value.astimezone().strftime("%Y-%m-%d %H:%M")


def config_key_label(dotted_key: str) -> str:
    """仕様書14章の内部キーを表示名へ変換する。"""
    return _CONFIG_LABELS.get(dotted_key, dotted_key)
