"""軽量なバックエンド契約値。標準ライブラリだけに依存する。"""

PARTICLE_SPLIT_ID = "particle_split_symmetric8_v1"


def training_eval_params() -> dict[str, dict[str, int | float | bool]]:
    """学習時評価の既定条件を、新しい辞書として返す。"""
    return {
        "mask_rcnn": {
            "box_score_thresh": 0.5,
            "box_nms_thresh": 0.5,
            "box_detections_per_img": 300,
            "mask_thresh": 0.5,
        },
        "cellpose": {
            "channel_axis": 2,
            "normalize": False,
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "min_size": 15,
            "max_size_fraction": 0.4,
            "bsize": 256,
        },
    }
