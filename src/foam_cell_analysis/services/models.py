"""画面とサービスで共有する Qt 非依存のデータ型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DataItem:
    """作業データセット内の画像とマスク。"""

    item_id: str
    source_filename: str
    source_relpath: str
    channels: list[str]
    classification: str | None
    quality: str | None
    mask_revisions: list[str]
    selected_mask_revision: str
    included: bool = True
    change: str | None = None
    previous_change: str | None = None
    sha256: str = ""
    seed: int = 0


@dataclass
class CheckResult:
    """整合性検査項目の結果。"""

    key: str
    label: str
    status: str = "未実行"
    count: int = 0


@dataclass
class ValidationIssue:
    """整合性検査で見つかった問題。"""

    severity: str
    item_id: str
    check: str
    message: str


@dataclass
class ValidationReport:
    """作業データセットの検査結果。"""

    checks: list[CheckResult] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)


@dataclass
class WorkingDataset:
    """編集可能なデータセット。"""

    purpose: str
    base_version: str
    items: list[DataItem]
    state: str = "WORKING"
    last_saved_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    validation: ValidationReport | None = None


@dataclass
class DatasetVersion:
    """確定済みデータセット版。"""

    version: str
    purpose: str
    parent_version: str | None
    created_at: datetime
    item_ids: list[str]
    n_images: int
    comment: str = ""
    status: str = "RELEASED"
    archive_status: str = "NOT_CREATED"
    archive_sha256: str | None = None
    base_validation_version: str | None = None
    app_version: str = "0.1.0"


@dataclass
class ImportCandidate:
    """取り込み検索で見つかった画像・マスクの組。"""

    source_filename: str
    source_relpath: str
    channels: list[str]
    mask_available: bool
    seed: int
    sha256: str


@dataclass
class TransformSetting:
    """データ拡張の変換設定。"""

    key: str
    label: str
    enabled: bool = True
    probability: float = 0.5
    range_min: float | None = None
    range_max: float | None = None


@dataclass
class AugmentationProfile:
    """バージョン付きデータ拡張設定。"""

    profile_id: str
    name: str
    base_profile: str | None
    transforms: list[TransformSetting]
    order: list[str]
    used_by_experiments: list[str] = field(default_factory=list)


@dataclass
class EpochMetrics:
    """学習エポックの記録。"""

    epoch: int
    loss: float
    map: float | None = None


@dataclass
class Checkpoint:
    """途中保存モデル。"""

    name: str
    epoch: int
    map: float | None
    saved_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


@dataclass
class RunAttempt:
    """学習実行試行。"""

    attempt: int
    started_at: datetime
    finished_at: datetime | None = None
    result: str = "実行中"
    environment: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """実験設定を保持する辞書ラッパー。"""

    values: dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        """設定を YAML 文字列にする。"""
        import yaml

        return yaml.safe_dump(self.values, allow_unicode=True, sort_keys=False)


@dataclass
class Experiment:
    """学習実験とその実行履歴。"""

    experiment_id: str
    study_id: str
    description: str
    model_type: str
    config: ExperimentConfig
    status: str
    current_epoch: int = 0
    total_epochs: int = 100
    history: list[EpochMetrics] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    runs: list[RunAttempt] = field(default_factory=list)
    used_item_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


@dataclass
class InferenceConfig:
    """推論設定。"""

    config_id: str
    model_type: str
    params: dict[str, Any]


@dataclass
class Evaluation:
    """全体および分類別の評価値。"""

    overall_map: float
    per_class: dict[str, tuple[float, int]]


@dataclass
class ExternalResult:
    """外部粒子解析の入力結果。"""

    name: str
    value: float | str
    unit: str = ""


@dataclass
class Candidate:
    """比較・リリース候補。"""

    candidate_id: str
    experiment_id: str
    checkpoint: str
    inference_config_id: str
    status: str = "candidate"
    evaluations: dict[str, Evaluation] = field(default_factory=dict)
    external_results: list[ExternalResult] = field(default_factory=list)
    external_software: str = ""
    external_date: str = ""
    comment: str = ""


@dataclass
class ReleasedModel:
    """変更不可のリリースモデル。"""

    model_id: str
    candidate_id: str
    experiment_id: str
    checkpoint: str
    preprocessing_config: dict[str, Any]
    inference_config: dict[str, Any]
    validation_dataset: str
    evaluation_result: Evaluation
    released_at: datetime
    comment: str = ""


@dataclass
class RoutingEntry:
    """分類に割り当てられた有効モデル。"""

    classification: str
    model_id: str | None


@dataclass
class RoutingHistory:
    """振り分け変更履歴。"""

    changed_at: datetime
    classification: str
    before_model_id: str | None
    after_model_id: str | None
