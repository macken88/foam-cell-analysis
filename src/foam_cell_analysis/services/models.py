"""画面とサービスで共有する Qt 非依存のデータ型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PreparedRun:
    """起動可能な学習試行の説明。"""

    run_id: str
    run_dir: str
    program: str
    args: list[str]
    env: dict[str, str]
    fake: bool = False


@dataclass
class JobExit:
    """学習・評価プロセス終了時の補助情報。

    protocol_error は hello 以降に外形の欠けたイベントを受け取ったことを表す
    （比較・推論設計 7.4。終端判定ではエラーとして扱う）。
    """

    returncode: int | None = None
    start_failed: bool = False
    process_alive: bool = False
    message: str = ""
    protocol_error: bool = False


@dataclass
class TrainingOutcome:
    """確定した試行結果。"""

    experiment_id: str
    attempt: int
    queue_id: str | None
    status: str
    reason: str | None = None
    message: str = ""


@dataclass
class ExperimentDeletionInfo:
    """実験を削除できるかと、削除で消える記録の量。"""

    experiment_id: str
    allowed: bool
    reason: str = ""
    attempts: int = 0
    size_bytes: int | None = None


@dataclass
class CandidateSnapshot:
    """比較候補へ送る時点の学習結果スナップショット。"""

    experiment_id: str
    attempt: int
    selected_epoch: int
    run_id: str
    checkpoint_path: str
    oof_evaluation: dict[str, Any] = field(default_factory=dict)
    experiment_config: dict[str, Any] = field(default_factory=dict)
    weights_size: int | None = None
    weights_sha256: str | None = None
    training_eval_params: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    training_dataset: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactGroup:
    """成果物の整理で扱う、試行ごと・種類ごとのファイルのまとまり。"""

    experiment_id: str
    attempt: int
    category: str
    n_files: int
    size_bytes: int
    deletable: bool
    reason: str = ""
    # 他の種類（または試行の外）とハードリンクで実体を共有し、このまとまりだけ消しても空かない容量
    shared_bytes: int = 0


@dataclass
class PruneResult:
    """成果物の整理の結果。条件を満たさず消さなかったまとまりは skipped に入る。"""

    freed_bytes: int
    n_files: int
    skipped: list[ArtifactGroup] = field(default_factory=list)


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
    usage: str = "train"
    change: str | None = None
    previous_change: str | None = None
    sha256: str = ""
    seed: int = 0

    @property
    def included(self) -> bool:
        """旧モードとの互換用に、版へ収録可能な用途か返す。"""
        return self.usage in {"train", "val"}

    @included.setter
    def included(self, value: bool) -> None:
        """旧呼び出しの採否変更を用途へ変換する。"""
        if not value:
            self.usage = "excluded"
        elif self.usage in {"excluded", "unassigned"}:
            self.usage = "train"

    @property
    def source_folder(self) -> str:
        """取り込み元フォルダの相対パスを返す。"""
        from pathlib import PurePosixPath

        return str(PurePosixPath(self.source_relpath.replace("\\", "/")).parent)


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
    base_train_version: str | None = None
    base_val_version: str | None = None


@dataclass
class AutoTriageSettings:
    """自動振り分けの設定値。"""

    validation_ratio: int = 20
    by_folder: bool = False
    stratify: bool = True
    bad_quality_to_excluded: bool = False
    seed: int = 42


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
    fold: int | None = None


@dataclass
class RunAttempt:
    """学習実行試行。"""

    attempt: int
    started_at: datetime
    finished_at: datetime | None = None
    result: str = "実行中"
    environment: dict[str, Any] = field(default_factory=dict)
    history: list[EpochMetrics] = field(default_factory=list)
    fold_histories: dict[int, list[EpochMetrics]] = field(default_factory=dict)
    oof_history: list[EpochMetrics] = field(default_factory=list)
    final_history: list[EpochMetrics] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    used_item_ids: list[str] = field(default_factory=list)
    fold_assignments: dict[str, int] = field(default_factory=dict)
    selected_epoch: int | None = None
    oof_evaluation: Evaluation | None = None
    oof_predictions: dict[str, float] = field(default_factory=dict)
    total_epochs: int = 0
    phase: str = "cross_validation"

    @property
    def checkpoint_reference(self) -> str:
        return f"試行 {self.attempt}/final.pt"


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
    fold_assignments: dict[str, int] = field(default_factory=dict)
    fold_histories: dict[int, list[EpochMetrics]] = field(default_factory=dict)
    oof_history: list[EpochMetrics] = field(default_factory=list)
    oof_predictions: dict[str, float] = field(default_factory=dict)
    final_history: list[EpochMetrics] = field(default_factory=list)
    selected_epoch: int | None = None
    phase: str = "cross_validation"
    oof_evaluation: Evaluation | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    queue_id: str | None = field(default=None, repr=False, compare=False)
    queue_retry_attempt: int | None = field(default=None, repr=False, compare=False)
    queue_is_retry: bool = field(default=False, repr=False, compare=False)


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
    # 全体の対象画像数（比較・評価設計 9.2。旧データでは 0）
    n_images: int = 0


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
    oof_evaluation: Evaluation | None = None
    oof_experiment_id: str = ""
    oof_epoch: int | None = None
    external_results: list[ExternalResult] = field(default_factory=list)
    external_software: str = ""
    external_date: str = ""
    comment: str = ""
    source_attempt_number: int = 1
    checkpoint_reference: str = ""
    snapshot: CandidateSnapshot | None = None
    # 学習時 OOF AP をこの候補と比べられるか（matching / different / unknown。比較・評価設計 6 章）
    oof_applicability: str = ""
    oof_reason: str = ""
    released_model_id: str | None = None
    # 候補に固定した検証用データセットの版（学習用の版の組。None は組を確認できていない旧候補）
    validation_version: str | None = None
    # validation_version が None のときの理由（評価・リリースできない理由として表示する）
    pairing_issue: str = ""
    # 元の試行の学習用データセットの版
    training_version: str = ""
    # 推論で実際に使う値の全体（比較・評価設計 5.2）
    effective_params: dict[str, Any] = field(default_factory=dict)
    # 採用している評価に対する最新の外部解析の集計（9.4。なければ None）
    external_summary: dict[str, Any] | None = None


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
    oof_evaluation: Evaluation | None = None
    comment: str = ""
    source_attempt_number: int = 1
    model_type: str = ""
    inference_config_id: str = ""
    # 学習時 OOF AP の適用可否（matching / different / unknown。空は記録なし）
    oof_applicability: str = ""
    oof_reason: str = ""
    # リリースに使った評価と、その評価に対する最新の外部解析の集計（なければ None）
    evaluation_id: str = ""
    external_summary: dict[str, Any] | None = None


@dataclass
class EvaluationRecord:
    """比較候補の 1 回分の評価（eval_NNN）の読み取り結果。

    contamination は旧形式（schema 1）の評価にだけある学習混入の検査結果。
    現行形式（schema 2）の評価では空。
    """

    evaluation_id: str
    candidate_id: str
    validation_version: str
    status: str
    evaluation: Evaluation | None = None
    contamination: dict[str, Any] = field(default_factory=dict)
    completed_at: str | None = None
    broken: bool = False
    input_fingerprint: str | None = None
    schema: int | None = None

    @property
    def contamination_found(self) -> bool:
        """旧形式の評価で、学習データと同じ画像が見つかっていたか（リリースに使えない）。"""
        return self.contamination.get("status") == "found"


@dataclass
class EvaluationOutcome:
    """確定した評価 1 件の結果（EvaluationRunner.ended で通知する）。

    evaluation_id は準備に失敗した・開始前に取り消したときは None。
    reason は stopped / failed の理由（user_stop、app_exit、interrupted、error、
    start_failed、prepare_failed、cancelled、conclusion_failed など）。
    """

    candidate_id: str
    evaluation_id: str | None
    status: str
    message: str = ""
    reason: str | None = None
    validation_version: str = ""


@dataclass
class EvaluationProgress:
    """実行中の評価の進み具合（画面表示用。保存しない）。"""

    candidate_id: str
    evaluation_id: str
    validation_version: str
    completed: int = 0
    total: int = 0
    phase: str = "starting"


@dataclass
class RoutingState:
    """振り分けの現在値と、楽観的排他に使う revision。"""

    revision: int
    assignments: dict[str, str | None] = field(default_factory=dict)


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
