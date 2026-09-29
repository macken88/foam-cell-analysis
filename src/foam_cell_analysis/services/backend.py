"""GUI が利用する Backend の型付き契約。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

import numpy as np

from .models import (
    AugmentationProfile,
    Candidate,
    CandidateSnapshot,
    DataItem,
    DatasetVersion,
    Evaluation,
    Experiment,
    ExperimentDeletionInfo,
    ExternalResult,
    ImportCandidate,
    InferenceConfig,
    JobExit,
    PreparedRun,
    ReleasedModel,
    RoutingHistory,
    TrainingOutcome,
    ValidationReport,
    WorkingDataset,
)


def normalization_for_weights(weights: str) -> tuple[list[float], list[float]]:
    """TorchVision の Mask R-CNN 事前学習重みに対応する画像正規化値を返す。"""
    if weights not in {"coco", "imagenet"}:
        raise ValueError(f"未対応の事前学習済み重みです: {weights}")
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Backend(Protocol):
    """GUI が呼び出す全データ操作 API。実装は Qt に依存しない。"""

    def get_working_dataset(self, purpose: str = "train") -> WorkingDataset:
        """用途別の作業中データを返す。"""

    def get_working_items(self) -> list[DataItem]:
        """全用途を含む作業項目を返す。"""

    def update_items(self, item_ids: list[str], **changes: Any) -> None:
        """複数項目をまとめて更新する。"""

    def set_usage(self, item_ids: list[str], usage: str) -> None:
        """複数項目へ同じ用途を設定する。"""

    def scan_import_folders(
        self, paths: dict[str, str], mask_rule: Any = None, channel_rule: Any = None
    ) -> list[ImportCandidate]:
        """複数チャンネルの取り込み元を走査する。"""

    def import_folders(
        self, scans: list[ImportCandidate], uniform_values: dict[str, dict[str, Any]]
    ) -> list[DataItem]:
        """フォルダごとの設定を各取り込み画像へ適用する。"""

    def preview_auto_split(
        self, settings: dict[str, Any], item_ids: list[str] | None = None
    ) -> dict[str, int]:
        """自動振り分けの実行後件数を返す。"""

    def preview_auto_split_assignments(
        self, settings: dict[str, Any], item_ids: list[str] | None = None
    ) -> dict[str, str]:
        """自動振り分けの対象ごとの実行後用途を返す。"""

    def apply_auto_split(
        self, settings: dict[str, Any], item_ids: list[str] | None = None
    ) -> dict[str, int]:
        """自動振り分けを作業データへ適用する。"""

    def validate_items(self, item_ids: list[str] | None = None) -> ValidationReport:
        """全件または指定項目を検査する。"""

    def summarize_finalize(self) -> dict[str, dict[str, int | str]]:
        """学習用・検証用の一括確定内容を集計する。"""

    def finalize_working(
        self, comment: str = "", create_archive: bool = False
    ) -> list[DatasetVersion]:
        """学習・検証の変更版を同時に確定する。"""

    def validate_all_working_items(self) -> ValidationReport:
        """確定対象の全作業項目を自動検査する。"""

    def finalize_working_dataset(
        self, comment: str = "", create_archive: bool = False
    ) -> list[DatasetVersion]:
        """学習・検証の変更版をまとめて確定する。"""

    def apply_auto_triage(
        self, settings: dict[str, Any], item_ids: list[str] | None = None
    ) -> dict[str, int]:
        """未振り分け項目を設定に従って学習・検証へ分ける。"""

    def update_item(self, purpose: str, item_id: str, **changes: Any) -> DataItem:
        """画像の分類、品質、採用状態、選択マスク版を更新する。"""

    def bulk_update_items(self, purpose: str, item_ids: list[str], **changes: Any) -> None:
        """複数画像を一括更新する。"""

    def add_imported_items(self, purpose: str, items: list[DataItem]) -> list[DataItem]:
        """作業データへ画像項目を追加する。"""

    def add_mask_revision(self, purpose: str, item_id: str) -> str:
        """上書きせず新しいマスク版を追加する。"""

    def validate_working_dataset(self, purpose: str) -> ValidationReport:
        """作業データの整合性を確認する。"""

    def next_dataset_version(self, purpose: str) -> str:
        """用途別の次のデータセット版名を返す。"""

    def summarize_working_changes(self, purpose: str) -> dict[str, int | str]:
        """確定ダイアログ用の差分と件数を返す。"""

    def check_dataset_duplicates(
        self, purpose: str, validation_version: str | None = None
    ) -> list[tuple[str, str]]:
        """識別子または画像ハッシュの重複を返す。"""

    def finalize_dataset(
        self, purpose: str, comment: str = "", base_validation_version: str | None = None
    ) -> DatasetVersion:
        """検証済み作業版を確定する。"""

    def list_dataset_versions(self, purpose: str | None = None) -> list[DatasetVersion]:
        """確定済みデータセット版を返す。"""

    def get_dataset_version_items(self, version: str) -> list[DataItem]:
        """指定版に含まれる確定時点の画像一覧を返す。"""

    def list_validation_versions(self) -> list[DatasetVersion]:
        """検証用データセット版を返す。"""

    def record_archive_result(
        self, version: str, output_path: str, ok: bool = True
    ) -> DatasetVersion:
        """アーカイブ作成結果を記録する。"""

    def get_last_saved_at(self, purpose: str = "train") -> datetime:
        """作業データの最終自動保存時刻を返す。"""

    def scan_import_source(
        self, image_dirs: dict[str, str], mask_dir: str
    ) -> list[ImportCandidate]:
        """取り込み元を検索して候補一覧を返す。"""

    def import_items(
        self, purpose: str, candidates: list[ImportCandidate], classification: str | None
    ) -> list[DataItem]:
        """検索候補を作業データとして登録する。"""

    def get_item_image(self, purpose: str, item_id: str, channel: str) -> np.ndarray:
        """画像項目の指定チャンネルを返す。"""

    def get_item_thumbnail(self, item_id: str, size: tuple[int, int]) -> np.ndarray:
        """指定サイズの小さいサムネイル画像を返す。"""

    def get_item_mask(self, purpose: str, item_id: str, revision: str) -> np.ndarray:
        """画像項目の指定マスク版を返す。"""

    def get_dataset_item_image(self, version: str, item_id: str, channel: str) -> np.ndarray:
        """確定済み学習版の画像を返す。"""

    def get_dataset_item_mask(self, version: str, item_id: str, revision: str) -> np.ndarray:
        """確定済み学習版のラベル画像を返す。"""

    def get_candidate_prediction(self, candidate_id: str, item_id: str) -> np.ndarray:
        """候補モデルの予測ラベルを返す。"""

    def get_inference_result(self, filename: str, model_id: str) -> tuple[np.ndarray, np.ndarray]:
        """ファイル名とモデルの合成推論結果を返す。"""

    def list_training_options(self) -> dict[str, list[str]]:
        """学習フォームの選択肢を返す。"""

    def default_experiment_config(self, model_type: str) -> dict[str, Any]:
        """モデル種別の仕様準拠既定設定を返す。"""

    def migrate_experiment_config(self, config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """旧形式の実験設定を現行形式へ移行する。"""

    def validate_experiment_config(self, config: dict[str, Any]) -> list[dict[str, str]]:
        """学習設定のエラーと警告を返す。"""

    def estimate_training_items(
        self,
        dataset_version: str | None = None,
        classification: str = "all",
        quality_filter: str = "all",
        n_folds: int = 5,
        **filters: Any,
    ) -> tuple[int, int]:
        """条件適用後の学習総数と各フォールドの概算検証件数を返す。"""

    def next_experiment_id(self) -> str:
        """次の実験識別子を返す。"""

    def add_training_queue_item(self, config: dict[str, Any]) -> Experiment:
        """設定を queued 状態の実験として保存し、末尾へ追加する。"""

    def add_training_retry_reservation(self, experiment_id: str) -> Experiment:
        """既存実験の次の試行を、記録を変えずにキューへ予約する。"""

    def list_training_queue(self) -> list[Experiment]:
        """順番付きの学習キューを返す。"""

    def update_training_queue_item(self, experiment_id: str, config: dict[str, Any]) -> Experiment:
        """キュー項目の設定を置き換える。"""

    def reorder_training_queue(self, experiment_ids: list[str]) -> list[Experiment]:
        """キュー順を指定順に更新する。"""

    def duplicate_training_queue_items(self, experiment_ids: list[str]) -> list[Experiment]:
        """指定したキュー行を新しい識別子で複製する。"""

    def delete_training_queue_items(self, experiment_ids: list[str]) -> None:
        """指定した待機中のキュー項目を削除する。"""

    def clear_finished_training_queue_items(self) -> None:
        """完了・失敗・中断したキュー項目を片付ける。"""

    def take_next_training_queue_item(self) -> Experiment | None:
        """設定検証を通った待機行の先頭を実行対象にする。"""

    def finish_training_queue_item(self, queue_id: str | None, status: str) -> None:
        """試行予約のキュー表示状態を更新する。"""

    def save_experiment_draft(
        self, config: dict[str, Any], experiment_id: str | None = None
    ) -> Experiment:
        """実験下書きを新規保存または更新する。"""

    def start_training(
        self, config: dict[str, Any], experiment_id: str | None = None
    ) -> Experiment:
        """学習実行試行を追加して実験を開始状態にする。"""

    def prepare_training_run(
        self, experiment_id: str, queue_id: str | None = None, retry: bool = False
    ) -> PreparedRun:
        """設定を固定し、試行を一度だけ準備する。"""

    def record_training_process(
        self, experiment_id: str, attempt: int, pid: int, creation_time: float
    ) -> None:
        """子プロセス識別情報を復旧用ファイルへ保存する。"""

    def fail_training_preparation(self, experiment_id: str) -> None:
        """試行の準備前に失敗した実験を失敗状態にする。"""

    def apply_training_event(self, experiment_id: str, event: dict[str, Any]) -> Experiment:
        """学習イベントを実験履歴へ反映する。"""

    def request_training_stop(self, experiment_id: str, attempt: int, reason: str) -> None:
        """プロセス終了前に中断要求を記録する。"""

    def conclude_training_run(
        self, experiment_id: str, attempt: int, job_exit: JobExit | None = None
    ) -> TrainingOutcome:
        """終了成果物と要求を確認して試行状態を確定する。"""

    def record_epoch(
        self,
        experiment_id: str,
        epoch: int,
        loss: float,
        map_value: float | None = None,
        fold: int | None = None,
    ) -> Experiment:
        """フォールドまたは最終学習のエポック値を記録する。"""

    def finish_training(self, experiment_id: str, status: str = "completed") -> Experiment:
        """実験を完了・失敗・中断状態にする。"""

    def retry_experiment(self, experiment_id: str) -> Experiment:
        """同一実験に実行試行を追加する。"""

    def experiment_deletion_info(
        self, experiment_id: str, *, measure_size: bool = True
    ) -> ExperimentDeletionInfo:
        """実験を削除できるか、理由、試行数、解放される容量を返す。"""

    def delete_experiment(self, experiment_id: str) -> None:
        """実験の記録（設定・全試行・途中保存モデル・ログ）とキュー行を削除する。"""

    def list_experiments(self) -> list[Experiment]:
        """実験一覧を返す。"""

    def get_experiment(self, experiment_id: str) -> Experiment:
        """識別子で実験を返す。"""

    def list_augmentation_profiles(self) -> list[AugmentationProfile]:
        """拡張プロファイル一覧を返す。"""

    def get_augmentation_profile(self, profile_id: str) -> AugmentationProfile:
        """識別子で拡張プロファイルを返す。"""

    def save_augmentation_profile(self, profile: AugmentationProfile) -> AugmentationProfile:
        """呼び出し元から分離した新しい版を保存する。"""

    def list_candidates(self) -> list[Candidate]:
        """比較候補一覧を返す。"""

    def get_candidate(self, candidate_id: str) -> Candidate:
        """識別子で比較候補を返す。"""

    def list_inference_configs(self, model_type: str | None = None) -> list[InferenceConfig]:
        """推論設定一覧を返す。"""

    def create_inference_config(self, model_type: str, params: dict[str, Any]) -> InferenceConfig:
        """新しい推論設定を保存する。"""

    def add_candidate(
        self, experiment_id: str, checkpoint: str, inference_config_id: str, comment: str = ""
    ) -> Candidate:
        """重複確認後に比較候補を追加する。"""

    def add_candidate_from_snapshot(
        self, snapshot: CandidateSnapshot, inference_config_id: str, comment: str = ""
    ) -> Candidate:
        """固定試行の実測結果スナップショットを比較候補へ加える。"""

    def start_evaluation(
        self, candidate_ids: list[str], validation_version: str
    ) -> list[Candidate]:
        """評価対象を評価中状態にする。"""

    def evaluate_candidate(
        self, candidate_id: str, validation_version: str = "val_v003"
    ) -> Evaluation:
        """全体・分類別評価を記録し候補状態に戻す。"""

    def save_external_results(
        self,
        candidate_id: str,
        results: list[ExternalResult | dict[str, Any]],
        software: str = "",
        date: str = "",
        comment: str = "",
    ) -> Candidate:
        """外部解析結果とコメントを保存する。"""

    def reject_candidate(self, candidate_id: str) -> Candidate:
        """候補を非採用にする。"""

    def release_candidate(
        self, candidate_id: str, comment: str = "", validation_version: str = "val_v003"
    ) -> ReleasedModel:
        """評価済み候補をリリース登録する。"""

    def list_validation_items(
        self, validation_version: str = "val_v003", classification: str | None = None
    ) -> list[DataItem]:
        """検証版の画像を分類条件付きで返す。"""

    def list_released_models(self) -> list[ReleasedModel]:
        """リリース済みモデルを返す。"""

    def get_routing(self) -> dict[str, str | None]:
        """現在の分類振り分けを返す。"""

    def apply_routing(self, changes: dict[str, str | None]) -> list[RoutingHistory]:
        """分類振り分けを適用して変更履歴を返す。"""

    def list_routing_history(self) -> list[RoutingHistory]:
        """振り分け変更履歴を返す。"""

    def resolve_model(self, classification: str | None) -> ReleasedModel | None:
        """分類に有効なリリースモデルを返す。"""

    def validate_excel_import(self, filename: str) -> dict[str, Any]:
        """Excel 取込ファイルを検証する。"""

    def preview_excel_import(self, filename: str) -> dict[str, Any]:
        """Excel 取込の検証結果と変更予定を返す。"""

    def apply_excel_import(self, purpose: str, changes: list[dict[str, Any]]) -> None:
        """検証済み Excel 変更を反映する。"""
