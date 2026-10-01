"""モデル比較・リリース・振り分けの保存と復元（比較・評価設計 3〜7・9・13・14 章）。

Qt・torch に依存しない。評価は準備（7.1）・プロセスの記録・停止要求・進捗・終端判定（7.5）・
起動時の復旧（7.6）と、評価フォルダの読み取り（7.7 の採用と破損）を持つ。
評価プロセスの起動そのものは GUI 側（gui/evaluation_runner.py）が行う。
候補は作成時に、元の試行の学習用の版と組になる検証用の版を固定する（3.5）。
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import stat
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from foam_cell_analysis.data.dataset_store import DatasetStore
from foam_cell_analysis.inference.protocol import LEGACY_SCHEMA as EVALUATION_LEGACY_SCHEMA
from foam_cell_analysis.inference.protocol import SCHEMA as EVALUATION_SCHEMA
from foam_cell_analysis.inference.protocol import SCHEMAS as EVALUATION_SCHEMAS
from foam_cell_analysis.jobs.lifecycle import (
    decide_terminal_state,
    process_alive,
    terminate_process,
    write_process_record,
)
from foam_cell_analysis.jobs.protocol import classify_seq
from foam_cell_analysis.services.models import (
    Candidate,
    CandidateSnapshot,
    Evaluation,
    EvaluationOutcome,
    EvaluationProgress,
    EvaluationRecord,
    ExternalResult,
    InferenceConfig,
    JobExit,
    PreparedRun,
    ReleasedModel,
    RoutingHistory,
    RoutingState,
)
from foam_cell_analysis.training.protocol import atomic_write_json, read_json

logger = logging.getLogger(__name__)

SCHEMA = 1
METRIC_ID = "cellpose_ap_iou50_95_image_mean_v1"
PARTICLE_SPLIT_ID = "particle_split_symmetric8_v1"
RELEASE_SPACE_MARGIN = 100 * 1024**2
# release_candidate が公開済みリリースを再利用したときに、返すモデルへ付ける属性名
RECOVERED_RELEASE_ATTR = "recovered_release"
BROKEN_MESSAGE = "評価結果のファイルが壊れています。再評価してください"
# 検証用データセットの組を確認できていない旧候補の、評価・リリースできない理由（既定の文）
UNPAIRED_MESSAGE = (
    "この候補は検証用データセットの組を確認できていないため、評価・リリースできません"
)
CONTAMINATION_FOUND_MESSAGE = (
    "この評価（以前の形式）では検証画像と同じ画像が学習データに見つかっているため、"
    "リリースに使えません。再評価してください"
)

# 外部解析（9.4）: 画像ごとの「検出された全気泡の円相当径の中央値」を手入力する
EXTERNAL_FORMAT = "median_equivalent_diameter_v1"
EXTERNAL_METRIC = "median_equivalent_diameter"
EXTERNAL_UNITS = ("µm", "px")
UNCLASSIFIED = "未分類"

# ユーザーが変更できる推論設定の項目（5.1）: キー → (最小, 最大, 整数か, 画面の名前)
INFERENCE_PARAM_SPECS: dict[str, dict[str, tuple[float, float, bool, str]]] = {
    "mask_rcnn": {
        "box_score_thresh": (0.0, 1.0, False, "検出スコア閾値"),
        "box_nms_thresh": (0.0, 1.0, False, "Box NMS閾値"),
        "box_detections_per_img": (1, 1000, True, "最大検出数"),
    },
    "cellpose": {
        "cellprob_threshold": (-6.0, 6.0, False, "セル確率閾値"),
        "flow_threshold": (0.0, 3.0, False, "フロー閾値"),
    },
}

# 学習時の評価パラメータ（学習 10.2・10.3）。run_spec に記録がない旧形式の試行の補完にだけ使う
LEGACY_TRAINING_EVAL_PARAMS: dict[str, dict[str, Any]] = {
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

_CANDIDATE_ID = re.compile(r"RC-(\d{3,})")
_EVALUATION_ID = re.compile(r"eval_(\d{3,})")
_MODEL_ID = re.compile(r"model_(\d{3,})")
_CONFIG_ID = re.compile(r"infer_v(\d{3,})")
_NAME = re.compile(r"[A-Za-z0-9_.-]+")


# ---- 共通の小道具 ----


def _normalize_numbers(value: Any) -> Any:
    """fingerprint 用に数値を float に揃える（0.5 と 1/2、1 と 1.0 を区別しない）。"""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_numbers(item) for item in value]
    raise ValueError(f"fingerprint に使えない値です: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """5.4 の正規化 JSON を返す。"""
    return json.dumps(
        _normalize_numbers(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """ファイルの sha256 を 1 MiB ずつ読んで求める。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_fingerprint(
    weights_sha256: str,
    model_type: str,
    model_config: dict[str, Any],
    preprocessing: dict[str, Any],
    effective_params: dict[str, Any],
) -> str:
    """候補の fingerprint（5.4）。"""
    return _sha256_text(
        canonical_json(
            {
                "weights_sha256": weights_sha256.lower(),
                "model_type": model_type,
                "model_config": model_config,
                "preprocessing": preprocessing,
                "effective_params": effective_params,
            }
        )
    )


def compute_input_fingerprint(
    candidate_fp: str,
    dataset_sha256: dict[str, str],
    item_ids: Iterable[str],
    metric_id: str = METRIC_ID,
) -> str:
    """評価の input_fingerprint（5.4）。評価の準備（段階 B）でも同じ関数を使う。

    dataset_sha256 は {"manifest.csv": ..., "metadata.csv": ...}（学習の run_spec と同じ形）。
    """
    return _sha256_text(
        canonical_json(
            {
                "candidate_fingerprint": candidate_fp,
                "validation_sha256": dict(dataset_sha256),
                "item_ids": list(item_ids),
                "metric_id": metric_id,
            }
        )
    )


def resolve_recorded_path(base: str | Path, relative: Any) -> Path:
    """記録された相対パスを base の中のパスへ変換する（4.1）。

    絶対パス・..・ドライブ指定・base の外へ出るパスは ValueError にする。
    """
    if not isinstance(relative, str) or not relative:
        raise ValueError("記録されたパスが空です")
    posix_path = PurePosixPath(relative)
    windows_path = PureWindowsPath(relative)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"記録されたパスが不正です: {relative}")
    base_path = Path(base)
    target = base_path / Path(*windows_path.parts)
    try:
        target.resolve().relative_to(base_path.resolve())
    except (OSError, ValueError) as error:
        raise ValueError(f"記録されたパスが基準フォルダの外を指しています: {relative}") from error
    return target


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """allow_nan=False で検査してから原子的に書く。共有違反は 0.1 秒間隔で 5 回まで再試行する。"""
    json.dumps(value, ensure_ascii=False, allow_nan=False)
    for attempt in range(5):
        try:
            atomic_write_json(path, value)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_time(value: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.fromtimestamp(0).astimezone()


def _number(pattern: re.Pattern[str], name: str) -> int | None:
    match = pattern.fullmatch(name)
    return int(match.group(1)) if match else None


def _check_id(pattern: re.Pattern[str], value: Any, label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label}が不正です: {value}")
    return value


def _check_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{label}が不正です: {value}")
    return value


def _evaluation_from_summary(overall: Any, per_class: Any) -> Evaluation | None:
    """result.json・OOF の overall / per_class を GUI の Evaluation へ変換する（9.2）。

    対象なしの AP（null）は None のまま渡す（表示側は None を「—」にする）。
    """
    if not isinstance(overall, dict) or not (per_class is None or isinstance(per_class, dict)):
        return None
    classes: dict[str, tuple[float, int]] = {}
    for name, value in (per_class or {}).items():
        if isinstance(value, dict):
            ap, count = value.get("ap"), value.get("n_images", 0)
        elif isinstance(value, list | tuple) and len(value) == 2:
            ap, count = value
        else:
            continue
        classes[str(name)] = (None if ap is None else float(ap), int(count or 0))
    ap = overall.get("ap")
    return Evaluation(
        None if ap is None else float(ap),
        classes,
        int(overall.get("n_images") or 0),
    )


def _normalize_oof(raw: dict[str, Any]) -> dict[str, Any]:
    """学習の result.json の oof（ap, n_images, per_class）を overall / per_class の形にする。"""
    per_class = {}
    for name, value in (raw.get("per_class") or {}).items():
        if isinstance(value, dict):
            per_class[str(name)] = {
                "ap": value.get("ap"),
                "n_images": int(value.get("n_images") or 0),
            }
        elif isinstance(value, list | tuple) and len(value) == 2:
            per_class[str(name)] = {"ap": value[0], "n_images": int(value[1] or 0)}
    return {
        "overall": {"ap": raw.get("ap"), "n_images": int(raw.get("n_images") or 0)},
        "per_class": per_class,
    }


# ---- 評価フォルダの妥当性（7.5） ----


def _artifact_path(run_dir: Path, result: dict[str, Any], key: str, default: str) -> Path:
    entry = result.get(key)
    relative = entry.get("path") if isinstance(entry, dict) else entry
    return resolve_recorded_path(run_dir, relative or default)


def _valid_ap_entry(entry: Any) -> bool:
    """AP が 0〜1 の実数か null で、n_images が 0 以上の整数か。"""
    if not isinstance(entry, dict):
        return False
    ap = entry.get("ap")
    n_images = entry.get("n_images")
    if type(n_images) is not int or n_images < 0:
        return False
    if ap is None:
        return True
    return (
        isinstance(ap, (int, float))
        and not isinstance(ap, bool)
        and math.isfinite(ap)
        and 0.0 <= ap <= 1.0
    )


def _valid_summary(result: dict[str, Any], target_count: int, schema: int) -> bool:
    """result.json の overall・per_class と、形式ごとの学習混入の項目を確かめる。

    旧形式（schema 1）は contamination が必須、現行形式（schema 2）は持ってはいけない。
    """
    overall = result.get("overall")
    if not _valid_ap_entry(overall) or overall["n_images"] != target_count:
        return False
    per_class = result.get("per_class")
    if not isinstance(per_class, dict) or not all(
        _valid_ap_entry(entry) for entry in per_class.values()
    ):
        return False
    if sum(entry["n_images"] for entry in per_class.values()) != target_count:
        return False
    if schema == EVALUATION_SCHEMA:
        return "contamination" not in result
    contamination = result.get("contamination")
    return isinstance(contamination, dict) and contamination.get("status") in {
        "none",
        "found",
        "unknown",
    }


def validate_evaluation_result(run_dir: str | Path) -> bool:
    """評価の result.json が妥当か（7.5）。sha256 は再計算せず、存在とバイト数だけを見る。

    形式は run_spec の schema（1 は旧形式、2 は現行）で決め、result.json も同じ形式に限る。
    """
    directory = Path(run_dir)
    try:
        spec = read_json(directory / "run_spec.json")
        result = read_json(directory / "result.json")
        schema = spec.get("schema")
        if schema not in EVALUATION_SCHEMAS or result.get("schema") != schema:
            return False
        for key in ("run_id", "evaluation_id", "input_fingerprint"):
            if not spec.get(key) or result.get(key) != spec.get(key):
                return False
        item_ids = (spec.get("validation") or {}).get("item_ids")
        predictions = result.get("predictions")
        if not isinstance(item_ids, list) or not isinstance(predictions, dict):
            return False
        for item_id in item_ids:
            entry = predictions.get(item_id)
            if not isinstance(entry, dict) or type(entry.get("bytes")) is not int:
                return False
            path = resolve_recorded_path(directory, entry.get("path"))
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                return False
        if not _valid_summary(result, len(item_ids), schema):
            return False
        per_image = _artifact_path(directory, result, "per_image_csv", "per_image.csv")
        with per_image.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.reader(stream) if row]
        if len(rows) - 1 != len(item_ids):
            return False
        others = (("eval_counts", "eval_counts.npz"), ("instances_csv", "instances.csv"))
        for key, default in others:
            if not _artifact_path(directory, result, key, default).is_file():
                return False
    except (OSError, ValueError, TypeError, KeyError, csv.Error):
        return False
    return True


def _verify_evaluation_hashes(run_dir: Path) -> bool:
    """予測と集計ファイルの sha256 を記録と照合する（7.7）。"""
    if not validate_evaluation_result(run_dir):
        return False
    try:
        spec = read_json(run_dir / "run_spec.json")
        result = read_json(run_dir / "result.json")
        for item_id in spec["validation"]["item_ids"]:
            entry = result["predictions"][item_id]
            path = resolve_recorded_path(run_dir, entry["path"])
            if file_sha256(path) != str(entry.get("sha256", "")).lower():
                return False
        for key, default in (
            ("per_image_csv", "per_image.csv"),
            ("eval_counts", "eval_counts.npz"),
            ("instances_csv", "instances.csv"),
        ):
            entry = result.get(key)
            expected = entry.get("sha256") if isinstance(entry, dict) else None
            if expected is None:
                return False
            if file_sha256(_artifact_path(run_dir, result, key, default)) != expected.lower():
                return False
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
    return True


_default_process_alive = process_alive
_default_terminate_process = terminate_process


def _within_any(path: Path, dirs: list[Path]) -> bool:
    """path が dirs のどれか（自身を含む）の中にあるか。"""
    return any(path == item or item in path.parents for item in dirs)


def is_recovered_release(model: Any) -> bool:
    """release_candidate の戻り値が、前回の登録で公開済みだったリリースの再利用なら True。"""
    return bool(getattr(model, RECOVERED_RELEASE_ATTR, False))


class DuplicateCandidateError(ValueError):
    """同じ設定（fingerprint）の候補がすでにある。画面はその候補へ案内する。"""

    def __init__(self, candidate_id: str) -> None:
        super().__init__(f"既に {candidate_id} として登録済みです")
        self.candidate_id = candidate_id


# ---- 外部解析（9.4） ----


def is_external_analysis(record: Any) -> bool:
    """現行形式（画像ごとの円相当径の中央値）の外部解析の記録か。旧形式の自由項目は False。"""
    return isinstance(record, dict) and record.get("format") == EXTERNAL_FORMAT


def _external_value(item_id: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{item_id} の値が数値ではありません")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{item_id} の値が有限の数値ではありません")
    if number < 0:
        raise ValueError(f"{item_id} の値が負です")
    return number


def summarize_external_values(
    values: dict[str, float], classifications: dict[str, str | None]
) -> dict[str, Any]:
    """画像ごとの値の平均（全体・分類別）と、入力済み枚数／全枚数を求める。

    classifications は評価対象の全画像（item_id → 分類）。空欄の画像は平均に含めない。
    """

    def mean(numbers: list[float]) -> float | None:
        return sum(numbers) / len(numbers) if numbers else None

    groups: dict[str, list[str]] = {}
    for item_id, name in classifications.items():
        groups.setdefault(name or UNCLASSIFIED, []).append(item_id)
    per_class = {}
    for name in sorted(groups):
        entered = [values[item_id] for item_id in groups[name] if item_id in values]
        per_class[name] = {
            "mean": mean(entered),
            "n_images": len(entered),
            "n_total": len(groups[name]),
        }
    return {
        "mean": mean(list(values.values())),
        "n_images": len(values),
        "n_total": len(classifications),
        "per_class": per_class,
    }


def build_external_analysis(
    values: dict[str, Any],
    classifications: dict[str, str | None],
    *,
    unit: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """入力値を検証し、（保存する値, 集計）を返す。不正なら ValueError（何も保存しない）。

    None・空欄は未入力（値の削除）として扱う。評価対象にない画像・負値・NaN・無限大は拒否する。
    """
    if unit not in EXTERNAL_UNITS:
        raise ValueError("単位は µm または px を選んでください")
    if not isinstance(values, dict):
        raise ValueError("外部解析の値の形が不正です")
    unknown = sorted(str(item_id) for item_id in values if item_id not in classifications)
    if unknown:
        raise ValueError("評価の対象にない画像があります: " + "、".join(unknown[:5]))
    cleaned = {
        item_id: _external_value(item_id, value)
        for item_id, value in sorted(values.items())
        if value is not None and value != ""
    }
    return cleaned, summarize_external_values(cleaned, classifications)


class ComparisonService:
    """推論設定・比較候補・評価の読み取り・外部解析・リリース・振り分けの窓口。"""

    def __init__(
        self,
        workspace_root: str | Path,
        training_service: Any,
        *,
        is_evaluation_active: Callable[[str], bool] | None = None,
        app_version: str | None = None,
        process_alive: Callable[[dict[str, Any]], bool] | None = None,
        process_terminator: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.training_service = training_service
        self.dataset_store = training_service.dataset_store
        self.root = self.workspace_root / "comparison"
        self.configs_root = self.root / "inference_configs"
        self.candidates_root = self.root / "candidates"
        self.releases_root = self.workspace_root / "releases"
        self.is_evaluation_active: Callable[[str], bool] = is_evaluation_active or (
            lambda _candidate_id: False
        )
        self.app_version = app_version or getattr(training_service, "app_version", "0.1.0")
        self.git_commit = getattr(training_service, "git_commit", None)
        self.process_alive = process_alive or _default_process_alive
        self.process_terminator = process_terminator or _default_terminate_process
        # 復旧で終了できなかった評価プロセスの説明。GUI が新しい計算を止めるために使う
        self.recovery_blockers: list[str] = []
        # 実行中の評価の進捗（保存しない）と、評価ごとの最後の seq・記録の形式
        self._progress: dict[str, EvaluationProgress] = {}
        self._last_event_seq: dict[tuple[str, str], int] = {}
        self._event_schema: dict[tuple[str, str], int] = {}
        # 検証用データセットの組を補完できなかった旧候補 → 理由（起動時の移行で決める）
        self.pairing_issues: dict[str, str] = {}
        self.migrated_candidates: list[str] = []

    # ---- 起動時の復旧 ----

    def recover(self) -> list[str]:
        """*.tmp と .preparing_* を消し、旧候補の組を補完し、リリースと候補の食い違いを直す。

        評価フォルダ（evaluations/ 以下）はプロセスの照合が要るので、評価の復旧（7.6）に任せる。
        修復した候補 ID を返す。
        """
        for path in self.configs_root.glob("*.tmp"):
            self._remove(path)
        for path in self.candidates_root.glob(".preparing_*"):
            self._remove(path)
        for path in self.candidates_root.glob("*/*.tmp"):
            self._remove(path)
        for path in self.releases_root.glob("*.tmp"):
            self._remove(path)
        self.migrate_candidate_pairs()
        self.recover_release_deletions()
        return self.recover_releases()

    # ---- 旧候補の検証用データセットの組の補完（比較・評価設計 3.5） ----

    def migrate_candidate_pairs(self) -> list[str]:
        """組（validation_version）の記録がない旧候補に、一度だけ組を補完する。

        元の学習用の版に記録された組の検証用の版を使い、学習用の版が学習時と同じ内容で、
        検証用の版と同じ識別子・同じ画像がないことを確かめてから candidate.json に書く。
        確かめられない候補は書き換えず、理由を pairing_issues に残す（評価・リリース不可）。
        起動時の復旧からだけ呼ぶ（読み取りの途中では書き換えない）。補完した候補 ID を返す。
        """
        migrated: list[str] = []
        self.pairing_issues = {}
        for record in self._candidate_records():
            if record.get("validation_version"):
                continue
            candidate_id = record["candidate_id"]
            try:
                pair = self._verified_legacy_pair(record)
            except ValueError as error:
                self.pairing_issues[candidate_id] = str(error)
                logger.warning(
                    "候補 %s の検証用データセットの組を補完できません: %s", candidate_id, error
                )
                continue
            record["validation_version"] = pair["validation_version"]
            record["dataset_pair"] = pair
            try:
                self._write_candidate(record)
            except OSError as error:
                self.pairing_issues[candidate_id] = "候補の記録を保存できませんでした"
                logger.warning("候補 %s の記録を保存できません: %s", candidate_id, error)
                continue
            migrated.append(candidate_id)
            logger.info("候補 %s の検証用データセットを %s に固定しました", candidate_id, pair)
        self.migrated_candidates = migrated
        return migrated

    def _training_version_of(self, record: dict[str, Any]) -> str:
        """候補の元の試行の学習用の版。記録がなければ ValueError。"""
        pair = record.get("dataset_pair") or {}
        source = record.get("source") or {}
        version = (
            pair.get("training_version")
            or (source.get("training_dataset") or {}).get("version")
            or ((source.get("experiment_config") or {}).get("data") or {}).get("dataset_version")
        )
        if not isinstance(version, str) or not version:
            raise ValueError("元の試行の学習用データセットの記録がありません")
        return _check_name(version, "学習用データセットの版")

    def _verified_legacy_pair(self, record: dict[str, Any]) -> dict[str, Any]:
        """旧候補の組を確かめて返す。確かめられなければ理由付きの ValueError。"""
        source = record.get("source") or {}
        training = self._training_version_of(record)
        validation = self.dataset_store.paired_validation_version(training)
        training_dir = self.workspace_root / "datasets" / training
        recorded = (source.get("training_dataset") or {}).get("sha256")
        if not isinstance(recorded, dict) or not all(
            isinstance(recorded.get(name), str) for name in ("manifest.csv", "metadata.csv")
        ):
            raise ValueError(
                "学習時の学習用データセットの記録（sha256）がないため、組を確認できません"
            )
        for name in ("manifest.csv", "metadata.csv"):
            try:
                same = file_sha256(training_dir / name) == recorded[name].lower()
            except OSError as error:
                raise ValueError(f"学習用データセット {training} を読めません") from error
            if not same:
                raise ValueError(f"学習用データセット {training} が学習時の内容と一致しません")
        try:
            train_rows = DatasetStore._read_csv(training_dir / "manifest.csv")
            val_rows = DatasetStore._read_csv(
                self.workspace_root / "datasets" / validation / "manifest.csv"
            )
        except (OSError, ValueError, csv.Error) as error:
            raise ValueError(
                "データセットの manifest を読めないため、組を確認できません"
            ) from error

        def image_hash(row: dict[str, str]) -> str:
            value = str(row.get("image_sha256") or "").lower()
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("画像の sha256 の記録がない画像があるため、重複を確認できません")
            return value

        train_ids = {item_id.casefold() for item_id in train_rows}
        train_hashes = {image_hash(row) for row in train_rows.values()}
        same_ids = [item_id for item_id in val_rows if item_id.casefold() in train_ids]
        same_images = [
            item_id for item_id, row in val_rows.items() if image_hash(row) in train_hashes
        ]
        if same_ids or same_images:
            raise ValueError(
                f"学習用データセット {training} と検証用データセット {validation} に"
                f"同じ識別子 {len(same_ids)} 件・同じ画像 {len(same_images)} 件があるため、"
                "組を確定できません"
            )
        return {
            "training_version": training,
            "validation_version": validation,
            "method": "migrated",
            "recorded_at": _now(),
            "duplicate_check": {
                "same_item_ids": 0,
                "same_images": 0,
                "training_items": len(train_rows),
                "validation_items": len(val_rows),
            },
        }

    def _pairing_issue(self, record: dict[str, Any]) -> str:
        return self.pairing_issues.get(record.get("candidate_id", ""), "") or UNPAIRED_MESSAGE

    def _fixed_pair(self, record: dict[str, Any]) -> tuple[str, str]:
        """評価開始・新規リリースの前に、候補に固定した組を保存された組の情報と照合する。

        画像の読み直しや重複検査はしない。組を確認できなければ理由付きの ValueError
        （黙って別の検証用の版へ切り替えない）。（学習用の版, 検証用の版）を返す。
        """
        version = record.get("validation_version")
        if not isinstance(version, str) or not version:
            raise ValueError(self._pairing_issue(record))
        training = self._training_version_of(record)
        current = self.dataset_store.paired_validation_version(training)
        if current != version:
            raise ValueError(
                f"学習用データセット {training} の組（{current}）が"
                f"候補の記録（{version}）と一致しません"
            )
        return training, version

    def recover_releases(self) -> list[str]:
        """13.3: releases/.preparing_* を消し、公開済みリリースの候補状態を修復する。"""
        for path in self.releases_root.glob(".preparing_*"):
            self._remove(path)
        repaired = []
        releases = self._release_records()
        candidate_ids = dict.fromkeys(
            (release.get("candidate") or {}).get("candidate_id") for _model_id, release in releases
        )
        for candidate_id in candidate_ids:
            try:
                record = self._read_candidate(candidate_id)
                published = self._published_release(record, releases)
            except (KeyError, ValueError, OSError) as error:
                # 食い違いは自動では直さない（公開済みフォルダも消さない）
                logger.warning("候補 %s のリリースを修復できません: %s", candidate_id, error)
                continue
            if published is not None and self._mark_released(record, published[0]):
                repaired.append(candidate_id)
        return repaired

    @staticmethod
    def _remove(path: Path) -> None:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError as error:
            logger.warning("一時ファイルを削除できません (%s): %s", path, error)

    # ---- 推論設定（5.1） ----

    @staticmethod
    def default_inference_params(model_type: str) -> dict[str, Any]:
        """推論設定の初期値（学習時の評価パラメータと同じ値）を返す。"""
        if model_type not in INFERENCE_PARAM_SPECS:
            raise ValueError(f"未対応のモデル種類です: {model_type}")
        defaults = LEGACY_TRAINING_EVAL_PARAMS[model_type]
        return {key: defaults[key] for key in INFERENCE_PARAM_SPECS[model_type]}

    @classmethod
    def validate_inference_params(cls, model_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """範囲を検査し、型を揃えた params を返す。省いた項目は初期値で補う。"""
        if model_type not in INFERENCE_PARAM_SPECS:
            raise ValueError(f"未対応のモデル種類です: {model_type}")
        specs = INFERENCE_PARAM_SPECS[model_type]
        unknown = sorted(set(params) - set(specs))
        if unknown:
            raise ValueError("変更できない推論設定の項目です: " + "、".join(map(str, unknown)))
        merged = {**cls.default_inference_params(model_type), **params}
        result: dict[str, Any] = {}
        for key, (low, high, integer, label) in specs.items():
            value = merged[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{label}は数値で指定してください")
            if not math.isfinite(value):
                raise ValueError(f"{label}は有限の数値で指定してください")
            if integer:
                if float(value) != int(value):
                    raise ValueError(f"{label}は整数で指定してください")
                value = int(value)
                if not low <= value <= high:
                    raise ValueError(f"{label}は {int(low)}〜{int(high)} の範囲で指定してください")
            else:
                value = float(value)
                if not low <= value <= high:
                    raise ValueError(f"{label}は {low:g}〜{high:g} の範囲で指定してください")
            result[key] = value
        return result

    def list_inference_configs(self, model_type: str | None = None) -> list[InferenceConfig]:
        """保存済みの推論設定を ID 順に返す。読めないファイルは除外してログに残す。"""
        configs = []
        for path in sorted(self.configs_root.glob("infer_v*.json")):
            if _number(_CONFIG_ID, path.stem) is None:
                continue
            try:
                value = read_json(path)
                config = InferenceConfig(
                    _check_id(_CONFIG_ID, value["config_id"], "推論設定 ID"),
                    str(value["model_type"]),
                    dict(value["params"]),
                )
                if config.config_id != path.stem:
                    raise ValueError("ファイル名と config_id が一致しません")
            except (OSError, ValueError, KeyError, TypeError) as error:
                logger.warning("推論設定を読めません (%s): %s", path, error)
                continue
            if model_type is None or config.model_type == model_type:
                configs.append(config)
        return sorted(configs, key=lambda item: _number(_CONFIG_ID, item.config_id) or 0)

    def get_inference_config(self, config_id: str) -> InferenceConfig:
        _check_id(_CONFIG_ID, config_id, "推論設定 ID")
        config = next(
            (item for item in self.list_inference_configs() if item.config_id == config_id), None
        )
        if config is None:
            raise ValueError(f"推論設定がありません: {config_id}")
        return config

    def create_inference_config(self, model_type: str, params: dict[str, Any]) -> InferenceConfig:
        """推論設定を保存する。同じ model_type・同じ params があればそれを返す。"""
        normalized = self.validate_inference_params(model_type, params)
        existing = self.list_inference_configs(model_type)
        for config in existing:
            try:
                if canonical_json(config.params) == canonical_json(normalized):
                    return config
            except ValueError:
                continue
        numbers = [
            _number(_CONFIG_ID, path.stem) or 0 for path in self.configs_root.glob("infer_v*.json")
        ]
        config_id = f"infer_v{max(numbers, default=0) + 1:03d}"
        _write_json(
            self.configs_root / f"{config_id}.json",
            {
                "schema": SCHEMA,
                "config_id": config_id,
                "model_type": model_type,
                "params": normalized,
                "created_at": _now(),
            },
        )
        return InferenceConfig(config_id, model_type, copy.deepcopy(normalized))

    # ---- 候補（5.2〜5.5・6 章） ----

    def _candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates_root / _check_id(_CANDIDATE_ID, candidate_id, "候補 ID")

    def _read_candidate(self, candidate_id: Any) -> dict[str, Any]:
        path = self._candidate_dir(candidate_id) / "candidate.json"
        if not path.is_file():
            raise KeyError(candidate_id)
        record = read_json(path)
        if record.get("candidate_id") != candidate_id:
            raise ValueError(f"candidate.json の候補 ID が一致しません: {candidate_id}")
        return record

    def _write_candidate(self, record: dict[str, Any]) -> None:
        _write_json(self._candidate_dir(record["candidate_id"]) / "candidate.json", record)

    def _candidate_records(self) -> list[dict[str, Any]]:
        records = []
        if not self.candidates_root.is_dir():
            return records
        for folder in self.candidates_root.iterdir():
            number = _number(_CANDIDATE_ID, folder.name)
            if number is None or not folder.is_dir():
                continue
            try:
                records.append(self._read_candidate(folder.name))
            except (OSError, ValueError, KeyError) as error:
                logger.warning("比較候補を読めません (%s): %s", folder, error)
        return sorted(records, key=lambda item: _number(_CANDIDATE_ID, item["candidate_id"]))

    def _run_dir(self, source: dict[str, Any]) -> Path:
        experiment_id = _check_name(source.get("experiment_id"), "実験 ID")
        attempt = source.get("attempt")
        if type(attempt) is not int or attempt < 1:
            raise ValueError(f"試行番号が不正です: {attempt}")
        runs = self.workspace_root / "experiments" / experiment_id / "runs"
        return runs / f"attempt_{attempt:03d}"

    def _weights_path(self, source: dict[str, Any]) -> Path:
        relative = (source.get("weights") or {}).get("path")
        return resolve_recorded_path(self._run_dir(source), relative)

    def add_candidate(
        self, experiment_id: str, attempt: int, inference_config_id: str, comment: str = ""
    ) -> Candidate:
        """試行を明示して比較候補を作る（5.3）。"""
        config = self.get_inference_config(inference_config_id)
        snapshot: CandidateSnapshot = self.training_service.create_candidate_snapshot(
            experiment_id, attempt=attempt
        )
        experiment_config = copy.deepcopy(snapshot.experiment_config or {})
        model_config = copy.deepcopy(experiment_config.get("model") or {})
        model_type = model_config.get("type")
        if config.model_type != model_type:
            raise ValueError("実験のモデル種類に合う推論設定を選択してください")
        if (
            not isinstance(snapshot.weights_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", snapshot.weights_sha256) is None
            or type(snapshot.weights_size) is not int
        ):
            raise ValueError("最終学習モデルの記録（大きさ・sha256）がありません")
        source = {
            "experiment_id": snapshot.experiment_id,
            "attempt": snapshot.attempt,
            "run_id": snapshot.run_id,
            "selected_epoch": snapshot.selected_epoch,
            "model_type": model_type,
            "weights": {
                "path": snapshot.checkpoint_path,
                "size": snapshot.weights_size,
                "sha256": snapshot.weights_sha256.lower(),
            },
            "experiment_config": experiment_config,
            "preprocessing": copy.deepcopy(snapshot.preprocessing or {}),
            "training_dataset": copy.deepcopy(snapshot.training_dataset or {}),
            "training_eval_params": copy.deepcopy(snapshot.training_eval_params or {}),
            "training_eval_params_filled": False,
        }
        weights = self._weights_path(source)
        if not weights.is_file() or weights.stat().st_size != snapshot.weights_size:
            raise ValueError("最終学習モデルのファイルが記録と一致しません")
        # 元の試行の学習用の版と組になる検証用の版を、候補に固定する（3.5）
        training_version = (source["training_dataset"] or {}).get("version")
        if not isinstance(training_version, str) or not training_version:
            raise ValueError("元の試行の学習用データセットの記録がありません")
        validation_version = self.dataset_store.paired_validation_version(training_version)
        training_params = source["training_eval_params"]
        if not training_params:
            # 旧形式の試行: 学習 10.2・10.3 の既知の値で補い、補ったことを記録する（5.2）
            training_params = copy.deepcopy(LEGACY_TRAINING_EVAL_PARAMS[model_type])
            source["training_eval_params"] = training_params
            source["training_eval_params_filled"] = True
        effective = {**copy.deepcopy(training_params), **copy.deepcopy(config.params)}
        fingerprint = candidate_fingerprint(
            source["weights"]["sha256"],
            model_type,
            model_config,
            source["preprocessing"],
            effective,
        )
        for other in self._candidate_records():
            if other.get("status") != "rejected" and other.get("fingerprint") == fingerprint:
                raise DuplicateCandidateError(other["candidate_id"])
        if source["training_eval_params_filled"]:
            applicability, reason = "unknown", "学習時の評価条件を確認できません"
            source_effective = None
        elif canonical_json(effective) == canonical_json(training_params):
            applicability, reason = "matching", None
            source_effective = copy.deepcopy(training_params)
        else:
            applicability, reason = "different", "推論設定が学習時と異なります"
            source_effective = copy.deepcopy(training_params)
        record = {
            "schema": SCHEMA,
            "candidate_id": "",
            "created_at": _now(),
            "comment": comment,
            "status": "candidate",
            "inference_config_id": config.config_id,
            "effective_params": effective,
            "validation_version": validation_version,
            "dataset_pair": {
                "training_version": training_version,
                "validation_version": validation_version,
                "method": "created",
                "recorded_at": _now(),
            },
            "source": source,
            "oof": {
                "source_run_id": snapshot.run_id,
                "selected_epoch": snapshot.selected_epoch,
                "evaluation": _normalize_oof(snapshot.oof_evaluation or {}),
                "source_effective_params": source_effective,
                "applicability": applicability,
                "reason": reason,
            },
            "fingerprint": fingerprint,
        }
        self.candidates_root.mkdir(parents=True, exist_ok=True)
        numbers = [
            _number(_CANDIDATE_ID, path.name) or 0 for path in self.candidates_root.glob("RC-*")
        ]
        candidate_id = f"RC-{max(numbers, default=0) + 1:03d}"
        record["candidate_id"] = candidate_id
        preparing = self.candidates_root / f".preparing_{uuid.uuid4()}"
        preparing.mkdir()
        _write_json(preparing / "candidate.json", record)
        os.replace(preparing, self.candidates_root / candidate_id)
        return self._to_candidate(record)

    def list_candidates(self) -> list[Candidate]:
        return [self._to_candidate(record) for record in self._candidate_records()]

    def get_candidate(self, candidate_id: str) -> Candidate:
        return self._to_candidate(self._read_candidate(candidate_id))

    def get_candidate_record(self, candidate_id: str) -> dict[str, Any]:
        """candidate.json の内容（コピー）を返す。"""
        return copy.deepcopy(self._read_candidate(candidate_id))

    def reject_candidate(
        self,
        candidate_id: str,
        *,
        is_evaluation_active: Callable[[str], bool] | None = None,
    ) -> Candidate:
        """候補状態の候補を非採用にする。評価中・評価待ちの候補は非採用にできない。"""
        record = self._read_candidate(candidate_id)
        if record.get("status") != "candidate":
            raise ValueError("候補状態のモデルだけ非採用にできます")
        active = is_evaluation_active or self.is_evaluation_active
        if active(candidate_id):
            raise ValueError("評価中または評価待ちの候補は非採用にできません")
        record["status"] = "rejected"
        self._write_candidate(record)
        return self._to_candidate(record)

    def restore_candidate(self, candidate_id: str) -> Candidate:
        """非採用候補を未決定に戻す。"""
        record = self._read_candidate(candidate_id)
        if record.get("status") != "rejected":
            raise ValueError("この候補は一覧へ戻せません")
        source = record.get("source") or {}
        weights = source.get("weights") or {}
        source_path = self._weights_path(source)
        if (
            not source_path.is_file()
            or source_path.stat().st_size != weights.get("size")
            or file_sha256(source_path) != weights.get("sha256")
        ):
            raise ValueError("元の学習モデルが見つからないか、記録と一致しません")
        for other in self._candidate_records():
            if (
                other.get("candidate_id") != candidate_id
                and other.get("status") != "rejected"
                and other.get("fingerprint") == record.get("fingerprint")
            ):
                raise DuplicateCandidateError(other["candidate_id"])
        record["status"] = "candidate"
        self._write_candidate(record)
        return self._to_candidate(record)

    def _to_candidate(self, record: dict[str, Any]) -> Candidate:
        """candidate.json を GUI の Candidate へ変換する。"""
        source = record.get("source") or {}
        oof = record.get("oof") or {}
        oof_eval = oof.get("evaluation") or {}
        evaluations = {}
        version = record.get("validation_version")
        version = version if isinstance(version, str) and version else None
        adopted = self._adopted(record, version) if version else None
        if adopted is not None and not adopted.broken and adopted.evaluation is not None:
            evaluations[version] = adopted.evaluation
        externals = self._read_external(record["candidate_id"])
        legacy = [item for item in externals if not is_external_analysis(item)]
        latest = legacy[-1] if legacy else {}
        summary = None
        if adopted is not None and not adopted.broken:
            analysis = self._latest_external_analysis(externals, adopted.evaluation_id)
            summary = self._external_summary(analysis)
        try:
            training_version = self._training_version_of(record)
        except ValueError:
            training_version = ""
        attempt = int(source.get("attempt") or 1)
        weights = source.get("weights") or {}
        snapshot = CandidateSnapshot(
            experiment_id=str(source.get("experiment_id", "")),
            attempt=attempt,
            selected_epoch=int(source.get("selected_epoch") or 0),
            run_id=str(source.get("run_id", "")),
            checkpoint_path=str(weights.get("path", "")),
            oof_evaluation=copy.deepcopy(oof_eval),
            experiment_config=copy.deepcopy(source.get("experiment_config") or {}),
            weights_size=weights.get("size"),
            weights_sha256=weights.get("sha256"),
            training_eval_params=copy.deepcopy(source.get("training_eval_params") or {}),
            preprocessing=copy.deepcopy(source.get("preprocessing") or {}),
            training_dataset=copy.deepcopy(source.get("training_dataset") or {}),
        )
        return Candidate(
            candidate_id=record["candidate_id"],
            experiment_id=str(source.get("experiment_id", "")),
            checkpoint="final.pt",
            inference_config_id=str(record.get("inference_config_id", "")),
            status=str(record.get("status", "candidate")),
            evaluations=evaluations,
            oof_evaluation=_evaluation_from_summary(
                oof_eval.get("overall"), oof_eval.get("per_class")
            ),
            oof_experiment_id=str(source.get("experiment_id", "")),
            oof_epoch=oof.get("selected_epoch"),
            external_results=[
                ExternalResult(
                    str(item.get("name", "")), item.get("value", ""), str(item.get("unit", ""))
                )
                for item in latest.get("results", [])
            ],
            external_software=str(latest.get("software", "")),
            external_date=str(latest.get("analyzed_on", "")),
            comment=str(record.get("comment", "")),
            source_attempt_number=attempt,
            checkpoint_reference=f"試行 {attempt}/final.pt",
            snapshot=snapshot,
            oof_applicability=str(oof.get("applicability") or ""),
            oof_reason=str(oof.get("reason") or ""),
            released_model_id=record.get("released_model_id"),
            validation_version=version,
            pairing_issue="" if version else self._pairing_issue(record),
            training_version=training_version,
            effective_params=copy.deepcopy(record.get("effective_params") or {}),
            external_summary=summary,
        )

    # ---- 評価の読み取り（7.5・7.7） ----

    def _evaluations_root(self, candidate_id: str) -> Path:
        return self._candidate_dir(candidate_id) / "evaluations"

    def _evaluated_versions(self, candidate_id: str) -> list[str]:
        root = self._evaluations_root(candidate_id)
        if not root.is_dir():
            return []
        return sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and _NAME.fullmatch(path.name) and not path.name.startswith(".")
        )

    def _evaluation_dirs(self, candidate_id: str, version: str) -> list[tuple[str, Path]]:
        root = self._evaluations_root(candidate_id) / _check_name(version, "検証版")
        if not root.is_dir():
            return []
        found = [
            (path.name, path)
            for path in root.iterdir()
            if path.is_dir() and _number(_EVALUATION_ID, path.name) is not None
        ]
        return sorted(found, key=lambda item: _number(_EVALUATION_ID, item[0]))

    def _locate_evaluation(self, candidate_id: str, evaluation_id: str) -> tuple[str, Path]:
        """evaluation_id から（検証版, フォルダ）を探す。版をまたいで重複すれば ValueError。"""
        _check_id(_EVALUATION_ID, evaluation_id, "評価 ID")
        matches = [
            (version, self._evaluations_root(candidate_id) / version / evaluation_id)
            for version in self._evaluated_versions(candidate_id)
            if (self._evaluations_root(candidate_id) / version / evaluation_id).is_dir()
        ]
        if not matches:
            raise ValueError(f"評価がありません: {candidate_id}/{evaluation_id}")
        if len(matches) > 1:
            raise ValueError(f"評価 ID が複数の検証版にあります: {candidate_id}/{evaluation_id}")
        return matches[0]

    def evaluation_run_dir(self, candidate_id: str, evaluation_id: str) -> Path:
        """評価フォルダ（candidates/<候補>/evaluations/<検証版>/<評価>）を返す。

        evaluation_id は候補ごとの通し番号なので、検証版を指定しなくても 1 つに決まる。
        見つからなければ ValueError。
        """
        _version, run_dir = self._locate_evaluation(candidate_id, evaluation_id)
        return run_dir

    def _dataset_state(self, version: str) -> tuple[dict[str, str], list[str]]:
        """検証版の manifest・metadata の sha256 と、評価対象の item_id 一覧（3.2）を返す。"""
        folder = self.workspace_root / "datasets" / _check_name(version, "検証版")
        items = self.dataset_store.select_evaluation_items(version)
        sha = {name: file_sha256(folder / name) for name in ("manifest.csv", "metadata.csv")}
        return sha, [item.item_id for item in items]

    def evaluation_input_fingerprint(self, candidate_id: str, validation_version: str) -> str:
        """今の候補・検証版から評価の input_fingerprint を求める（5.4）。"""
        record = self._read_candidate(candidate_id)
        sha, item_ids = self._dataset_state(validation_version)
        return compute_input_fingerprint(record["fingerprint"], sha, item_ids)

    def _read_record(self, candidate_id: str, version: str, run_dir: Path) -> EvaluationRecord:
        """1 回分の評価フォルダを読む。completed ならサイズの照合で破損を判定する。"""
        record = EvaluationRecord(run_dir.name, candidate_id, version, "running")
        try:
            spec = read_json(run_dir / "run_spec.json")
            record.input_fingerprint = spec.get("input_fingerprint")
            schema = spec.get("schema")
            record.schema = schema if schema in EVALUATION_SCHEMAS else None
        except (OSError, ValueError):
            record.status = "failed"
            record.broken = True
            return record
        status_path = run_dir / "status.json"
        if status_path.is_file():
            try:
                record.status = str(read_json(status_path).get("status") or "failed")
            except (OSError, ValueError):
                record.status = "failed"
        if record.status != "completed":
            return record
        try:
            result = read_json(run_dir / "result.json")
            record.evaluation = _evaluation_from_summary(
                result.get("overall"), result.get("per_class")
            )
            # 学習混入の検査結果は旧形式の評価にだけある（現行形式では空のまま）
            contamination = result.get("contamination")
            if record.schema == EVALUATION_LEGACY_SCHEMA and isinstance(contamination, dict):
                record.contamination = dict(contamination)
            record.completed_at = result.get("completed_at")
        except (OSError, ValueError, TypeError, AttributeError):
            record.broken = True
            return record
        record.broken = record.evaluation is None or not validate_evaluation_result(run_dir)
        return record

    def list_candidate_evaluations(
        self, candidate_id: str, validation_version: str | None = None
    ) -> list[EvaluationRecord]:
        """その候補・検証版の全評価を番号順に返す（状態を問わない）。

        validation_version を省くと、全ての検証版の評価（固定した版と異なる版の旧評価を含む
        履歴）を評価番号順に返す。
        """
        self._read_candidate(candidate_id)
        versions = (
            [validation_version]
            if validation_version is not None
            else self._evaluated_versions(candidate_id)
        )
        records = [
            self._read_record(candidate_id, version, path)
            for version in versions
            for _evaluation_id, path in self._evaluation_dirs(candidate_id, version)
        ]
        return sorted(records, key=lambda item: _number(_EVALUATION_ID, item.evaluation_id) or 0)

    def _adopted(
        self,
        record: dict[str, Any],
        version: str,
        cache: dict[str, tuple[dict[str, str], list[str]] | None] | None = None,
    ) -> EvaluationRecord | None:
        candidate_id = record["candidate_id"]
        directories = self._evaluation_dirs(candidate_id, version)
        if not directories:
            return None
        cache = {} if cache is None else cache
        if version not in cache:
            try:
                cache[version] = self._dataset_state(version)
            except (OSError, ValueError, KeyError) as error:
                logger.warning("検証版 %s を読めません: %s", version, error)
                cache[version] = None
        state = cache[version]
        if state is None:
            return None
        current = compute_input_fingerprint(record["fingerprint"], state[0], state[1])
        for _evaluation_id, path in reversed(directories):
            item = self._read_record(candidate_id, version, path)
            if item.status == "completed" and item.input_fingerprint == current:
                return item
        return None

    def get_candidate_evaluation(
        self, candidate_id: str, validation_version: str | None = None
    ) -> EvaluationRecord | None:
        """採用する評価（7.7）を返す。破損していれば broken=True のまま返す。

        validation_version を省くと候補に固定した検証用の版を使う（組が未確認なら None）。
        """
        record = self._read_candidate(candidate_id)
        version = validation_version or record.get("validation_version")
        if not isinstance(version, str) or not version:
            return None
        return self._adopted(record, version)

    def verify_evaluation(self, candidate_id: str, evaluation_id: str) -> bool:
        """評価が completed で、予測と集計ファイルの sha256 が記録と一致するか。"""
        _version, run_dir = self._locate_evaluation(candidate_id, evaluation_id)
        try:
            if read_json(run_dir / "status.json").get("status") != "completed":
                return False
        except (OSError, ValueError):
            return False
        return _verify_evaluation_hashes(run_dir)

    def get_candidate_prediction(
        self, candidate_id: str, evaluation_id: str, item_id: str
    ) -> np.ndarray:
        """評価の予測ラベルを読み、バイト数と sha256 を照合して返す（11 章）。"""
        _version, run_dir = self._locate_evaluation(candidate_id, evaluation_id)
        try:
            result = read_json(run_dir / "result.json")
            entry = result["predictions"][item_id]
            path = resolve_recorded_path(run_dir, entry["path"])
            data = path.read_bytes()
        except (OSError, KeyError, TypeError) as error:
            raise ValueError(BROKEN_MESSAGE) from error
        if (
            len(data) != entry.get("bytes")
            or hashlib.sha256(data).hexdigest() != str(entry.get("sha256", "")).lower()
        ):
            raise ValueError(BROKEN_MESSAGE)
        if path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile

            array = np.asarray(tifffile.imread(io.BytesIO(data)))
        else:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as image:
                array = np.asarray(image).copy()
        shape = entry.get("shape")
        if shape is not None and list(array.shape) != list(shape):
            raise ValueError(BROKEN_MESSAGE)
        return array

    # ---- 評価の準備・記録・終端判定・復旧（7.1〜7.6） ----

    def _next_evaluation_number(self, candidate_id: str) -> int:
        """候補ごとの通し番号（検証版をまたいで同じ番号を使わない）。"""
        root = self._evaluations_root(candidate_id)
        numbers = [
            _number(_EVALUATION_ID, path.name) or 0
            for path in root.glob("*/eval_*")
            if path.is_dir() and not path.parent.name.startswith(".")
        ]
        return max(numbers, default=0) + 1

    def _process_env(self) -> dict[str, str]:
        """評価プロセスの環境変数。TORCH_HOME などは学習と同じ場所に固定する（7.2）。"""
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "TORCH_HOME": str(self.workspace_root / "pretrained" / "torch"),
                "CELLPOSE_LOCAL_MODELS_PATH": str(self.workspace_root / "pretrained" / "cellpose"),
            }
        )
        return env

    def prepare_evaluation_run(self, candidate_id: str) -> PreparedRun:
        """評価の唯一の作成口（7.1）。run_spec を書いてから eval_NNN へ改名する。

        評価先は候補に固定した検証用の版だけ（利用者は選ばない）。開始前に、保存された組の
        情報と候補の固定値を照合する。再評価は常に新しい eval_NNN を作り、過去の評価は
        変更・削除しない。
        """
        from foam_cell_analysis.evaluation.ap import METRIC
        from foam_cell_analysis.inference.protocol import evaluation_run_id, validate_run_spec

        try:
            record = self._read_candidate(candidate_id)
        except KeyError as error:
            raise ValueError(f"比較候補がありません: {candidate_id}") from error
        if record.get("status") not in {"candidate", "released"}:
            raise ValueError("候補状態または公開済みのモデルだけ評価できます")
        training_version, version = self._fixed_pair(record)
        version = _check_name(version, "検証版")
        if version not in self.dataset_store.list_versions("val"):
            raise ValueError(f"検証用データセット {version} がありません")
        items = self.dataset_store.select_evaluation_items(version)
        if not items:
            raise ValueError(f"検証用データセット {version} に画像がありません")
        source = record["source"]
        experiment_config = source.get("experiment_config") or {}
        channels = list((experiment_config.get("data") or {}).get("input_channels") or [])
        for channel in channels:
            if any(channel not in item.channels for item in items):
                raise ValueError(
                    "この検証用データセットには、モデルが使うチャンネル "
                    f"{channel} がない画像があります"
                )
        folder = self.workspace_root / "datasets" / version
        dataset_sha = {
            name: file_sha256(folder / name) for name in ("manifest.csv", "metadata.csv")
        }
        item_ids = [item.item_id for item in items]
        input_fingerprint = compute_input_fingerprint(record["fingerprint"], dataset_sha, item_ids)
        weights = source["weights"]
        weights_path = self._weights_path(source)
        if record.get("status") == "released":
            lifecycle = self._release_lifecycle(record.get("released_model_id") or "")
            if lifecycle.get("status") in {"deleted", "deleting"}:
                raise ValueError("公開モデルの重みは削除済みのため再評価できません")
            published = self._published_release(record)
            if published is None or published[0] != record.get("released_model_id"):
                raise ValueError("公開済みモデルの参照を確認できないため再評価できません")
            release = self.get_release_record(published[0])
            release_weights = release.get("weights") or {}
            if release_weights.get("sha256") != weights.get("sha256") or release_weights.get(
                "size"
            ) != weights.get("size"):
                raise ValueError("公開モデルの重みが候補の元重みと一致しません")
            weights_path = self._ensure_release_path_safe(
                self.releases_root / published[0] / "model.pt", check_ancestors=False
            )
        if not weights_path.is_file() or weights_path.stat().st_size != weights["size"]:
            raise ValueError("評価に使うモデル重みが見つからないか、サイズが一致しません")
        weights_relative = (
            weights_path.resolve().relative_to(self.workspace_root.resolve()).as_posix()
        )
        root = self._evaluations_root(candidate_id) / version
        root.mkdir(parents=True, exist_ok=True)
        number = self._next_evaluation_number(candidate_id)
        evaluation_id = f"eval_{number:03d}"
        run_id = evaluation_run_id(candidate_id, version, evaluation_id)
        spec = {
            "schema": EVALUATION_SCHEMA,
            "protocol": 1,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "evaluation_id": evaluation_id,
            "created_at": _now(),
            "weights": {
                "path": weights_relative,
                "size": weights["size"],
                "sha256": str(weights["sha256"]).lower(),
            },
            "model_type": source.get("model_type"),
            "model_config": copy.deepcopy(experiment_config.get("model") or {}),
            "preprocessing": copy.deepcopy(source.get("preprocessing") or {}),
            "effective_params": copy.deepcopy(record.get("effective_params") or {}),
            "validation": {
                "version": version,
                "path": f"datasets/{version}",
                "sha256": dataset_sha,
                "item_ids": item_ids,
            },
            "dataset_pair": {
                "training_version": training_version,
                "validation_version": version,
            },
            "metric": copy.deepcopy(METRIC),
            "device_request": "auto",
            "app_version": self.app_version,
            "git_commit": self.git_commit,
            "input_fingerprint": input_fingerprint,
        }
        final_dir = root / evaluation_id
        validate_run_spec(spec, final_dir)
        preparing = root / f".preparing_{uuid.uuid4()}"
        preparing.mkdir()
        try:
            _write_json(preparing / "run_spec.json", spec)
            # ここで初めて評価として数える
            os.replace(preparing, final_dir)
        except BaseException:
            shutil.rmtree(preparing, ignore_errors=True)
            raise
        run_dir = str(final_dir.resolve())
        return PreparedRun(
            run_id,
            run_dir,
            str(Path(sys.executable).resolve()),
            ["-m", "foam_cell_analysis.inference.run", "--run-dir", run_dir],
            self._process_env(),
        )

    def record_evaluation_process(
        self, candidate_id: str, evaluation_id: str, pid: int, creation_time: float
    ) -> None:
        """評価プロセスの PID と作成時刻を process.json に保存する（復旧時の照合用）。"""
        _version, run_dir = self._locate_evaluation(candidate_id, evaluation_id)
        write_process_record(run_dir, pid, creation_time)

    def request_evaluation_stop(self, candidate_id: str, evaluation_id: str, reason: str) -> None:
        """stop_request.json を保存する。親は kill の前に呼ぶ。"""
        if reason not in {"user_stop", "app_exit"}:
            raise ValueError("中断理由は user_stop または app_exit です")
        _version, run_dir = self._locate_evaluation(candidate_id, evaluation_id)
        _write_json(run_dir / "stop_request.json", {"reason": reason, "requested_at": _now()})

    @staticmethod
    def _apply_progress(progress: EvaluationProgress, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "started":
            progress.phase = "preflight"
        elif kind == "preflight":
            progress.phase = "inference"
            progress.total = int(event["n_images"])
        elif kind == "image_done":
            progress.phase = "inference"
            progress.completed = int(event["completed"])
            progress.total = int(event["total"])
        elif kind == "completed":
            progress.phase = "completed"
        elif kind == "error":
            progress.phase = "error"

    def apply_evaluation_event(
        self, candidate_id: str, evaluation_id: str, event: dict[str, Any]
    ) -> EvaluationProgress | None:
        """評価イベントを検証して進捗へ反映する。成功は確定しない（7.4・7.5）。

        必須項目の欠けたイベントは ValueError（呼び出し側がプロトコルエラーとして扱う）。
        seq の重複は無視し、欠落は events.jsonl から再生して埋める。
        """
        from foam_cell_analysis.inference.protocol import parse_run_id, read_events, validate_event

        if isinstance(event, dict) and event.get("type") == "hello":
            validate_event(event, schema=EVALUATION_SCHEMA, allow_hello=True)
            return self.get_evaluation_progress(candidate_id)
        event_candidate, version, event_evaluation = parse_run_id(
            event.get("run_id") if isinstance(event, dict) else None
        )
        if event_candidate != candidate_id or event_evaluation != evaluation_id:
            raise ValueError("イベントの run_id が評価に対応していません")
        schema = self._evaluation_schema(candidate_id, evaluation_id)
        validate_event(event, schema=schema, allow_hello=False)
        key = (candidate_id, evaluation_id)
        progress = self._progress.get(candidate_id)
        if progress is None or progress.evaluation_id != evaluation_id:
            progress = EvaluationProgress(candidate_id, evaluation_id, version)
            self._progress[candidate_id] = progress
        last_seq = self._last_event_seq.get(key, 0)
        state = classify_seq(last_seq, event["seq"])
        if state == "duplicate":
            return copy.copy(progress)
        if state == "gap":
            _version, run_dir = self._locate_evaluation(candidate_id, evaluation_id)
            events = read_events(
                run_dir / "events.jsonl", schema=schema, expected_run_id=event["run_id"]
            )
            progress = EvaluationProgress(candidate_id, evaluation_id, version)
            self._progress[candidate_id] = progress
            for recorded in events:
                self._apply_progress(progress, recorded)
            last_seq = events[-1]["seq"] if events else 0
            self._last_event_seq[key] = last_seq
            if event["seq"] <= last_seq:
                return copy.copy(progress)
            if event["seq"] != last_seq + 1:
                raise ValueError("イベント seq の欠落を events.jsonl から復元できません")
        self._apply_progress(progress, event)
        self._last_event_seq[key] = event["seq"]
        return copy.copy(progress)

    def _evaluation_schema(self, candidate_id: str, evaluation_id: str) -> int:
        """評価の記録の形式（run_spec の schema）。イベントの検証に使う。"""
        key = (candidate_id, evaluation_id)
        if key not in self._event_schema:
            _version, run_dir = self._locate_evaluation(candidate_id, evaluation_id)
            try:
                schema = read_json(run_dir / "run_spec.json").get("schema")
            except (OSError, ValueError) as error:
                raise ValueError("評価の run_spec を読めません") from error
            if schema not in EVALUATION_SCHEMAS:
                raise ValueError("評価の run_spec の形式が不正です")
            self._event_schema[key] = schema
        return self._event_schema[key]

    def get_evaluation_progress(self, candidate_id: str) -> EvaluationProgress | None:
        """実行中の評価の進捗（completed / total）を返す。なければ None。"""
        progress = self._progress.get(candidate_id)
        return copy.copy(progress) if progress is not None else None

    def _conclude_evaluation(
        self, candidate_id: str, version: str, run_dir: Path, job_exit: JobExit | None
    ) -> EvaluationOutcome:
        """7.5: 共通の優先順位で終端状態を決め、未保存なら status.json を書く。"""
        status_path = run_dir / "status.json"
        stop_path = run_dir / "stop_request.json"
        error_path = run_dir / "error.json"
        process_path = run_dir / "process.json"
        protocol_error = bool(job_exit and job_exit.protocol_error)
        decision = decide_terminal_state(
            existing_status=read_json(status_path) if status_path.exists() else None,
            stop_request=read_json(stop_path) if stop_path.exists() else None,
            result_valid=lambda: validate_evaluation_result(run_dir),
            error_present=lambda: error_path.exists() or protocol_error,
            start_failed=bool(job_exit and job_exit.start_failed),
            process_alive=lambda: (
                bool(job_exit and job_exit.process_alive)
                or self.process_alive(read_json(process_path) if process_path.exists() else {})
            ),
        )
        evaluation_id = run_dir.name
        if decision is None:
            return EvaluationOutcome(
                candidate_id, evaluation_id, "running", validation_version=version
            )
        message = decision.message
        if decision.source == "start_failed" or (
            decision.source == "error" and not error_path.exists()
        ):
            message = job_exit.message if job_exit else ""
        elif decision.source == "error":
            try:
                message = str(read_json(error_path).get("message", ""))
            except (OSError, ValueError):
                message = "評価プロセスのエラー記録を読めません"
        outcome = EvaluationOutcome(
            candidate_id,
            evaluation_id,
            decision.status,
            message,
            decision.reason,
            version,
        )
        if decision.source != "existing":
            _write_json(
                status_path,
                {
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "message": outcome.message,
                    "concluded_at": _now(),
                    "returncode": job_exit.returncode if job_exit else None,
                },
            )
        return outcome

    def conclude_evaluation_run(
        self, candidate_id: str, evaluation_id: str, job_exit: JobExit | None = None
    ) -> EvaluationOutcome:
        """評価の終端処理（7.5）。runner が 1 回だけ呼ぶ。status.json の保存に失敗すれば例外。"""
        version, run_dir = self._locate_evaluation(candidate_id, evaluation_id)
        outcome = self._conclude_evaluation(candidate_id, version, run_dir, job_exit)
        progress = self._progress.get(candidate_id)
        if outcome.status != "running" and progress and progress.evaluation_id == evaluation_id:
            del self._progress[candidate_id]
        return outcome

    def recover_evaluations(self) -> list[EvaluationOutcome]:
        """起動時の復旧（7.6）。未確定の評価を確定し、*.tmp と .preparing_* を消す。

        生きている評価プロセス（PID と作成時刻が一致）は終了させてから確定する。
        実行中だった評価は stopped（interrupted、停止要求があればその理由）になり、
        自動では再開しない。
        """
        outcomes: list[EvaluationOutcome] = []
        self.recovery_blockers = []
        if not self.candidates_root.is_dir():
            return outcomes
        for candidate_dir in sorted(self.candidates_root.iterdir()):
            if _number(_CANDIDATE_ID, candidate_dir.name) is None or not candidate_dir.is_dir():
                continue
            candidate_id = candidate_dir.name
            root = candidate_dir / "evaluations"
            if not root.is_dir():
                continue
            blocked_dirs: list[Path] = []
            for version in self._evaluated_versions(candidate_id):
                for _evaluation_id, run_dir in self._evaluation_dirs(candidate_id, version):
                    if (run_dir / "status.json").exists():
                        continue
                    try:
                        process_file = run_dir / "process.json"
                        if process_file.exists():
                            process = read_json(process_file)
                            if self.process_alive(process):
                                self.process_terminator(process)
                                deadline = time.monotonic() + 5.0
                                while self.process_alive(process) and time.monotonic() < deadline:
                                    time.sleep(0.1)
                                if self.process_alive(process):
                                    logger.error("評価プロセスを終了できませんでした: %s", run_dir)
                                    blocked_dirs.append(run_dir)
                                    self.recovery_blockers.append(
                                        f"評価 {run_dir.name}（候補 {candidate_id}）のプロセスを"
                                        "終了できませんでした。タスクマネージャーで終了してから"
                                        "アプリを再起動してください。"
                                    )
                                    continue
                        outcomes.append(
                            self._conclude_evaluation(candidate_id, version, run_dir, None)
                        )
                    except (OSError, ValueError) as error:
                        logger.error("評価の状態を確定できませんでした (%s): %s", run_dir, error)
            for path in root.glob("*/.preparing_*"):
                if not _within_any(path, blocked_dirs):
                    self._remove(path)
            for path in root.rglob("*.tmp"):
                if not _within_any(path, blocked_dirs):
                    self._remove(path)
        return outcomes

    # ---- 外部解析結果（9.4） ----

    def _read_external(self, candidate_id: str) -> list[dict[str, Any]]:
        path = self._candidate_dir(candidate_id) / "external.json"
        if not path.is_file():
            return []
        try:
            records = read_json(path).get("records")
        except (OSError, ValueError) as error:
            logger.warning("外部解析結果を読めません (%s): %s", path, error)
            return []
        return [item for item in records or [] if isinstance(item, dict)]

    def list_external_results(self, candidate_id: str) -> list[dict[str, Any]]:
        self._read_candidate(candidate_id)
        return copy.deepcopy(self._read_external(candidate_id))

    @staticmethod
    def _latest_external_analysis(
        records: list[dict[str, Any]], evaluation_id: str | None
    ) -> dict[str, Any] | None:
        """その評価に対する、現行形式の外部解析の最新の記録。"""
        matching = [
            item
            for item in records
            if is_external_analysis(item) and item.get("evaluation_id") == evaluation_id
        ]
        return matching[-1] if matching else None

    @staticmethod
    def _external_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
        """外部解析の記録から、一覧・リリースに出す集計（単位と評価 ID 付き）を作る。"""
        if record is None or not isinstance(record.get("summary"), dict):
            return None
        return {
            **copy.deepcopy(record["summary"]),
            "unit": record.get("unit", ""),
            "evaluation_id": record.get("evaluation_id"),
            "record_id": record.get("record_id"),
        }

    def save_external_analysis(
        self,
        candidate_id: str,
        evaluation_id: str,
        values: dict[str, float | None],
        *,
        unit: str = "µm",
        software: str = "",
        software_version: str = "",
        analyzed_on: str = "",
        comment: str | None = None,
    ) -> Candidate:
        """外部解析（画像ごとの円相当径の中央値）を、評価 ID に結び付けて新しい記録として追記する。

        values はその時点の全画像分の値（未入力の画像は含めないか None）。保存のたびに全体の
        スナップショットを 1 件追加し、過去の記録は変えない。値と対象画像はここでも検証し、
        平均と分類別平均をここで求める。comment を渡すと候補のコメントも更新する。
        """
        record = self._read_candidate(candidate_id)
        if record.get("status") == "released":
            raise ValueError("リリース済みの候補の外部解析結果は変更できません")
        version, run_dir = self._locate_evaluation(candidate_id, evaluation_id)
        self._fixed_pair(record)
        if version != record.get("validation_version"):
            raise ValueError("候補に固定した検証用データセットと異なる過去の評価は編集できません")
        evaluation = self._read_record(candidate_id, version, run_dir)
        if evaluation.status != "completed" or evaluation.broken:
            raise ValueError("完了した評価にだけ外部解析結果を保存できます")
        try:
            item_ids = list(read_json(run_dir / "run_spec.json")["validation"]["item_ids"])
            items = {
                item.item_id: item for item in self.dataset_store.select_evaluation_items(version)
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError("評価の対象画像を読めません") from error
        classifications = {
            item_id: (items[item_id].classification if item_id in items else None)
            for item_id in item_ids
        }
        cleaned, summary = build_external_analysis(values, classifications, unit=unit)
        records = self._read_external(candidate_id)
        numbers = [
            int(match.group(1))
            for item in records
            if (match := re.fullmatch(r"ext_(\d+)", str(item.get("record_id", ""))))
        ]
        number = max(numbers, default=0) + 1
        records.append(
            {
                "record_id": f"ext_{number:03d}",
                "format": EXTERNAL_FORMAT,
                "evaluation_id": evaluation_id,
                "validation_version": version,
                "metric": EXTERNAL_METRIC,
                "unit": unit,
                "software": software,
                "software_version": software_version,
                "analyzed_on": analyzed_on,
                "values": cleaned,
                "summary": summary,
                "comment": comment or "",
                "saved_at": _now(),
            }
        )
        _write_json(
            self._candidate_dir(candidate_id) / "external.json",
            {"schema": SCHEMA, "records": records},
        )
        if comment is not None and comment != record.get("comment"):
            record["comment"] = comment
            self._write_candidate(record)
        return self._to_candidate(record)

    # ---- リリース（13 章） ----

    def _release_records(self) -> list[tuple[str, dict[str, Any]]]:
        releases = []
        if not self.releases_root.is_dir():
            return releases
        for folder in self.releases_root.iterdir():
            if _number(_MODEL_ID, folder.name) is None or not folder.is_dir():
                continue
            try:
                value = read_json(folder / "release.json")
                if value.get("model_id") != folder.name:
                    raise ValueError("フォルダ名と model_id が一致しません")
            except (OSError, ValueError) as error:
                logger.warning("リリースを読めません (%s): %s", folder, error)
                continue
            releases.append((folder.name, value))
        return sorted(releases, key=lambda item: _number(_MODEL_ID, item[0]))

    @staticmethod
    def _released_model(
        value: dict[str, Any], lifecycle: dict[str, Any] | None = None
    ) -> ReleasedModel:
        """release.json だけから ReleasedModel を作る（13.4。実験は引かない）。"""
        source = value.get("source") or {}
        evaluation = value.get("evaluation") or {}
        oof_record = value.get("oof") or {}
        oof = oof_record.get("evaluation") or {}
        return ReleasedModel(
            model_id=value["model_id"],
            candidate_id=str((value.get("candidate") or {}).get("candidate_id", "")),
            experiment_id=str(source.get("experiment_id", "")),
            checkpoint="final.pt",
            preprocessing_config=copy.deepcopy(value.get("preprocessing") or {}),
            inference_config=copy.deepcopy(value.get("effective_params") or {}),
            validation_dataset=str(evaluation.get("validation_version", "")),
            evaluation_result=_evaluation_from_summary(
                evaluation.get("overall"), evaluation.get("per_class")
            )
            or Evaluation(None, {}),
            released_at=_parse_time(value.get("released_at")),
            oof_evaluation=_evaluation_from_summary(oof.get("overall"), oof.get("per_class")),
            comment=str(value.get("comment", "")),
            source_attempt_number=int(source.get("attempt") or 1),
            model_type=str(value.get("model_type") or ""),
            inference_config_id=str(value.get("inference_config_id") or ""),
            oof_applicability=str(oof_record.get("applicability") or ""),
            oof_reason=str(oof_record.get("reason") or ""),
            evaluation_id=str(evaluation.get("evaluation_id") or ""),
            external_summary=ComparisonService._external_summary(
                ComparisonService._latest_external_analysis(
                    [
                        item
                        for item in value.get("external_results") or []
                        if isinstance(item, dict)
                    ],
                    evaluation.get("evaluation_id"),
                )
            ),
            lifecycle_status=str((lifecycle or {}).get("status") or "active"),
            releasable=(lifecycle or {}).get("status", "active") == "active",
        )

    def list_released_models(
        self, *, include_archived: bool = False, include_deleted: bool = False
    ) -> list[ReleasedModel]:
        models = []
        for model_id, value in self._release_records():
            try:
                lifecycle = self._release_lifecycle(model_id)
                status = lifecycle.get("status", "active")
                if status == "archived" and not include_archived:
                    continue
                if status == "deleting" or (status == "deleted" and not include_deleted):
                    continue
                models.append(self._released_model(value, lifecycle))
            except (KeyError, TypeError, ValueError) as error:
                logger.warning("リリース %s を読めません: %s", model_id, error)
        return models

    def _release_lifecycle_path(self, model_id: str) -> Path:
        path = self.releases_root / _check_id(_MODEL_ID, model_id, "モデル ID") / "lifecycle.json"
        return self._ensure_release_path_safe(path, check_ancestors=False)

    def _release_lifecycle(self, model_id: str) -> dict[str, Any]:
        path = self._release_lifecycle_path(model_id)
        if not path.is_file():
            return {"schema": 1, "status": "active"}
        value = read_json(path)
        if value.get("model_id") != model_id or value.get("status") not in {
            "active",
            "archived",
            "deleting",
            "deleted",
        }:
            raise ValueError(f"{model_id} の保管・削除状態の記録が不正です")
        return value

    def _write_release_lifecycle(self, model_id: str, value: dict[str, Any]) -> None:
        value = {"schema": 1, "model_id": model_id, **value}
        _write_json(self._release_lifecycle_path(model_id), value)

    def _ensure_release_path_safe(self, path: Path, *, check_ancestors: bool = True) -> Path:
        """releases 配下の対象まで reparse point をたどらず検証する。"""
        root = self.releases_root.absolute()
        target = path.absolute()
        try:
            relative = target.relative_to(root)
        except ValueError as error:
            raise ValueError("対象がリリース保存先の外にあります") from error
        if check_ancestors:
            current = Path(root.anchor)
            for part in root.parts[1:]:
                current /= part
                if current.exists() or current.is_symlink():
                    info = current.lstat()
                    attributes = getattr(info, "st_file_attributes", 0)
                    if stat.S_ISLNK(info.st_mode) or attributes & 0x400:
                        raise ValueError("リリース保存先に reparse point があるため操作できません")
        current = root
        for part in relative.parts:
            current /= part
            if current.exists() or current.is_symlink():
                info = current.lstat()
                attributes = getattr(info, "st_file_attributes", 0)
                if stat.S_ISLNK(info.st_mode) or attributes & 0x400:
                    raise ValueError("リリース内に reparse point があるため操作できません")
        return target

    def set_release_archived(self, model_id: str, archived: bool) -> ReleasedModel:
        """公開モデルを保管または通常一覧へ戻す。公開記録は変更しない。"""
        model_id = _check_id(_MODEL_ID, model_id, "モデル ID")
        lifecycle = self._release_lifecycle(model_id)
        if lifecycle.get("status") in {"deleted", "deleting"}:
            raise ValueError("削除済みまたは削除処理中のモデルは保管状態を変更できません")
        self.get_release_record(model_id)
        assignments = self._read_routing()["assignments"]
        if archived and model_id in assignments.values():
            raise ValueError("振り分けに使用中のモデルは保管できません")
        self._write_release_lifecycle(model_id, {"status": "archived" if archived else "active"})
        value = next(value for key, value in self._release_records() if key == model_id)
        return self._released_model(value, self._release_lifecycle(model_id))

    def estimate_release_delete_bytes(self, model_id: str) -> int:
        """リリースが所有するモデル重みの実ファイル容量を返す。"""
        model_id = _check_id(_MODEL_ID, model_id, "モデル ID")
        lifecycle = self._release_lifecycle(model_id)
        if lifecycle.get("status") in {"deleting", "deleted"}:
            raise ValueError("このモデルは削除済みまたは削除処理中です")
        if model_id in self._read_routing()["assignments"].values():
            raise ValueError("振り分けに使用中のモデルは削除できません")
        folder = self._ensure_release_path_safe(self.releases_root / model_id)
        record = self.get_release_record(model_id)
        candidate_id = (record.get("candidate") or {}).get("candidate_id")
        if candidate_id and self.is_evaluation_active(candidate_id):
            raise ValueError("公開モデルを評価中または評価待ちのため削除できません")
        relative = (record.get("weights") or {}).get("path")
        if relative != "model.pt":
            raise ValueError("公開重みの保存先が想定と異なるため削除できません")
        weight = self._ensure_release_path_safe(folder / "model.pt")
        info = weight.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("公開重みが通常ファイルでないため削除できません")
        return info.st_size if info.st_nlink <= 1 else 0

    def delete_released_model(self, model_id: str) -> int:
        """公開重みだけを削除し、公開記録と履歴要約を保持する。"""
        model_id = _check_id(_MODEL_ID, model_id, "モデル ID")
        size = self.estimate_release_delete_bytes(model_id)
        lifecycle = self._release_lifecycle(model_id)
        previous_status = lifecycle.get("status", "active")
        record = self.get_release_record(model_id)
        folder = self._ensure_release_path_safe(self.releases_root / model_id)
        weight = self._ensure_release_path_safe(folder / "model.pt")
        digest = file_sha256(weight)
        stage_root = self._ensure_release_path_safe(self.releases_root / ".deleting")
        stage = self._ensure_release_path_safe(stage_root / model_id)
        stage_root.mkdir(exist_ok=True)
        stage.mkdir(exist_ok=True)
        if any(stage.iterdir()):
            raise ValueError("前回の削除処理が残っています。復旧後にもう一度お試しください")
        self._write_release_lifecycle(
            model_id,
            {
                "status": "deleting",
                "started_at": _now(),
                "previous_status": previous_status,
                "size": weight.stat().st_size,
                "sha256": digest,
                "evaluation": copy.deepcopy(record.get("evaluation") or {}),
            },
        )
        staged = self._ensure_release_path_safe(stage / "model.pt")
        try:
            os.replace(weight, staged)
            if file_sha256(staged) != digest:
                raise ValueError("削除対象の重みが移動前後で一致しません")
            staged.unlink()
            self._write_release_lifecycle(
                model_id,
                {
                    "status": "deleted",
                    "deleted_at": _now(),
                    "size": weight.stat().st_size if weight.exists() else size,
                    "sha256": digest,
                    "evaluation": copy.deepcopy(record.get("evaluation") or {}),
                },
            )
            try:
                stage.rmdir()
            except OSError:
                logger.warning("削除済みリリース %s の空 staging を後で整理します", model_id)
            return size
        except BaseException:
            if staged.exists() and not weight.exists():
                try:
                    os.replace(staged, weight)
                except OSError:
                    logger.exception("リリース %s の削除中断後に重みを戻せません", model_id)
            if weight.exists():
                try:
                    self._write_release_lifecycle(model_id, {"status": previous_status})
                    if stage.exists() and not any(stage.iterdir()):
                        stage.rmdir()
                except OSError:
                    logger.exception("リリース %s の削除中断後に状態を戻せません", model_id)
            raise

    def recover_release_deletions(self) -> list[str]:
        """削除途中の公開重みを安全に戻すか、完了記録へ進める。"""
        recovered = []
        for model_id, _record in self._release_records():
            try:
                lifecycle = self._release_lifecycle(model_id)
                if lifecycle.get("status") != "deleting":
                    continue
                folder = self._ensure_release_path_safe(self.releases_root / model_id)
                weight = self._ensure_release_path_safe(folder / "model.pt")
                stage = self._ensure_release_path_safe(
                    self.releases_root / ".deleting" / model_id / "model.pt"
                )
                if stage.exists() and not weight.exists():
                    if file_sha256(stage) != lifecycle.get("sha256"):
                        raise ValueError("削除途中の重みが記録と一致しません")
                    os.replace(stage, weight)
                    stage.parent.rmdir()
                    self._write_release_lifecycle(
                        model_id,
                        {"status": lifecycle.get("previous_status", "active")},
                    )
                elif weight.exists() and not stage.exists():
                    self._write_release_lifecycle(
                        model_id,
                        {"status": lifecycle.get("previous_status", "active")},
                    )
                elif not weight.exists() and not stage.exists():
                    self._write_release_lifecycle(
                        model_id,
                        {
                            "status": "deleted",
                            "deleted_at": _now(),
                            "size": lifecycle.get("size", 0),
                            "sha256": lifecycle.get("sha256"),
                            "evaluation": copy.deepcopy(lifecycle.get("evaluation") or {}),
                        },
                    )
                else:
                    raise ValueError("削除途中の重みが保存先と staging の両方にあります")
                recovered.append(model_id)
            except (OSError, ValueError) as error:
                logger.warning("リリース %s の削除を復旧できません: %s", model_id, error)
        return recovered

    def get_release_record(self, model_id: str, *, allow_deleted: bool = False) -> dict[str, Any]:
        """release.json を読み、model.pt の存在と大きさを確かめて返す。読めなければ ValueError。"""
        folder = self.releases_root / _check_id(_MODEL_ID, model_id, "モデル ID")
        lifecycle = self._release_lifecycle(model_id)
        if lifecycle.get("status") == "deleted" and allow_deleted:
            try:
                value = read_json(folder / "release.json")
            except (OSError, ValueError) as error:
                raise ValueError(f"リリース {model_id} の履歴を読めません") from error
            if value.get("model_id") != model_id:
                raise ValueError(f"リリース {model_id} の記録が一致しません")
            return value
        if lifecycle.get("status") in {"deleted", "deleting"}:
            raise ValueError(f"リリース {model_id} の重みは削除済みまたは削除処理中です")
        try:
            value = read_json(folder / "release.json")
        except (OSError, ValueError) as error:
            raise ValueError(f"リリース {model_id} の記録を読めません") from error
        if value.get("model_id") != model_id:
            raise ValueError(f"リリース {model_id} の記録が一致しません")
        weights = value.get("weights") or {}
        path = resolve_recorded_path(folder, weights.get("path"))
        if not path.is_file() or path.stat().st_size != weights.get("size"):
            raise ValueError(f"リリース {model_id} のモデルファイルがありません")
        return value

    def _published_release(
        self,
        record: dict[str, Any],
        releases: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """候補を参照する公開済みリリースを 1 件返す（13.3）。なければ None。

        リリース登録の再試行と起動時の復旧が同じ規則で判定する。次の場合は ValueError:
        同じ候補のリリースが複数ある、候補の released_model_id が見つからないか別の候補を指す、
        リリースの候補 fingerprint・重み（sha256・バイト数）が候補の記録と一致しない、
        リリースが参照する評価が候補にない（または検証用の版が一致しない）。
        旧形式のリリース（学習混入の項目を持つもの）も同じ規則で読み、書き換えない。
        """
        candidate_id = record.get("candidate_id")
        if releases is None:
            releases = self._release_records()
        matches = [
            (model_id, release)
            for model_id, release in releases
            if (release.get("candidate") or {}).get("candidate_id") == candidate_id
        ]
        referenced = record.get("released_model_id")
        if referenced is not None and referenced not in [model_id for model_id, _ in matches]:
            raise ValueError(
                f"候補 {candidate_id} が参照するリリース {referenced} が見つからないか、"
                "別の候補のリリースです"
            )
        if len(matches) > 1:
            names = "、".join(model_id for model_id, _release in matches)
            raise ValueError(f"候補 {candidate_id} のリリースが複数あります（{names}）")
        if not matches:
            return None
        model_id, release = matches[0]
        weights = (record.get("source") or {}).get("weights") or {}
        published = release.get("weights") or {}
        if (
            (release.get("candidate") or {}).get("fingerprint") != record.get("fingerprint")
            or published.get("sha256") != weights.get("sha256")
            or published.get("size") != weights.get("size")
        ):
            raise ValueError(f"リリース {model_id} の記録が候補 {candidate_id} と一致しません")
        evaluation = release.get("evaluation") or {}
        try:
            version, _run_dir = self._locate_evaluation(
                candidate_id, evaluation.get("evaluation_id")
            )
        except ValueError as error:
            raise ValueError(
                f"リリース {model_id} が参照する評価が候補 {candidate_id} にありません"
            ) from error
        if version != evaluation.get("validation_version"):
            raise ValueError(
                f"リリース {model_id} が参照する評価の検証用データセットが一致しません"
            )
        return model_id, release

    def _mark_released(self, record: dict[str, Any], model_id: str) -> bool:
        """候補を released・released_model_id=model_id にする。書き換えたら True。"""
        if record.get("status") == "released" and record.get("released_model_id") == model_id:
            return False
        record["status"] = "released"
        record["released_model_id"] = model_id
        self._write_candidate(record)
        return True

    def _reuse_published_release(self, record: dict[str, Any]) -> ReleasedModel | None:
        """同じ候補の公開済みリリースがあれば、候補の状態だけを修復してそのリリースを返す。

        新しいリリースは作らず、公開済みの記録（評価・コメントを含む）は書き換えない。
        今回指定された評価が別の評価（再評価の後の再試行など）でも、公開済みリリースの評価を
        保持する。照合するのは公開済みリリースが参照する評価。返すモデルには
        recovered_release=True を付ける（is_recovered_release で判定できる）。
        参照が食い違う場合は何も作らずに ValueError。
        """
        candidate_id = record["candidate_id"]
        try:
            published = self._published_release(record)
        except ValueError as error:
            raise ValueError(
                "この候補のリリース記録に食い違いがあるため登録できません。"
                f"リリースと候補の記録を確認してください: {error}"
            ) from error
        if published is None:
            return None
        model_id, release = published
        # モデルファイルの有無と大きさを確かめる（読めなければ ValueError）
        self.get_release_record(model_id, allow_deleted=True)
        try:
            self._mark_released(record, model_id)
        except OSError as error:
            raise ValueError(
                f"{model_id} はリリース済みですが、候補の状態を保存できませんでした。"
                f"もう一度登録すると状態を修復します: {error}"
            ) from error
        logger.info("候補 %s の公開済みリリース %s を再利用しました", candidate_id, model_id)
        model = self._released_model(release, self._release_lifecycle(model_id))
        setattr(model, RECOVERED_RELEASE_ATTR, True)
        return model

    def release_candidate(
        self,
        candidate_id: str,
        evaluation_id: str,
        comment: str = "",
        *,
        is_evaluation_active: Callable[[str], bool] | None = None,
    ) -> ReleasedModel:
        """候補を指定した評価でリリースする（13.1・13.2）。重みは独立したコピーを持つ。

        公開（改名）の後の候補保存だけが失敗していた場合の再試行では、新しいリリースを作らず
        公開済みのリリースを返す（is_recovered_release が True。評価は公開済みのまま）。
        新規リリースでは、候補に固定した組を保存された組の情報と照合し、評価の検証用の版が
        候補の版と一致することを確かめる。旧形式で学習混入が見つかっていた評価は使えない。
        """
        record = self._read_candidate(candidate_id)
        if record.get("status") != "candidate":
            raise ValueError("候補状態のモデルだけリリースできます")
        active = is_evaluation_active or self.is_evaluation_active
        if active(candidate_id):
            raise ValueError("評価中または評価待ちの候補はリリースできません")
        # 採番・重みのコピーより前に、前回の登録で公開済みのリリースを探す
        reused = self._reuse_published_release(record)
        if reused is not None:
            return reused
        training_version, fixed_version = self._fixed_pair(record)
        version, run_dir = self._locate_evaluation(candidate_id, evaluation_id)
        if version != fixed_version:
            raise ValueError(
                f"この評価は候補の検証用データセット（{fixed_version}）ではなく {version} で"
                "行われたため、リリースに使えません"
            )
        evaluation = self._read_record(candidate_id, version, run_dir)
        if evaluation.status != "completed":
            raise ValueError("完了した評価だけでリリースできます")
        if not self.verify_evaluation(candidate_id, evaluation_id):
            raise ValueError(BROKEN_MESSAGE)
        if evaluation.contamination_found:
            raise ValueError(CONTAMINATION_FOUND_MESSAGE)
        spec = read_json(run_dir / "run_spec.json")
        result = read_json(run_dir / "result.json")
        if spec.get("candidate_id") not in (None, candidate_id):
            raise ValueError("評価の候補 ID が一致しません")
        source = record["source"]
        weights = source["weights"]
        source_path = self._weights_path(source)
        if not source_path.is_file() or source_path.stat().st_size != weights["size"]:
            raise ValueError("最終学習モデルのファイルがありません")
        self.releases_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.releases_root).free
        if free < weights["size"] + RELEASE_SPACE_MARGIN:
            raise ValueError("リリースに必要な空き容量がありません")
        preparing = self.releases_root / f".preparing_{uuid.uuid4()}"
        preparing.mkdir()
        try:
            temporary = preparing / "model.pt.tmp"
            # ハードリンクは使わない（元の実体への書き込みがリリースへ波及しないように）
            with source_path.open("rb") as reader, temporary.open("wb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, preparing / "model.pt")
            if file_sha256(preparing / "model.pt") != weights["sha256"]:
                raise ValueError("コピーした重みの sha256 が候補の記録と一致しません")
            numbers = [
                _number(_MODEL_ID, path.name) or 0 for path in self.releases_root.glob("model_*")
            ]
            model_id = f"model_{max(numbers, default=0) + 1:03d}"
            model_config = copy.deepcopy((source.get("experiment_config") or {}).get("model") or {})
            release = {
                "schema": SCHEMA,
                "model_id": model_id,
                "released_at": _now(),
                "comment": comment,
                "app_version": self.app_version,
                "weights": {
                    "path": "model.pt",
                    "size": weights["size"],
                    "sha256": weights["sha256"],
                },
                "source": {
                    "experiment_id": source["experiment_id"],
                    "attempt": source["attempt"],
                    "run_id": source["run_id"],
                    "weights_path": weights["path"],
                },
                "model_type": source.get("model_type"),
                "model_config": model_config,
                "preprocessing": copy.deepcopy(source.get("preprocessing") or {}),
                "effective_params": copy.deepcopy(record.get("effective_params") or {}),
                "inference_config_id": record.get("inference_config_id"),
                "candidate": {
                    "candidate_id": candidate_id,
                    "fingerprint": record.get("fingerprint"),
                },
                "dataset_pair": {
                    "training_version": training_version,
                    "validation_version": fixed_version,
                },
                "evaluation": {
                    "evaluation_id": evaluation_id,
                    "validation_version": version,
                    "schema": evaluation.schema,
                    "input_fingerprint": result.get("input_fingerprint"),
                    "overall": copy.deepcopy(result.get("overall")),
                    "per_class": copy.deepcopy(result.get("per_class") or {}),
                    "metric": copy.deepcopy(result.get("metric") or spec.get("metric")),
                },
                "oof": copy.deepcopy(record.get("oof") or {}),
                "external_results": copy.deepcopy(self._read_external(candidate_id)),
                "particle_split_id": PARTICLE_SPLIT_ID,
            }
            _write_json(preparing / "release.json", release)
            os.replace(preparing, self.releases_root / model_id)
        except BaseException:
            # 自分が作った準備フォルダだけは、失敗したその場で片付ける（大きな重みを残さない）
            self._remove(preparing)
            raise
        try:
            self._mark_released(record, model_id)
        except OSError as error:
            # 公開済みのフォルダは消さない。同じ登録の再試行で候補の状態を修復する
            raise ValueError(
                f"{model_id} としてリリースしましたが、候補の状態を保存できませんでした。"
                f"もう一度登録すると状態を修復します: {error}"
            ) from error
        return self._released_model(release)

    # ---- 振り分け（14 章） ----

    def _routing_path(self) -> Path:
        return self.releases_root / "routing.json"

    def _read_routing(self) -> dict[str, Any]:
        path = self._routing_path()
        if not path.is_file():
            return {"schema": SCHEMA, "revision": 0, "assignments": {}, "history": []}
        value = read_json(path)
        if (
            type(value.get("revision")) is not int
            or not isinstance(value.get("assignments"), dict)
            or not isinstance(value.get("history", []), list)
        ):
            raise ValueError("routing.json の形式が不正です")
        value.setdefault("history", [])
        return value

    def get_routing_state(self) -> RoutingState:
        value = self._read_routing()
        return RoutingState(value["revision"], dict(value["assignments"]))

    def get_routing(self) -> dict[str, str | None]:
        """分類ごとの現在の振り分け（本番推論画面との互換用）。"""
        return dict(self._read_routing()["assignments"])

    def list_routing_history(self) -> list[RoutingHistory]:
        return [
            RoutingHistory(
                _parse_time(item.get("changed_at")),
                str(item.get("classification", "")),
                item.get("before"),
                item.get("after"),
            )
            for item in self._read_routing()["history"]
            if isinstance(item, dict)
        ]

    def list_routing_classifications(self) -> list[str]:
        """確定済みの全データセット版（学習・検証）の分類 ∪ routing.json にある分類。"""
        names: set[str] = set(self._read_routing()["assignments"])
        for purpose in ("train", "val"):
            for version in self.dataset_store.list_versions(purpose):
                try:
                    items = self.dataset_store.get_items(version, expected_purpose=purpose)
                except (OSError, ValueError, KeyError) as error:
                    logger.warning("データセット %s を読めません: %s", version, error)
                    continue
                names.update(item.classification for item in items if item.classification)
        return sorted(names)

    def apply_routing(
        self, changes: dict[str, str | None], *, expected_revision: int
    ) -> RoutingState:
        """全変更を検証してから 1 回の原子的な置き換えで適用する。1 つでも不正なら何も変えない。"""
        value = self._read_routing()
        if value["revision"] != expected_revision:
            raise ValueError("振り分けが別の操作で変更されました。画面を開き直してください")
        classifications = set(self.list_routing_classifications())
        released = {
            model.model_id
            for model in self.list_released_models(include_archived=False, include_deleted=False)
        }
        errors = []
        for classification, model_id in changes.items():
            if not isinstance(classification, str) or classification not in classifications:
                errors.append(f"振り分けの対象にない分類です: {classification}")
            if model_id is not None and model_id not in released:
                known = next(
                    (value for key, value in self._release_records() if key == model_id), None
                )
                if known is None:
                    errors.append(f"未登録のリリースモデルです: {model_id}")
                else:
                    errors.append(
                        f"保管中または削除済みのリリースモデルは振り分けに使えません: {model_id}"
                    )
        if errors:
            raise ValueError("、".join(errors))
        assignments = dict(value["assignments"])
        changed = [
            (classification, assignments.get(classification), model_id)
            for classification, model_id in changes.items()
            if assignments.get(classification) != model_id
            or classification not in value["assignments"]
        ]
        if not changed:
            return RoutingState(value["revision"], assignments)
        revision = value["revision"] + 1
        changed_at = _now()
        history = list(value["history"])
        for classification, before, after in changed:
            assignments[classification] = after
            if before != after:
                history.append(
                    {
                        "changed_at": changed_at,
                        "revision": revision,
                        "classification": classification,
                        "before": before,
                        "after": after,
                    }
                )
        self.releases_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            self._routing_path(),
            {
                "schema": SCHEMA,
                "revision": revision,
                "assignments": assignments,
                "history": history,
            },
        )
        return RoutingState(revision, dict(assignments))

    def resolve_model(self, classification: str | None) -> ReleasedModel | None:
        """分類に割り当てたリリースを返す。読めないリリースは他へ切り替えずに ValueError。"""
        if not classification:
            return None
        model_id = self._read_routing()["assignments"].get(classification)
        if model_id is None:
            return None
        model = next(
            (
                item
                for item in self.list_released_models(include_archived=True)
                if item.model_id == model_id
            ),
            None,
        )
        if model is None or not model.releasable:
            raise ValueError("振り分け先のモデルは保管または削除済みです")
        self.get_release_record(model_id)
        return model

    # ---- 参照保護 ----

    def protected_final_reason(self, experiment_id: str, attempt: int) -> str:
        """final.pt を消せない理由（19.1）。候補状態の候補が参照していれば空でない文字列。"""
        users = [
            record["candidate_id"]
            for record in self._candidate_records()
            if (record.get("source") or {}).get("experiment_id") == experiment_id
            and (record.get("source") or {}).get("attempt") == attempt
            and (
                record.get("status") == "candidate"
                or self.is_evaluation_active(record["candidate_id"])
            )
        ]
        if not users:
            return ""
        return "比較候補 " + "、".join(users) + " がこの最終学習モデルを参照しています"

    def experiment_reference_reason(self, experiment_id: str) -> str:
        """実験を削除できない理由（13.5）。非採用以外の候補・リリースが参照していれば空でない。"""
        candidates = [
            record["candidate_id"]
            for record in self._candidate_records()
            if record.get("status") != "rejected"
            and (record.get("source") or {}).get("experiment_id") == experiment_id
        ]
        releases = [
            model_id
            for model_id, value in self._release_records()
            if (value.get("source") or {}).get("experiment_id") == experiment_id
        ]
        parts = []
        if candidates:
            parts.append("比較候補 " + "、".join(candidates))
        if releases:
            parts.append("リリース済みモデル " + "、".join(releases))
        if not parts:
            return ""
        return "・".join(parts) + " から参照されているため削除できません"
