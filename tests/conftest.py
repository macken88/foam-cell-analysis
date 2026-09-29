"""テスト全体で共有する準備（評価のための小さな workspace）。"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from foam_cell_analysis.data.dataset_store import DatasetStore
from foam_cell_analysis.services.comparison_service import (
    ComparisonService,
    candidate_fingerprint,
)

SIZE = 24


def _blobs(*corners: tuple[int, int], size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """背景 0・気泡 200 の画像と、気泡ごとに連番の正解ラベルを作る。"""
    image = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint16)
    for number, (row, col) in enumerate(corners, start=1):
        image[row : row + size, col : col + size] = 200
        mask[row : row + size, col : col + size] = number
    return image, mask


def write_dataset(
    root: Path,
    version: str,
    purpose: str,
    samples: dict[str, tuple[np.ndarray, np.ndarray, str | None]],
    *,
    channel: str = "A",
) -> Path:
    """確定済みデータセット（dataset_info・metadata・manifest・画像・マスク）を書く。"""
    folder = root / "datasets" / version
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir()
    metadata, manifest = [], []
    for item_id, (image, mask, classification) in samples.items():
        Image.fromarray(image).save(folder / "images" / f"{item_id}.png")
        Image.fromarray(mask).save(folder / "masks" / f"{item_id}.png")
        metadata.append(
            {
                "item_id": item_id,
                "source_relpath": f"g/{item_id}.png",
                "channel": channel,
                "usage": purpose,
                "classification": classification or "",
            }
        )
        manifest.append(
            {
                "item_id": item_id,
                "image_path": f"images/{item_id}.png",
                "image_sha256": hashlib.sha256(
                    (folder / "images" / f"{item_id}.png").read_bytes()
                ).hexdigest(),
                "mask_path": f"masks/{item_id}.png",
                "mask_sha256": hashlib.sha256(
                    (folder / "masks" / f"{item_id}.png").read_bytes()
                ).hexdigest(),
            }
        )
    for name, rows, fields in (
        (
            "metadata.csv",
            metadata,
            ["item_id", "source_relpath", "channel", "usage", "classification"],
        ),
        (
            "manifest.csv",
            manifest,
            ["item_id", "image_path", "image_sha256", "mask_path", "mask_sha256"],
        ),
    ):
        with (folder / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    info = {"purpose": purpose, "status": "RELEASED", "dataset_version": version}
    (folder / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    return folder


def validation_samples() -> dict[str, tuple[np.ndarray, np.ndarray, str | None]]:
    """検証版の 3 枚: 完全一致・過剰検出・空画像（分類 A・B と未分類）。"""
    perfect = _blobs((2, 2), (12, 12))
    extra_image, _ = _blobs((2, 2), (14, 6))
    _, one_mask = _blobs((2, 2))
    empty = (np.zeros((SIZE, SIZE), np.uint8), np.zeros((SIZE, SIZE), np.uint16))
    return {
        "val_a": (*perfect, "分類A"),
        "val_b": (extra_image, one_mask, "分類B"),
        "val_c": (*empty, None),
    }


class StubTraining:
    """ComparisonService が使う TrainingService の代役（dataset_store と版情報だけ）。"""

    app_version = "0.1.0"
    git_commit = None

    def __init__(self, root: Path) -> None:
        self.workspace_root = root
        self.dataset_store = DatasetStore(root)


class EvaluationEnv:
    """評価のテスト用 workspace と ComparisonService。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.alive: dict[int, bool] = {}
        self.terminated: list[dict] = []
        self.service = ComparisonService(
            root,
            StubTraining(root),
            process_alive=lambda process: self.alive.get(process.get("pid"), False),
            process_terminator=self._terminate,
        )

    def write_dataset(self, version: str, purpose: str, samples: dict) -> Path:
        return write_dataset(self.root, version, purpose, samples)

    def _terminate(self, process: dict) -> None:
        self.terminated.append(process)
        self.alive[process.get("pid")] = False

    def add_candidate(
        self,
        candidate_id: str = "RC-001",
        *,
        experiment_id: str = "exp_0001",
        input_channels: tuple[str, ...] = ("A",),
        weights: bytes = b"fake-weights",
        used_item_ids: list[str] | None = None,
        training_version: str = "train_v000",
    ) -> dict:
        """fake_numpy の比較候補を candidate.json として直接置く（add_candidate と同じ形）。"""
        run_dir = self.root / "experiments" / experiment_id / "runs" / "attempt_001"
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints" / "final.pt").write_bytes(weights)
        train_dir = self.root / "datasets" / training_version
        dataset_sha = {
            name: hashlib.sha256((train_dir / name).read_bytes()).hexdigest()
            for name in ("manifest.csv", "metadata.csv")
            if (train_dir / name).is_file()
        }
        (run_dir / "run_spec.json").write_text(
            json.dumps(
                {
                    "run_id": f"{experiment_id}/attempt_001",
                    "dataset": {
                        "version": training_version,
                        "path": f"datasets/{training_version}",
                        "sha256": dataset_sha,
                    },
                    "used_item_ids": used_item_ids or ["train_1", "train_2"],
                }
            ),
            encoding="utf-8",
        )
        model_config = {"type": "fake_numpy"}
        effective = {"threshold": 0.5}
        weights_sha = hashlib.sha256(weights).hexdigest()
        record = {
            "schema": 1,
            "candidate_id": candidate_id,
            "created_at": "2026-09-30T10:00:00+09:00",
            "comment": "",
            "status": "candidate",
            "inference_config_id": "infer_v001",
            "effective_params": copy.deepcopy(effective),
            "source": {
                "experiment_id": experiment_id,
                "attempt": 1,
                "run_id": f"{experiment_id}/attempt_001",
                "selected_epoch": 1,
                "model_type": "fake_numpy",
                "weights": {
                    "path": "checkpoints/final.pt",
                    "size": len(weights),
                    "sha256": weights_sha,
                },
                "experiment_config": {
                    "model": model_config,
                    "data": {
                        "dataset_version": training_version,
                        "input_channels": list(input_channels),
                    },
                },
                "preprocessing": {},
                "training_dataset": {"version": training_version, "sha256": dataset_sha},
                "training_eval_params": copy.deepcopy(effective),
                "training_eval_params_filled": False,
            },
            "oof": {"evaluation": {}, "applicability": "matching", "reason": None},
            "fingerprint": candidate_fingerprint(
                weights_sha, "fake_numpy", model_config, {}, effective
            ),
        }
        folder = self.service.candidates_root / candidate_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "candidate.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        return record


@pytest.fixture
def evaluation_env(tmp_path) -> EvaluationEnv:
    """検証版 val_v000（3 枚）と学習版 train_v000、fake_numpy の候補 RC-001 を持つ workspace。"""
    write_dataset(tmp_path, "val_v000", "val", validation_samples())
    other, other_mask = _blobs((6, 6))
    write_dataset(
        tmp_path,
        "train_v000",
        "train",
        {"train_1": (other, other_mask, "分類A"), "train_2": (*_blobs((1, 10)), "分類B")},
    )
    env = EvaluationEnv(tmp_path)
    env.add_candidate()
    return env
