"""データ拡張プロファイルの保存と初期版の作成。"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from foam_cell_analysis.services.models import AugmentationProfile, TransformSetting

TRANSFORMS = [
    ("horizontal_flip", "左右反転", True, 0.5, None, None),
    ("vertical_flip", "上下反転", True, 0.5, None, None),
    ("rotation", "回転", True, 0.5, -180.0, 180.0),
    ("scale", "拡大縮小", True, 0.3, 0.8, 1.2),
    ("translation", "平行移動", True, 0.0, -0.1, 0.1),
    ("crop", "切り出し", True, 0.0, 0.5, 1.0),
    ("elastic", "弾性変形", True, 0.0, 0.0, 1.0),
    ("brightness", "明るさ", True, 0.3, -10.0, 10.0),
    ("contrast", "コントラスト", True, 0.3, 0.9, 1.1),
    ("gamma", "ガンマ", True, 0.0, 0.7, 1.5),
    ("blur", "ぼかし", True, 0.2, 0.0, 1.0),
    ("noise", "ノイズ", True, 0.2, 0.0, 12.0),
    ("channel_dropout", "チャンネル欠落", True, 0.0, 0.0, 1.0),
    ("channel_intensity", "チャンネル強度変動", True, 0.0, 0.8, 1.2),
]

logger = logging.getLogger(__name__)


class ProfileStore:
    """augmentation/*.json を原子的に保存する。"""

    def __init__(self, workspace_root: str | Path) -> None:
        self.directory = Path(workspace_root) / "augmentation"
        self.profiles: dict[str, AugmentationProfile] = {}
        self.recovery_issues: dict[str, str] = {}
        self._load_or_seed()

    @staticmethod
    def _profile_from_dict(value: dict[str, Any]) -> AugmentationProfile:
        transforms = [TransformSetting(**item) for item in value.get("transforms", [])]
        for transform in transforms:
            if transform.key == "translate":
                transform.key = "translation"
        order = ["translation" if key == "translate" else key for key in value.get("order", [])]
        return AugmentationProfile(
            value["profile_id"],
            value.get("name", ""),
            value.get("base_profile"),
            transforms,
            order,
            value.get("used_by_experiments", []),
        )

    @staticmethod
    def _profile_dict(profile: AugmentationProfile) -> dict[str, Any]:
        return {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "base_profile": profile.base_profile,
            "transforms": [vars(item) for item in profile.transforms],
            "order": profile.order,
            "used_by_experiments": profile.used_by_experiments,
        }

    def _load_or_seed(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        files = sorted(self.directory.glob("aug_v*.yaml"))
        for path in files:
            try:
                value = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("形式が不正です")
                self.profiles[path.stem] = self._profile_from_dict(value)
            except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
                self.recovery_issues[path.stem] = "データ拡張設定を読めません"
                logger.warning("データ拡張設定を読めません (%s): %s", path, error)
        if not self.profiles:
            if self.recovery_issues:
                return
            profile = AugmentationProfile(
                "aug_v001",
                "標準",
                None,
                [TransformSetting(*definition) for definition in TRANSFORMS],
                [definition[0] for definition in TRANSFORMS],
            )
            self.save(profile)

    def list(self) -> list[AugmentationProfile]:
        return [copy.deepcopy(self.profiles[key]) for key in sorted(self.profiles)]

    def get(self, profile_id: str) -> AugmentationProfile:
        return copy.deepcopy(self.profiles[profile_id])

    def save(self, profile: AugmentationProfile) -> AugmentationProfile:
        saved = copy.deepcopy(profile)
        numbers = [int(key[-3:]) for key in self.profiles if key[-3:].isdigit()]
        numbers.extend(
            int(path.stem[-3:])
            for path in self.directory.glob("aug_v*.yaml")
            if path.stem[-3:].isdigit()
        )
        number = max(numbers, default=0) + 1
        saved.profile_id = f"aug_v{number:03d}"
        saved.base_profile = saved.base_profile or (max(self.profiles) if self.profiles else None)
        saved.used_by_experiments = []
        target = self.directory / f"{saved.profile_id}.yaml"
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            yaml.safe_dump(self._profile_dict(saved), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        self.profiles[saved.profile_id] = saved
        return copy.deepcopy(saved)
