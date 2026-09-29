"""学習・検証デバッグ用の合成データセット（train_v000 / val_v000 など）を生成する。"""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

SAMPLES = [
    ("分類A", "group_a01", "良", 25),
    ("分類A", "group_a01", "良", 25),
    ("分類A", "group_a02", "良", 25),
    ("分類A", "group_a02", "不良", 25),
    ("分類A", "group_a03", "良", 25),
    ("分類A", "group_a03", "良", 0),
    ("分類B", "group_b01", "良", 25),
    ("分類B", "group_b01", "良", 25),
    ("分類B", "group_b02", "良", 25),
    ("分類B", "group_b02", "不良", 25),
    ("分類B", "group_b03", "良", 25),
    ("分類B", "group_b03", "良", 0),
]


def make_sample(
    seed: int, n_instances: int = 25, poor: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """重ならない気泡と、連番の正解インスタンスマスクを作る。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:512, :512]
    labels = np.zeros((512, 512), dtype=np.uint16)
    image = rng.normal(40, 18 if poor else 7, labels.shape)
    # 各区画に一つ配置し、画像に見える気泡と正解マスクを一致させる。
    for index in range(n_instances):
        row, col = divmod(index, 5)
        cy, cx = np.array([row, col]) * 100 + rng.integers(42, 60, size=2)
        radius = int(rng.integers(12, 33))
        distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        disk = distance <= radius
        labels[disk] = index + 1
        image[disk] += 20 if poor else 70
        image[disk & (distance > radius * 0.78)] += 25 if poor else 75
    return np.clip(image, 0, 255).astype(np.uint8), labels


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(
    output: Path, seed: int = 42, purpose: str = "train", version: str | None = None
) -> None:
    """既存の保存先を上書きせず、12組の画像とメタデータを保存する。"""
    version = version or f"{purpose}_v000"
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    metadata, manifest = [], []
    for number, (classification, group, quality, n_instances) in enumerate(SAMPLES, 1):
        sample_seed = seed + number - 1
        image, labels = make_sample(sample_seed, n_instances, quality == "不良")
        filename = f"dummy_{number:03d}.png"
        image_path, mask_path = f"images/{filename}", f"masks/{filename}"
        Image.fromarray(image).save(output / image_path)
        Image.fromarray(labels).save(output / mask_path)
        item_id = f"dummy_{number:03d}"
        metadata.append(
            {
                "item_id": item_id,
                "source_filename": filename,
                "source_relpath": f"synthetic/{group}/{filename}",
                "group_id": group,
                "channel": "A",
                "usage": purpose,
                "classification": classification,
                "quality": quality,
                "n_instances": n_instances,
                "mask_revision": "rev_001",
                "seed": sample_seed,
                "is_dummy": True,
            }
        )
        manifest.append(
            {
                "item_id": item_id,
                "image_path": image_path,
                "image_sha256": hashlib.sha256((output / image_path).read_bytes()).hexdigest(),
                "mask_path": mask_path,
                "mask_sha256": hashlib.sha256((output / mask_path).read_bytes()).hexdigest(),
                "mask_revision": "rev_001",
            }
        )
    write_csv(output / "metadata.csv", metadata)
    write_csv(output / "manifest.csv", manifest)
    info = {
        "schema_version": 1,
        "dataset_version": version,
        "purpose": purpose,
        "parent_version": None,
        "base_validation_version": None,
        "created_at": datetime.now(UTC).isoformat(),
        "n_images": len(SAMPLES),
        "application_version": "0.1.0",
        "status": "RELEASED",
        "is_dummy": True,
        "comment": (
            "学習処理のデバッグ用合成データ。実データの精度評価には使用しない。"
            if purpose == "train"
            else "評価処理のデバッグ用合成データ。実データの精度評価には使用しない。"
        ),
        "generator": "scripts/generate_dummy_dataset.py",
        "generator_version": 2,
        "seed": seed,
        "image_size": [512, 512],
        "image_dtype": "uint8",
        "mask_dtype": "uint16",
        "instances_per_image": [sample[3] for sample in SAMPLES],
        "classification_counts": {"分類A": 6, "分類B": 6},
        "quality_counts": {"良": 10, "不良": 2},
        "n_groups": 6,
        "n_empty_images": 2,
        "numpy_version": np.__version__,
    }
    (output / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purpose", choices=["train", "val"], default="train")
    parser.add_argument("--version", default=None, help="版名（既定: <purpose>_v000）")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None, help="既定: train=42, val=1042")
    args = parser.parse_args()
    version = args.version or f"{args.purpose}_v000"
    seed = args.seed if args.seed is not None else (42 if args.purpose == "train" else 1042)
    output = args.output or Path(__file__).resolve().parents[1] / "workspace/datasets" / version
    if output.exists():
        parser.error(f"保存先が既に存在します。別の --output を指定してください: {output}")
    generate(output, seed, args.purpose, version)
    print(f"Created {version}: {output.resolve()} ({len(SAMPLES)} images + masks)")


if __name__ == "__main__":
    main()
