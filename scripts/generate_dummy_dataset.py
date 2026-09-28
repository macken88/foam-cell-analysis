"""学習デバッグ用 train_v000 を生成する（アプリへの登録は別途行う）。"""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image


def make_sample(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """重ならない気泡と、連番の正解インスタンスマスクを作る。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:512, :512]
    labels = np.zeros((512, 512), dtype=np.uint16)
    image = rng.normal(40, 7, labels.shape)
    # 各区画に一つ配置し、画像に見える気泡と正解マスクを一致させる。
    for index in range(25):
        row, col = divmod(index, 5)
        cy, cx = np.array([row, col]) * 100 + rng.integers(42, 60, size=2)
        radius = int(rng.integers(12, 33))
        distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        disk = distance <= radius
        labels[disk] = index + 1
        image[disk] += 70
        image[disk & (distance > radius * 0.78)] += 75
    return np.clip(image, 0, 255).astype(np.uint8), labels


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(output: Path, seed: int = 42) -> None:
    """既存の保存先を上書きせず、5組の画像とメタデータを保存する。"""
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    metadata, manifest = [], []
    for number in range(1, 6):
        sample_seed = seed + number - 1
        image, labels = make_sample(sample_seed)
        filename = f"dummy_{number:03d}.png"
        image_path, mask_path = f"images/{filename}", f"masks/{filename}"
        Image.fromarray(image).save(output / image_path)
        Image.fromarray(labels).save(output / mask_path)
        item_id = f"dummy_{number:03d}"
        metadata.append({
            "item_id": item_id,
            "source_filename": filename,
            "source_relpath": f"synthetic/source_{number:03d}/{filename}",
            "channel": "A",
            "usage": "train",
            "classification": "分類A",
            "quality": "良",
            "mask_revision": "rev_001",
            "seed": sample_seed,
            "is_dummy": True,
        })
        manifest.append({
            "item_id": item_id,
            "image_path": image_path,
            "image_sha256": hashlib.sha256((output / image_path).read_bytes()).hexdigest(),
            "mask_path": mask_path,
            "mask_sha256": hashlib.sha256((output / mask_path).read_bytes()).hexdigest(),
            "mask_revision": "rev_001",
        })
    write_csv(output / "metadata.csv", metadata)
    write_csv(output / "manifest.csv", manifest)
    info = {
        "schema_version": 1,
        "dataset_version": "train_v000",
        "purpose": "train",
        "parent_version": None,
        "base_validation_version": None,
        "created_at": datetime.now(UTC).isoformat(),
        "n_images": 5,
        "application_version": "0.1.0",
        "status": "RELEASED",
        "is_dummy": True,
        "comment": "学習処理のデバッグ用合成データ。実データの精度評価には使用しない。",
        "generator": "scripts/generate_dummy_dataset.py",
        "generator_version": 1,
        "seed": seed,
        "image_size": [512, 512],
        "image_dtype": "uint8",
        "mask_dtype": "uint16",
        "instances_per_image": 25,
        "numpy_version": np.__version__,
    }
    (output / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "workspace/datasets/train_v000",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"保存先が既に存在します。別の --output を指定してください: {args.output}")
    generate(args.output, args.seed)
    print(f"Created train_v000: {args.output.resolve()} (5 images + 5 masks)")


if __name__ == "__main__":
    main()
