"""偽アダプタ等で共有する CV・OOF・最終学習ループ。"""

from __future__ import annotations

import csv
import random
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from foam_cell_analysis.evaluation.ap import METRIC, aggregate, match_counts
from foam_cell_analysis.training.adapters.base import ModelAdapter, Sample
from foam_cell_analysis.training.augmentation import apply_profile
from foam_cell_analysis.training.checkpoints import (
    artifact_record,
    create_selected,
    save_state,
    write_checkpoint,
    write_checkpoint_metadata,
    write_label,
)
from foam_cell_analysis.training.preprocessing import normalize_image
from foam_cell_analysis.training.protocol import (
    write_partial_result,
    write_result,
)


def _seed_everything(seed: int) -> np.random.Generator:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)


def _sample_for_training(
    item_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    normalization: dict[str, Any] | None,
    profile: dict[str, Any],
    adapter: ModelAdapter,
    model_config: dict[str, Any],
    rng: np.random.Generator,
    *,
    training: bool,
) -> Sample:
    normalized = normalize_image(image, normalization)
    transformed, labels = (
        apply_profile(normalized, mask, profile, rng)
        if training
        else (normalized, np.asarray(mask).copy())
    )
    sample = Sample(item_id, transformed, labels)
    return adapter.prepare_sample(sample, training=training, rng=rng)


def _save_counts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        item_ids=np.asarray([row["item_id"] for row in rows]),
        tp=np.stack([row["tp"] for row in rows]),
        fp=np.stack([row["fp"] for row in rows]),
        fn=np.stack([row["fn"] for row in rows]),
    )


def _output_files(run_dir: Path) -> list[Path]:
    files = []
    for folder in ("eval", "oof", "checkpoints"):
        base = run_dir / folder
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file())
    return sorted(files)


def _result_artifacts(run_dir: Path, *, include_final: bool) -> list[dict[str, Any]]:
    hashes: dict[tuple[int, int, int, int], str] = {}
    artifacts = [artifact_record(path, run_dir, hashes) for path in _output_files(run_dir)]
    if include_final:
        final_path = run_dir / "checkpoints" / "final.pt"
        if final_path.is_file() and all(row["path"] != "checkpoints/final.pt" for row in artifacts):
            artifacts.append(artifact_record(final_path, run_dir, hashes))
    return artifacts


def _sample_stream(
    item_ids: list[str],
    arrays: Any,
    normalization: dict[str, Any] | None,
    profile: dict[str, Any],
    adapter: ModelAdapter,
    model_config: dict[str, Any],
    rng: np.random.Generator,
    *,
    training: bool,
) -> Iterator[Sample]:
    """変換済み画像を保持せず、必要な 1 枚ずつ作る。"""
    for item_id in item_ids:
        image, mask = arrays[item_id]
        yield _sample_for_training(
            item_id,
            image,
            mask,
            normalization,
            profile,
            adapter,
            model_config,
            rng,
            training=training,
        )


def common_candidate_epochs(validation_epochs: list[set[int]]) -> set[int]:
    """全 fold が実際に検証したエポックの共通部分を返す。"""
    candidates = set.intersection(*validation_epochs) if validation_epochs else set()
    if not candidates:
        raise ValueError("候補エポックの共通部分が空です")
    return candidates


def execute_training(
    run_dir: str | Path,
    spec: dict[str, Any],
    preflight: dict[str, Any],
    adapter_factory: Callable[[], ModelAdapter],
    device: Any,
    emit: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """fold CV から最終学習・manifest 作成まで実行する。"""
    run_path = Path(run_dir).resolve()
    arrays = preflight["arrays"]
    folds = preflight["folds"]
    resolved = preflight["resolved"]
    config = spec["config"]
    training_config = config["training"]
    checkpoint_config = config["checkpoint"]
    model_config = dict(config["model"])
    model_config["eval_params"] = spec["eval_params"]
    epochs = int(training_config["epochs"])
    validation_interval = int(checkpoint_config["validation_interval"])
    save_every = int(checkpoint_config["save_every"])
    save_fold_models = bool(checkpoint_config.get("save_fold_models", True))
    profile = spec["augmentation_profile"]["value"]
    model_config["augmentation_profile"] = profile
    if model_config.get("type") == "cellpose":
        model_config["_shared_array_cache"] = arrays.store._cache
    normalization = spec.get("preprocessing")
    dataset_store_items = {item.item_id: item for item in preflight["items"]}
    classifications = {item_id: dataset_store_items[item_id].classification for item_id in arrays}
    class_names = sorted({name for name in classifications.values() if name})
    validation_epochs = set(range(validation_interval, epochs + 1, validation_interval))
    validation_epochs.add(epochs)
    validation_batch_size = 4
    fold_results: dict[int, dict[int, dict[str, dict[str, Any]]]] = {}
    fold_records: dict[int, dict[str, Any]] = {}

    for fold in folds:
        seed = int(spec["seeds"]["folds"][str(fold)])
        rng = _seed_everything(seed)
        adapter = adapter_factory()
        adapter.build(model_config, device)
        adapter.make_optimizer(training_config)
        train_ids = resolved["training_item_ids"][str(fold)]
        validation_ids = resolved["validation_item_ids"][str(fold)]
        emit("phase", phase="cv", fold=fold, total_epochs=epochs)
        best_score = float("-inf")
        best_epoch: int | None = None
        wait = 0
        last_epoch = 0
        stopped_early = False
        fold_results[fold] = {}
        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            epoch_ids = adapter.epoch_item_ids(list(train_ids), model_config, rng)
            samples = _sample_stream(
                epoch_ids,
                arrays,
                normalization,
                profile,
                adapter,
                model_config,
                rng,
                training=True,
            )
            lr = adapter.lr(epoch, epochs, float(training_config["learning_rate"]))
            loss = adapter.train_one_epoch(samples, lr)
            emit("epoch", phase="cv", fold=fold, epoch=epoch, loss=loss, lr=lr)
            do_validate = epoch in validation_epochs
            periodic = epoch % save_every == 0
            checkpoint_source = None
            if save_fold_models and (periodic or do_validate):
                if periodic:
                    checkpoint_source = write_checkpoint(
                        adapter,
                        run_path / "checkpoints" / f"fold_{fold}" / f"epoch_{epoch:03d}.pt",
                        model_type=str(model_config["type"]),
                        fold=fold,
                        epoch=epoch,
                        kind="periodic",
                        run_id=spec["run_id"],
                    )
                    emit(
                        "checkpoint",
                        fold=fold,
                        epoch=epoch,
                        kind="periodic",
                        path=checkpoint_source.relative_to(run_path).as_posix(),
                    )
                else:
                    checkpoint_source = save_state(
                        adapter, run_path / "tmp_ckpt" / f"fold_{fold}" / f"epoch_{epoch:03d}.pt"
                    )
            if not do_validate:
                continue
            count_rows = []
            for start in range(0, len(validation_ids), validation_batch_size):
                batch_ids = validation_ids[start : start + validation_batch_size]
                validation_samples = list(
                    _sample_stream(
                        batch_ids,
                        arrays,
                        normalization,
                        profile,
                        adapter,
                        model_config,
                        rng,
                        training=False,
                    )
                )
                predictions = adapter.predict([sample.image for sample in validation_samples])
                for sample, prediction in zip(validation_samples, predictions, strict=True):
                    tp, fp, fn = match_counts(sample.mask, prediction)
                    row = {"item_id": sample.item_id, "tp": tp, "fp": fp, "fn": fn}
                    count_rows.append(row)
                    write_label(
                        run_path
                        / "tmp_pred"
                        / f"fold_{fold}"
                        / f"epoch_{epoch:03d}"
                        / sample.item_id,
                        prediction,
                    )
            _save_counts(run_path / "eval" / f"fold_{fold}" / f"epoch_{epoch:03d}.npz", count_rows)
            fold_results[fold][epoch] = {row["item_id"]: row for row in count_rows}
            validation = aggregate(count_rows, classifications, class_names)
            score = float(validation["ap"])
            emit("val", fold=fold, epoch=epoch, ap=score, n_images=len(count_rows))
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                wait = 0
            else:
                wait += epoch - (best_epoch or epoch)
            if fold == folds[-1]:
                complete_folds = [
                    previous for previous in folds if epoch in fold_results.get(previous, {})
                ]
                if complete_folds == folds:
                    combined = [
                        row for previous in folds for row in fold_results[previous][epoch].values()
                    ]
                    oof_value = aggregate(combined, classifications, class_names)
                    per_fold = {
                        str(previous): float(
                            aggregate(
                                list(fold_results[previous][epoch].values()),
                                classifications,
                                class_names,
                            )["ap"]
                        )
                        for previous in folds
                    }
                    emit(
                        "oof",
                        epoch=epoch,
                        ap=float(oof_value["ap"]),
                        per_fold=per_fold,
                        n_images=oof_value["n_images"],
                    )
            patience = int(training_config.get("early_stopping", {}).get("patience", 10))
            early_enabled = bool(training_config.get("early_stopping", {}).get("enabled", False))
            if early_enabled and best_epoch is not None and epoch - best_epoch >= patience:
                stopped_early = epoch < epochs
                break
        fold_records[fold] = {
            "last_epoch": last_epoch,
            "best_epoch": best_epoch,
            "best_ap": None if best_epoch is None else best_score,
            "early_stopped": stopped_early,
        }
        emit(
            "fold_done",
            fold=fold,
            last_epoch=last_epoch,
            early_stopped=stopped_early,
            best_epoch=best_epoch,
        )

    candidates = common_candidate_epochs([set(fold_results[fold]) for fold in folds])
    oof_rows = {}
    for epoch in sorted(candidates):
        rows = [row for fold in folds for row in fold_results[fold][epoch].values()]
        oof_rows[epoch] = aggregate(rows, classifications, class_names)
    selected_epoch = min(candidates, key=lambda epoch: (-float(oof_rows[epoch]["ap"]), epoch))
    selected = oof_rows[selected_epoch]
    per_class_event = {
        name: [value["ap"], value["n_images"]] for name, value in selected["per_class"].items()
    }
    emit(
        "selected",
        epoch=selected_epoch,
        ap=float(selected["ap"]),
        per_class=per_class_event,
    )

    item_folds = {item_id: int(value) for item_id, value in resolved["fold_assignments"].items()}
    per_image = selected["per_image"]
    (run_path / "oof").mkdir(parents=True, exist_ok=True)
    with (run_path / "oof" / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["item_id", "fold", "classification", "ap", "tp", "fp", "fn"],
        )
        writer.writeheader()
        for item_id in sorted(per_image):
            counts = fold_results[item_folds[item_id]][selected_epoch][item_id]
            writer.writerow(
                {
                    "item_id": item_id,
                    "fold": item_folds[item_id],
                    "classification": classifications[item_id] or "",
                    "ap": per_image[item_id],
                    "tp": int(counts["tp"][0]),
                    "fp": int(counts["fp"][0]),
                    "fn": int(counts["fn"][0]),
                }
            )
            source = (
                run_path
                / "tmp_pred"
                / f"fold_{item_folds[item_id]}"
                / f"epoch_{selected_epoch:03d}"
            )
            label = next(source.glob(item_id + ".*"))
            destination = run_path / "oof" / label.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            label.replace(destination)

    if save_fold_models:
        for fold in folds:
            source_temp = run_path / "tmp_ckpt" / f"fold_{fold}" / f"epoch_{selected_epoch:03d}.pt"
            source_persistent = (
                run_path / "checkpoints" / f"fold_{fold}" / f"epoch_{selected_epoch:03d}.pt"
            )
            source = source_temp if source_temp.exists() else source_persistent
            target = run_path / "checkpoints" / f"fold_{fold}" / "selected.pt"
            create_selected(source, target)
            write_checkpoint_metadata(
                target,
                model_type=str(model_config["type"]),
                fold=fold,
                epoch=selected_epoch,
                kind="selected",
                run_id=spec["run_id"],
            )
            emit(
                "checkpoint",
                fold=fold,
                epoch=selected_epoch,
                kind="selected",
                path=target.relative_to(run_path).as_posix(),
            )
    shutil.rmtree(run_path / "tmp_pred", ignore_errors=True)
    shutil.rmtree(run_path / "tmp_ckpt", ignore_errors=True)
    partial = {
        "selected_epoch": selected_epoch,
        "oof": selected,
        "metric": spec["metric"],
        "folds": {str(key): value for key, value in fold_records.items()},
        "per_image": per_image,
        "artifacts": _result_artifacts(run_path, include_final=False),
    }
    write_partial_result(run_path, partial)
    emit("partial_result", path="result.partial.json")

    final_seed = int(spec["seeds"]["final"])
    rng = _seed_everything(final_seed)
    final_adapter = adapter_factory()
    final_adapter.build(model_config, device)
    final_adapter.make_optimizer(training_config)
    emit("phase", phase="final", fold=None, total_epochs=selected_epoch)
    retained_ids = list(resolved["final_training_item_ids"])
    for epoch in range(1, selected_epoch + 1):
        epoch_ids = final_adapter.epoch_item_ids(list(retained_ids), model_config, rng)
        samples = _sample_stream(
            epoch_ids,
            arrays,
            normalization,
            profile,
            final_adapter,
            model_config,
            rng,
            training=True,
        )
        lr = final_adapter.lr(epoch, epochs, float(training_config["learning_rate"]))
        loss = final_adapter.train_one_epoch(samples, lr)
        emit("epoch", phase="final", fold=None, epoch=epoch, loss=loss, lr=lr)
    final_path = write_checkpoint(
        final_adapter,
        run_path / "checkpoints" / "final.pt",
        model_type=str(model_config["type"]),
        fold=None,
        epoch=selected_epoch,
        kind="final",
        run_id=spec["run_id"],
    )
    emit(
        "checkpoint",
        fold=None,
        epoch=selected_epoch,
        kind="final",
        path=final_path.relative_to(run_path).as_posix(),
    )
    result = {
        "selected_epoch": selected_epoch,
        "oof": selected,
        "metric": spec["metric"] or METRIC,
        "folds": {str(key): value for key, value in fold_records.items()},
        "per_image": per_image,
        "artifacts": _result_artifacts(run_path, include_final=True),
    }
    write_result(run_path, result)
    emit("completed", path="result.json")
    return result
