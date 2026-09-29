"""ホームに表示するアプリ状態の集計。"""

from dataclasses import dataclass
from datetime import datetime

from ..services.backend import Backend
from .jobs import JobManager


@dataclass(frozen=True)
class HomeSummary:
    """ホーム画面に必要な表示用データ。"""

    unconfirmed_changes: int
    unassigned_items: int
    data_items: int
    data_errors: int
    latest_train: str
    latest_validation: str
    running_experiment: str | None
    running_epoch: int
    running_epochs: int
    completed_experiments: int
    experiment_count: int
    unevaluated_candidates: int
    candidate_count: int
    released_count: int
    routing: dict[str, str | None]
    recent: tuple[tuple[datetime, str], ...]
    queue_waiting: int = 0
    queue_running: bool = False


def build_home_summary(backend: Backend, jobs: JobManager, queue_controller=None) -> HomeSummary:
    """Backend とジョブ状態からホーム画面用の値を集計する。"""
    train_changes = backend.summarize_working_changes("train")
    validation_changes = backend.summarize_working_changes("val")
    changes = sum(
        int(data[key])
        for data in (train_changes, validation_changes)
        for key in ("added", "removed", "changed")
    )
    items = backend.get_working_items()
    unassigned = sum(item.usage == "unassigned" for item in items)
    errors = len(backend.validate_all_working_items().errors)
    experiments = backend.list_experiments()
    active = next(iter(jobs.training_jobs), None)
    experiment_id = getattr(active, "experiment_id", None) or (
        active.key.split(":", 1)[1] if active and active.key else None
    )
    try:
        experiment = backend.get_experiment(experiment_id) if experiment_id else None
    except (KeyError, ValueError):
        experiment = None
    candidates = backend.list_candidates()
    return HomeSummary(
        unconfirmed_changes=changes,
        unassigned_items=unassigned,
        data_items=len(items),
        data_errors=errors,
        latest_train=next(
            (item.version for item in reversed(backend.list_dataset_versions("train"))), "未確定"
        ),
        latest_validation=next(
            (item.version for item in reversed(backend.list_dataset_versions("val"))), "未確定"
        ),
        running_experiment=experiment_id,
        running_epoch=experiment.current_epoch if experiment else 0,
        running_epochs=experiment.total_epochs if experiment else 0,
        completed_experiments=sum(item.status == "completed" for item in experiments),
        experiment_count=len(experiments),
        unevaluated_candidates=sum(not item.evaluations for item in candidates),
        candidate_count=len(candidates),
        released_count=len(backend.list_released_models()),
        routing=backend.get_routing(),
        recent=(),
        queue_waiting=sum(item.status == "queued" for item in backend.list_training_queue()),
        queue_running=bool(queue_controller and queue_controller.executing),
    )
