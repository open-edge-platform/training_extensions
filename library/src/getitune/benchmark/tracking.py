# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MLflow experiment tracking integration for the benchmark runner."""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess  # nosec B404
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from getitune.benchmark.experiment import ExperimentResult
    from getitune.benchmark.manifest import Experiment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tracking configuration
# ---------------------------------------------------------------------------


@dataclass
class TrackingConfig:
    """Settings for MLflow tracking."""

    tracking_uri: str = "./mlruns"
    branch: str = ""
    trigger: str = "manual"
    accelerator: str = "gpu"
    baseline_branch: str = "develop"

    @property
    def experiment_name(self) -> str:
        """MLflow experiment name: ``getitune-benchmark/{branch}/{trigger}``."""
        branch = self.branch or get_git_branch()
        return f"getitune-benchmark/{branch}/{self.trigger}"


# ---------------------------------------------------------------------------
# Git / environment helpers
# ---------------------------------------------------------------------------


def get_git_sha() -> str:
    """Return the current short git commit SHA, or ``'unknown'``."""
    try:
        return (
            subprocess.check_output(  # noqa: S603
                ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return os.environ.get("GH_CTX_SHA", os.environ.get("GITHUB_SHA", "unknown"))


def get_git_branch() -> str:
    """Return the current git branch name, or ``'unknown'``."""
    try:
        return (
            subprocess.check_output(  # noqa: S603
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return os.environ.get("GITHUB_REF_NAME", "unknown")


# Backwards-compatible aliases (deprecated, prefer the public names above).
_get_git_sha = get_git_sha
_get_git_branch = get_git_branch


def _get_getitune_version() -> str:
    """Return the installed GetiTune version string."""
    try:
        from getitune import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _get_accelerator_info(accelerator: str) -> str:
    """Best-effort hardware description for the current accelerator.

    Multi-line tool output is flattened to a single line so the value stays
    readable in MLflow's run/experiment tag columns and CSV exports.
    """
    try:
        if accelerator == "gpu":
            raw = (
                subprocess.check_output(  # noqa: S603
                    ["nvidia-smi", "-L"],  # noqa: S607
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            # Drop the noisy "(UUID: GPU-xxxx-...)" suffix from each line.
            cleaned = re.sub(r"\s*\(UUID:[^)]*\)", "", raw)
            return cleaned.replace("\n", " | ")
        if accelerator == "xpu":
            raw = (
                subprocess.check_output(  # noqa: S603
                    ["xpu-smi", "discovery", "--dump", "1,2"],  # noqa: S607
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            return " | ".join(ret.replace('"', "").replace(",", " : ") for ret in raw.split("\n")[1:])
    except Exception:
        logger.debug("Could not detect accelerator info for '%s'.", accelerator)
    return accelerator


def _get_cpu_info() -> str:
    """Return a human-readable CPU brand string."""
    try:
        from cpuinfo import get_cpu_info

        return get_cpu_info()["brand_raw"]
    except Exception:
        return platform.processor() or "unknown"


# ---------------------------------------------------------------------------
# Primary-metric lookup per task
# ---------------------------------------------------------------------------

# Maps the task string produced by the manifest/experiment to the metric key
# (as written by the executor) that represents the "headline" quality of the
# run. Mirroring this under the unified ``primary_metric`` key makes the run
# table sortable across heterogeneous tasks.
_PRIMARY_METRIC: dict[str, str] = {
    "classification/multi_class_cls": "training:val/accuracy",
    "classification/multi_label_cls": "training:val/accuracy",
    "classification/h_label_cls": "training:val/accuracy",
    "detection": "training:val/map_50",
    "instance_segmentation": "training:val/map_50",
    "rotated_detection": "training:val/map_50",
    "semantic_segmentation": "training:val/mIoU",
    "keypoint_detection": "training:val/PCK",
    "anomaly": "training:val/image_AUROC",
    "action_classification": "training:val/accuracy",
    "action_detection": "training:val/map_50",
    "visual_prompting": "training:val/Dice",
    "zero_shot_visual_prompting": "training:val/Dice",
}


def _primary_metric_key(task: str) -> str | None:
    """Return the primary metric key for *task* (exact match or task-family prefix)."""
    if task in _PRIMARY_METRIC:
        return _PRIMARY_METRIC[task]
    # fall back to the most specific prefix match
    for prefix, key in sorted(_PRIMARY_METRIC.items(), key=lambda kv: -len(kv[0])):
        if task.startswith(prefix):
            return key
    return None


# ---------------------------------------------------------------------------
# Metric-key rewriting
# ---------------------------------------------------------------------------


def _rewrite_metric_key(key: str) -> str:
    """Normalize metric keys into tidy, UI-groupable namespaces.

    - ``<phase>:e2e_time``                  -> ``time/<phase>/e2e``
    - ``<phase>:test/e2e_time``             -> ``time/<phase>/test_e2e``
    - ``<phase>:test/latency``              -> ``time/<phase>/test_latency``
    - ``training:train/iter_time`` …        -> ``time/train/iter``
    - ``training:validation/iter_time``     -> ``time/train/val_iter``
    - ``torch:test/iter_time``              -> ``time/test_torch/iter``
    - Everything else is returned unchanged.
    """
    if key.endswith(":e2e_time"):
        return f"time/{key.split(':', 1)[0]}/e2e"
    if key.endswith(":test/e2e_time"):
        return f"time/{key.split(':', 1)[0]}/test_e2e"
    if key.endswith(":test/latency"):
        return f"time/{key.split(':', 1)[0]}/test_latency"
    if key.endswith(":test/iter_time"):
        return f"time/{key.split(':', 1)[0]}/iter"
    if key == "training:train/iter_time":
        return "time/train/iter"
    if key == "training:validation/iter_time":
        return "time/train/val_iter"
    return key


# ---------------------------------------------------------------------------
# MLflow tracker
# ---------------------------------------------------------------------------


@dataclass
class RunTags:
    """Identity/status tags that differ per run.

    Host/branch/version metadata is intentionally *not* here — those values
    are identical for every run in a given benchmark invocation and are
    therefore promoted to experiment-level tags (see
    :meth:`BenchmarkTracker.setup`). Only fields that are actually needed
    for per-run filtering or that vary per run belong in this dataclass.
    """

    task: str
    model: str
    dataset: str
    scenario: str
    seed: str
    size_tier: str
    accelerator: str  # kept per-run because baseline resolution filters on it
    branch: str  # ditto
    status: str  # "success" | "failed"
    run_type: str = "seed"  # "seed" (leaf) or "rollup" (parent aggregate)
    primary_metric_name: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, str]:
        """Return all tags as a flat dictionary for MLflow."""
        tags = {
            "task": self.task,
            "model": self.model,
            "dataset": self.dataset,
            "scenario": self.scenario,
            "seed": self.seed,
            "size_tier": self.size_tier,
            "accelerator": self.accelerator,
            "branch": self.branch,
            "status": self.status,
            "run_type": self.run_type,
        }
        if self.primary_metric_name:
            tags["primary_metric_name"] = self.primary_metric_name
        tags.update(self.extra)
        return tags


class BenchmarkTracker:
    """Manages MLflow experiment lifecycle for benchmark runs.

    Usage::

        tracker = BenchmarkTracker(config)
        tracker.setup()

        for result in results:
            tracker.log_run(result, experiment)

        baselines = tracker.resolve_baselines(keys)
    """

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._experiment_id: str | None = None

        # Pre-compute immutable environment metadata once
        self._git_sha = get_git_sha()
        self._git_branch = config.branch or get_git_branch()
        self._getitune_version = _get_getitune_version()
        self._accelerator_info = _get_accelerator_info(config.accelerator)
        self._machine_name = platform.node()
        self._cpu_info = _get_cpu_info()

    # -- setup -------------------------------------------------------------

    def setup(self) -> None:
        """Configure MLflow tracking URI and experiment.

        Publishes host/branch/version metadata as **experiment-level** tags
        so that values which are identical for every run in this invocation
        are surfaced once — not repeated across every row of the run table.
        """
        mlflow.set_tracking_uri(self.config.tracking_uri)

        experiment_name = self.config.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self._experiment_id = mlflow.create_experiment(experiment_name)
            logger.info("Created MLflow experiment: %s (id=%s)", experiment_name, self._experiment_id)
        else:
            self._experiment_id = experiment.experiment_id
            logger.info("Using existing MLflow experiment: %s (id=%s)", experiment_name, self._experiment_id)

        mlflow.set_experiment(experiment_id=self._experiment_id)

        # Environment metadata: set once at experiment level.
        try:
            client = mlflow.tracking.MlflowClient(self.config.tracking_uri)
            env_tags = {
                "getitune_version": self._getitune_version,
                "git_sha": self._git_sha,
                "branch": self._git_branch,
                "accelerator": self.config.accelerator,
                "accelerator_info": self._accelerator_info,
                "machine_name": self._machine_name,
                "cpu_info": self._cpu_info,
                "trigger": self.config.trigger,
            }
            exp_id = str(self._experiment_id)
            for k, v in env_tags.items():
                if v:
                    client.set_experiment_tag(exp_id, k, v)
        except Exception:
            logger.debug("Could not set experiment-level tags.", exc_info=True)

    # -- logging -----------------------------------------------------------

    @staticmethod
    def _build_run_name(*, model: str, dataset: str, scenario: str, seed: int | str) -> str:
        """Compact run name used in the MLflow UI ('Name' column).

        ``scenario`` is omitted when it is the default ("default") to keep
        names short for the common case. The noisy ``task/...`` path is
        dropped entirely because task is already a filterable tag.
        """
        scenario_part = "" if scenario == "default" else f" [{scenario}]"
        return f"{model} · {dataset}{scenario_part} · s{seed}"

    @staticmethod
    def _build_parent_run_name(*, model: str, dataset: str, scenario: str) -> str:
        """Run name for the per-experiment rollup (parent of seed runs)."""
        scenario_part = "" if scenario == "default" else f" [{scenario}]"
        return f"{model} · {dataset}{scenario_part}"

    def start_parent_run(self, experiment: Experiment) -> object:
        """Open a rollup run that will hold the per-seed nested children.

        Returns the active MLflow run object. The caller is responsible
        for closing it (use it as a context manager).
        """
        run_name = self._build_parent_run_name(
            model=experiment.model.name,
            dataset=experiment.dataset_name,
            scenario=experiment.scenario.name,
        )
        return mlflow.start_run(
            run_name=run_name,
            tags={
                "task": experiment.task,
                "model": experiment.model.name,
                "dataset": experiment.dataset_name,
                "scenario": experiment.scenario.name,
                "run_type": "rollup",
                "branch": self._git_branch,
                "accelerator": self.config.accelerator,
            },
        )

    @staticmethod
    def set_parent_run_duration(run_id: str, total_seconds: float) -> None:
        """Overwrite a rollup run's ``end_time`` so Duration = sum of seed wall times.

        By default MLflow measures Duration as ``end_time - start_time`` around
        the (fast) logging window, which makes rollup rows show sub-second
        durations even though the underlying seeds took many minutes. We
        post-process the parent run here so the MLflow UI reports a Duration
        equal to the sum of the seed wall-clock times.
        """
        if total_seconds <= 0:
            return
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            run = client.get_run(run_id)
            start_ms = int(run.info.start_time)
            end_ms = start_ms + round(total_seconds * 1000)
            # set_terminated with an explicit end_time rewrites the stored value,
            # which is what the UI uses to compute the Duration column.
            client.set_terminated(run_id, status=run.info.status, end_time=end_ms)
        except Exception:
            logger.debug("Could not adjust rollup run duration.", exc_info=True)

    def log_aggregate(
        self,
        results: list[ExperimentResult],
        experiment: Experiment,
    ) -> None:
        """Log seed-aggregated summary metrics onto the currently active run.

        Expected to be invoked while a rollup parent run is active (see
        :meth:`start_parent_run`). Emits:

        - ``num_seeds`` / ``num_successful_seeds``
        - ``primary_metric/mean`` / ``primary_metric/std`` (when the task
          has a known primary metric and at least one successful seed)
        - ``primary_metric/min`` / ``primary_metric/max`` for range context
        """
        import statistics

        mlflow.log_metric("num_seeds", float(len(results)))
        successful = [r for r in results if r.success]
        mlflow.log_metric("num_successful_seeds", float(len(successful)))

        primary_key = _primary_metric_key(experiment.task)
        if primary_key is None or not successful:
            return

        values = [
            float(r.all_metrics()[primary_key])
            for r in successful
            if primary_key in r.all_metrics() and isinstance(r.all_metrics()[primary_key], (int, float))
        ]
        if not values:
            return

        mlflow.set_tag("primary_metric_name", primary_key)
        mlflow.log_metric("primary_metric/mean", statistics.mean(values))
        mlflow.log_metric("primary_metric/min", min(values))
        mlflow.log_metric("primary_metric/max", max(values))
        if len(values) > 1:
            mlflow.log_metric("primary_metric/std", statistics.stdev(values))

    def log_run(
        self,
        result: ExperimentResult,
        experiment: Experiment,
        *,
        size_tier: str = "",
        nested: bool = False,
    ) -> None:
        """Log one ``(experiment, seed)`` result as an MLflow Run.

        Args:
            result: An :class:`ExperimentResult` instance.
            experiment: The :class:`Experiment` manifest entry.
            size_tier: Size tier of the dataset (for tagging).
            nested: When ``True``, the run is opened as a child of the
                currently-active MLflow run (used for per-seed rollups).
        """
        run_name = self._build_run_name(
            model=result.model,
            dataset=result.dataset,
            scenario=result.scenario,
            seed=result.seed,
        )

        # Scenario overrides -> queryable tags.
        extra_tags: dict[str, str] = {}
        if experiment.scenario.overrides:
            for key, value in experiment.scenario.overrides.items():
                short_key = key.rsplit(".", 1)[-1]
                extra_tags[f"override.{short_key}"] = str(value)

        primary_key = _primary_metric_key(result.task)

        tags = RunTags(
            task=result.task,
            model=result.model,
            dataset=result.dataset,
            scenario=result.scenario,
            seed=str(result.seed),
            size_tier=size_tier,
            accelerator=self.config.accelerator,
            branch=self._git_branch,
            status="success" if result.success else "failed",
            run_type="seed",
            primary_metric_name=primary_key or "",
            extra=extra_tags,
        )

        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            mlflow.set_tags(tags.as_dict())

            if result.success:
                all_metrics = result.all_metrics()
                numeric_metrics: dict[str, float] = {}
                for k, v in all_metrics.items():
                    if not isinstance(v, (int, float)):
                        continue
                    numeric_metrics[_rewrite_metric_key(k)] = float(v)

                if numeric_metrics:
                    mlflow.log_metrics(numeric_metrics)

                # Headline metric: mirror the task's primary metric under a
                # stable key so the UI table can be sorted across tasks.
                if primary_key and primary_key in all_metrics:
                    value = all_metrics[primary_key]
                    if isinstance(value, (int, float)):
                        mlflow.log_metric("primary_metric", float(value))
            elif result.error:
                mlflow.set_tag("error", result.error[:250])

            logger.info("Logged MLflow run %s (id=%s)", run_name, run.info.run_id)

        # Adjust the run's end time so the "Duration" column in the MLflow UI
        # reflects the actual experiment wall time, not just the logging time.
        total_wall_time = sum(p.wall_time for p in result.phases)
        if total_wall_time > 0:
            try:
                client = mlflow.tracking.MlflowClient(self.config.tracking_uri)
                start_ms = run.info.start_time
                end_ms = start_ms + int(total_wall_time * 1000)
                client.set_terminated(run.info.run_id, end_time=end_ms)
            except Exception:
                logger.debug("Could not update run end time for duration display.")

    # -- baseline resolution -----------------------------------------------

    def resolve_baseline(
        self,
        *,
        model: str,
        dataset: str,
        scenario: str = "default",
        accelerator: str | None = None,
        branch: str | None = None,
    ) -> dict[str, float] | None:
        """Fetch the most recent successful baseline from MLflow.

        Queries for the latest run on the given branch matching the
        ``(model, dataset, scenario, accelerator)`` combination.

        Returns the run's metrics dict, or ``None`` if no baseline exists.
        """
        acc = accelerator or self.config.accelerator
        br = branch or self.config.baseline_branch

        client = mlflow.tracking.MlflowClient(self.config.tracking_uri)

        # Search across ALL trigger-type experiments for the baseline branch.
        # This avoids missing baselines that were logged under a different
        # trigger (e.g. "weekly" baselines when running a "manual" trigger).
        experiments = client.search_experiments()
        matching = [e for e in experiments if e.name.startswith(f"getitune-benchmark/{br}/")]
        if not matching:
            return None
        experiment_ids = [e.experiment_id for e in matching]

        filter_string = (
            f"tags.branch = '{br}' "
            f"AND tags.model = '{model}' "
            f"AND tags.dataset = '{dataset}' "
            f"AND tags.scenario = '{scenario}' "
            f"AND tags.accelerator = '{acc}' "
            f"AND tags.status = 'success' "
            # Exclude rollup parents; baselines are individual seed runs.
            f"AND tags.run_type = 'seed'"
        )

        runs = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None

        return dict(runs[0].data.metrics)

    def resolve_baselines_for_results(
        self,
        results: list[ExperimentResult],
    ) -> dict[str, dict[str, float] | None]:
        """Resolve baselines for all successful results.

        Returns a dict keyed by ``"task/model/dataset/scenario"`` → baseline metrics.
        """
        baselines: dict[str, dict[str, float] | None] = {}
        seen: set[str] = set()

        for result in results:
            if not result.success:
                continue
            key = f"{result.task}/{result.model}/{result.dataset}/{result.scenario}"
            if key in seen:
                continue
            seen.add(key)

            baselines[key] = self.resolve_baseline(
                model=result.model,
                dataset=result.dataset,
                scenario=result.scenario,
            )

        return baselines

    # -- retention / cleanup -----------------------------------------------

    def purge_old_runs(
        self,
        *,
        max_age_days: int = 90,
        protect_branches: tuple[str, ...] = ("develop", "main"),
        dry_run: bool = False,
    ) -> int:
        """Delete MLflow runs older than *max_age_days*.

        Runs on protected branches (``develop``, ``main``) are never deleted
        because they serve as baselines.  Only runs tagged with a non-protected
        branch (i.e. feature branches, PR branches) are eligible for pruning.

        Args:
            max_age_days: Runs older than this are candidates for deletion.
            protect_branches: Branches whose runs are never deleted.
            dry_run: If ``True``, log what would be deleted without acting.

        Returns:
            Number of runs deleted (or that *would* be deleted in dry-run mode).
        """
        from datetime import datetime, timezone

        client = mlflow.tracking.MlflowClient(self.config.tracking_uri)

        # Find all otx-benchmark experiments
        experiments = client.search_experiments()
        benchmark_exps = [e for e in experiments if e.name.startswith("getitune-benchmark/")]

        if not benchmark_exps:
            logger.info("No benchmark experiments found — nothing to purge.")
            return 0

        cutoff_ms = int((datetime.now(tz=timezone.utc).timestamp() - max_age_days * 86400) * 1000)
        experiment_ids = [e.experiment_id for e in benchmark_exps]

        # Search for old runs on non-protected branches
        protect_filter = " AND ".join(f"tags.branch != '{b}'" for b in protect_branches)
        filter_string = protect_filter if protect_filter else ""

        runs = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            order_by=["attributes.start_time ASC"],
            max_results=5000,
        )

        deleted = 0
        for run in runs:
            start_time = run.info.start_time  # milliseconds since epoch
            if start_time >= cutoff_ms:
                # Runs are ordered ASC by start_time — once we pass the cutoff,
                # all remaining runs are newer.
                break

            branch_tag = run.data.tags.get("branch", "")
            if branch_tag in protect_branches:
                continue

            if dry_run:
                logger.info(
                    "Would delete run %s (branch=%s, started=%s)",
                    run.info.run_id,
                    branch_tag,
                    datetime.fromtimestamp(start_time / 1000, tz=timezone.utc).isoformat(),
                )
            else:
                client.delete_run(run.info.run_id)
                logger.debug("Deleted run %s (branch=%s)", run.info.run_id, branch_tag)
            deleted += 1

        action = "Would delete" if dry_run else "Deleted"
        logger.info("%s %d old run(s) (cutoff: %d days).", action, deleted, max_age_days)
        return deleted
