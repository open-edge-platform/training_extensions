# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MLflow experiment tracking integration for the benchmark runner."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from otx.benchmark.experiment import ExperimentResult
    from otx.benchmark.manifest import Experiment

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
        """MLflow experiment name: ``otx-benchmark/{branch}/{trigger}``."""
        branch = self.branch or get_git_branch()
        return f"otx-benchmark/{branch}/{self.trigger}"


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


def _get_otx_version() -> str:
    """Return the installed OTX version string."""
    try:
        from otx import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _get_accelerator_info(accelerator: str) -> str:
    """Best-effort hardware description for the current accelerator."""
    try:
        if accelerator == "gpu":
            return (
                subprocess.check_output(  # noqa: S603
                    ["nvidia-smi", "-L"],  # noqa: S607
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        if accelerator == "xpu":
            raw = (
                subprocess.check_output(  # noqa: S603
                    ["xpu-smi", "discovery", "--dump", "1,2"],  # noqa: S607
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            return "\n".join(ret.replace('"', "").replace(",", " : ") for ret in raw.split("\n")[1:])
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
# MLflow tracker
# ---------------------------------------------------------------------------


@dataclass
class RunTags:
    """Structured tags logged with every MLflow run."""

    task: str
    model: str
    dataset: str
    scenario: str
    seed: str
    size_tier: str
    otx_version: str
    git_sha: str
    branch: str
    accelerator: str
    accelerator_info: str
    machine_name: str
    cpu_info: str
    status: str  # "success" | "failed"
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
            "otx_version": self.otx_version,
            "git_sha": self.git_sha,
            "branch": self.branch,
            "accelerator": self.accelerator,
            "accelerator_info": self.accelerator_info,
            "machine_name": self.machine_name,
            "cpu_info": self.cpu_info,
            "status": self.status,
        }
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
        self._otx_version = _get_otx_version()
        self._accelerator_info = _get_accelerator_info(config.accelerator)
        self._machine_name = platform.node()
        self._cpu_info = _get_cpu_info()

    # -- setup -------------------------------------------------------------

    def setup(self) -> None:
        """Configure MLflow tracking URI and experiment."""
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

    # -- logging -----------------------------------------------------------

    def log_run(
        self,
        result: ExperimentResult,
        experiment: Experiment,
        *,
        size_tier: str = "",
    ) -> None:
        """Log one ``(experiment, seed)`` result as an MLflow Run.

        Args:
            result: An :class:`ExperimentResult` instance.
            experiment: The :class:`Experiment` manifest entry.
            size_tier: Size tier of the dataset (for tagging).
        """
        run_name = f"{result.task}/{result.model}/{result.dataset}/{result.scenario}/{result.seed}"

        # Build tags
        extra_tags: dict[str, str] = {}
        # Log scenario overrides as tags for MLflow query
        if experiment.scenario.overrides:
            for key, value in experiment.scenario.overrides.items():
                short_key = key.rsplit(".", 1)[-1]  # e.g. "lr" from "model.init_args.optimizer.init_args.lr"
                extra_tags[f"override.{short_key}"] = str(value)

        tags = RunTags(
            task=result.task,
            model=result.model,
            dataset=result.dataset,
            scenario=result.scenario,
            seed=str(result.seed),
            size_tier=size_tier,
            otx_version=self._otx_version,
            git_sha=self._git_sha,
            branch=self._git_branch,
            accelerator=self.config.accelerator,
            accelerator_info=self._accelerator_info,
            machine_name=self._machine_name,
            cpu_info=self._cpu_info,
            status="success" if result.success else "failed",
            extra=extra_tags,
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(tags.as_dict())

            if result.success:
                # Log all metrics from all phases
                all_metrics = result.all_metrics()
                # MLflow metric keys must be valid; filter out non-numeric values
                numeric_metrics = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float))}
                if numeric_metrics:
                    mlflow.log_metrics(numeric_metrics)

                # Log per-phase wall times as separate metrics
                for phase in result.phases:
                    if phase.wall_time > 0:
                        safe_key = phase.phase.replace("/", "_")
                        mlflow.log_metrics({f"phase_{safe_key}_wall_time": phase.wall_time})
            elif result.error:
                # Log failure info
                mlflow.set_tag("error", result.error[:250])  # MLflow tag value limit

            logger.debug("Logged MLflow run %s (id=%s)", run_name, run.info.run_id)

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
        matching = [e for e in experiments if e.name.startswith(f"otx-benchmark/{br}/")]
        if not matching:
            return None
        experiment_ids = [e.experiment_id for e in matching]

        filter_string = (
            f"tags.branch = '{br}' "
            f"AND tags.model = '{model}' "
            f"AND tags.dataset = '{dataset}' "
            f"AND tags.scenario = '{scenario}' "
            f"AND tags.accelerator = '{acc}' "
            f"AND tags.status = 'success'"
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
        benchmark_exps = [e for e in experiments if e.name.startswith("otx-benchmark/")]

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
