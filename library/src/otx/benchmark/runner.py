# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark runner — core orchestration loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from otx.benchmark.catalog import DatasetCatalog, provision_datasets
from otx.benchmark.experiment import (
    ExperimentExecutor,
    ExperimentResult,
    PhaseResult,
    detect_resume_point,
)
from otx.benchmark.manifest import (
    BenchmarkManifest,
    Experiment,
    ManifestFilters,
    iter_experiments,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Ordered list of phases that the runner steps through.  Each phase maps
# to a method name on :class:`ExperimentExecutor`.
_PHASE_CHAIN: list[tuple[str, str]] = [
    ("train", "train"),
    ("test/torch", "test_torch"),
    ("export", "export"),
    ("test/export", "test_export"),
    ("optimize", "optimize"),
    ("test/optimize", "test_optimize"),
]

# Which phases are included for a given ``eval_upto`` value.
_EVAL_UPTO_GATES: dict[str, set[str]] = {
    "train": {"train", "test/torch"},
    "export": {"train", "test/torch", "export", "test/export"},
    "optimize": {"train", "test/torch", "export", "test/export", "optimize", "test/optimize"},
}

_MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """All settings needed to drive a benchmark run."""

    manifest_path: Path
    catalog_path: Path
    data_root: Path
    output_root: Path
    accelerator: str = "gpu"
    deterministic: bool | str = True  # True, False, or "warn" (PyTorch semantics)
    max_epochs: int | None = None
    num_seeds: int | None = None  # override manifest default
    eval_upto: str | None = None  # override manifest default
    filters: ManifestFilters = field(default_factory=ManifestFilters)

    # Tracking & reporting
    mlflow_tracking_uri: str = "./mlruns"
    branch: str = ""
    trigger: str = "manual"
    baseline_branch: str = "develop"
    enable_tracking: bool = True
    enable_report: bool = True

    # Disk management
    keep_checkpoints: bool = False  # If False, delete .ckpt/.xml/.bin after metrics are extracted

    # Ad-hoc overrides (from CLI --override / --train-kwarg flags)
    ad_hoc_overrides: dict[str, str] = field(default_factory=dict)
    ad_hoc_train_kwargs: dict[str, str] = field(default_factory=dict)

    # Rotation logic
    rotation_group: int | None = None  # If set, only run extended models in this group
    no_rotation: bool = False  # If True, skip rotation filtering entirely


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrates the full benchmark run.

    The runner iterates over experiments sequentially (one at a time) so that
    each experiment has exclusive GPU access, producing stable and
    reproducible metrics.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.results: list[ExperimentResult] = []
        self.failures: list[ExperimentResult] = []
        self._tracker = None  # type: ignore[assignment]

    def run(
        self,
        manifest: BenchmarkManifest,
        catalog: DatasetCatalog,
    ) -> tuple[list[ExperimentResult], list[ExperimentResult]]:
        """Execute all matching experiments.

        Returns ``(successes, failures)``.
        """
        # Set up MLflow tracking if enabled
        if self.config.enable_tracking:
            self._setup_tracking()

        eval_upto = self.config.eval_upto or manifest.defaults.eval_upto
        allowed_phases = _EVAL_UPTO_GATES.get(eval_upto, _EVAL_UPTO_GATES["optimize"])

        catalog_names = {e.name for e in catalog.all_entries()}

        # Build a lookup for dataset size tiers (for filtering and MLflow tags)
        size_tier_map: dict[str, str] = {}
        for entry in catalog.all_entries():
            size_tier_map[entry.name] = entry.size_tier

        experiments = list(
            iter_experiments(
                manifest,
                self.config.filters,
                catalog_names,
                size_tier_map=size_tier_map,
            )
        )

        # Apply rotation logic for extended models (§6.4.2).
        # When --rotation-group is set and --no-rotation is *not* set, only
        # extended-priority models whose hash falls into the given group are
        # kept.  Core models always run regardless of rotation.
        rotation_groups = manifest.defaults.rotation.get("extended_groups", 0)
        if self.config.rotation_group is not None and not self.config.no_rotation and rotation_groups > 0:
            group = self.config.rotation_group
            experiments = [
                exp
                for exp in experiments
                if exp.model.priority != "extended" or (hash(exp.model.name) % rotation_groups) == group
            ]
            logger.info(
                "Rotation group %d/%d: %d experiment(s) after filtering.",
                group,
                rotation_groups,
                len(experiments),
            )

        if not experiments:
            logger.warning("No experiments match the current filters.")
            return [], []

        if self.config.filters.dry_run:
            logger.info("Dry run: would execute %d experiments.", len(experiments))
            for exp in experiments:
                seeds = self.config.num_seeds or exp.num_seeds
                logger.info(
                    "  %s / %s / %s / %s  (%d seeds)",
                    exp.task,
                    exp.model.name,
                    exp.dataset_name,
                    exp.scenario.name,
                    seeds,
                )
            return [], []

        # Provision required datasets
        required_names = {exp.dataset_name for exp in experiments}
        required_entries = catalog.filter(names=required_names)
        logger.info("Provisioning %d dataset(s)…", len(required_entries))
        dataset_paths = provision_datasets(catalog, self.config.data_root, entries=required_entries)

        total = len(experiments)
        for idx, experiment in enumerate(experiments, 1):
            logger.info(
                "[%d/%d] %s / %s / %s (scenario=%s)",
                idx,
                total,
                experiment.task,
                experiment.model.name,
                experiment.dataset_name,
                experiment.scenario.name,
            )
            data_path = dataset_paths.get(experiment.dataset_name)
            if data_path is None:
                logger.error("Dataset '%s' not provisioned; skipping.", experiment.dataset_name)
                continue

            num_seeds = self.config.num_seeds or experiment.num_seeds
            for seed in range(num_seeds):
                result = self._run_single(experiment, seed, data_path, allowed_phases)

                # Log to MLflow
                if self._tracker is not None:
                    try:
                        self._tracker.log_run(
                            result,
                            experiment,
                            size_tier=size_tier_map.get(experiment.dataset_name, ""),
                        )
                    except Exception:
                        logger.warning("Failed to log run to MLflow.", exc_info=True)

                if result.success:
                    self.results.append(result)
                else:
                    self.failures.append(result)

            # Persist report after each experiment so partial results survive crashes
            if self.config.enable_report:
                self._generate_report(manifest)

        logger.info(
            "Benchmark complete. %d succeeded, %d failed.",
            len(self.results),
            len(self.failures),
        )

        # Cleanup heavy artifacts to save disk space
        if not self.config.keep_checkpoints:
            self._cleanup_checkpoints()

        return self.results, self.failures

    # -- tracking setup ----------------------------------------------------

    def _setup_tracking(self) -> None:
        """Initialize the MLflow tracker."""
        from otx.benchmark.tracking import BenchmarkTracker, TrackingConfig

        tracking_config = TrackingConfig(
            tracking_uri=self.config.mlflow_tracking_uri,
            branch=self.config.branch,
            trigger=self.config.trigger,
            accelerator=self.config.accelerator,
            baseline_branch=self.config.baseline_branch,
        )
        self._tracker = BenchmarkTracker(tracking_config)
        self._tracker.setup()
        logger.info("MLflow tracking enabled (uri=%s).", self.config.mlflow_tracking_uri)

    # -- report generation -------------------------------------------------

    def _generate_report(self, manifest: BenchmarkManifest) -> None:
        """Generate a Markdown + CSV report after the run completes."""
        try:
            from otx.benchmark.report import generate_report
            from otx.benchmark.tracking import get_git_branch, get_git_sha

            # Resolve baselines from MLflow
            baselines: dict[str, dict[str, float] | None] = {}
            if self._tracker is not None:
                try:
                    baselines = self._tracker.resolve_baselines_for_results(self.results)
                except Exception:
                    logger.warning("Failed to resolve baselines from MLflow.", exc_info=True)

            # Build criteria mapping by task
            criteria_by_task = {task_key: section.criteria for task_key, section in manifest.experiments.items()}

            branch = self.config.branch or get_git_branch()
            git_sha = get_git_sha()

            generate_report(
                results=self.results,
                failures=self.failures,
                baselines=baselines,
                criteria_by_task=criteria_by_task,
                output_dir=self.config.output_root,
                branch=branch,
                git_sha=git_sha,
                accelerator=self.config.accelerator,
            )
        except Exception:
            logger.warning("Failed to generate report.", exc_info=True)

    # -- disk cleanup ------------------------------------------------------

    def _cleanup_checkpoints(self) -> None:
        """Delete training checkpoints and exported models to save disk space.

        Keeps only ``metrics.csv`` and ``result.json`` files which contain
        the extracted metrics.  Skipped when ``RunConfig.keep_checkpoints``
        is ``True``.
        """
        output_root = self.config.output_root
        if not output_root.exists():
            return

        extensions = {".ckpt", ".xml", ".bin", ".onnx"}
        removed = 0
        freed_bytes = 0
        for path in output_root.rglob("*"):
            if path.is_file() and path.suffix in extensions:
                freed_bytes += path.stat().st_size
                path.unlink()
                removed += 1

        if removed:
            freed_mb = freed_bytes / (1024 * 1024)
            logger.info("Cleanup: removed %d checkpoint/model files (%.1f MB freed).", removed, freed_mb)

    # -- single experiment -------------------------------------------------

    @staticmethod
    def _is_deterministic_error(exc: BaseException) -> bool:
        """Check if the exception is due to a missing deterministic implementation."""
        return isinstance(exc, RuntimeError) and "does not have a deterministic implementation" in str(exc)

    # Deterministic fallback chain: True → "warn" → False
    _DETERMINISTIC_FALLBACKS: ClassVar[dict[bool | str, bool | str]] = {
        True: "warn",
        "warn": False,
    }

    def _run_single(
        self,
        experiment: Experiment,
        seed: int,
        data_path: Path,
        allowed_phases: set[str],
    ) -> ExperimentResult:
        """Run one ``(experiment, seed)`` with retry logic."""
        last_exc: BaseException | None = None
        deterministic: bool | str = self.config.deterministic

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return self._execute(experiment, seed, data_path, allowed_phases, deterministic=deterministic)
            except Exception as exc:  # noqa: PERF203
                last_exc = exc
                if attempt < _MAX_ATTEMPTS:
                    # If the failure is due to a missing deterministic kernel,
                    # relax the deterministic setting one level for the retry.
                    fallback = self._DETERMINISTIC_FALLBACKS.get(deterministic)
                    if self._is_deterministic_error(exc) and fallback is not None:
                        logger.warning(
                            "  seed=%d attempt %d/%d failed due to non-deterministic op; "
                            "retrying with deterministic=%r: %s",
                            seed,
                            attempt,
                            _MAX_ATTEMPTS,
                            fallback,
                            exc,
                        )
                        deterministic = fallback
                    else:
                        logger.warning(
                            "  seed=%d attempt %d/%d failed, retrying: %s",
                            seed,
                            attempt,
                            _MAX_ATTEMPTS,
                            exc,
                        )
                else:
                    logger.exception(
                        "  seed=%d failed after %d attempts",
                        seed,
                        _MAX_ATTEMPTS,
                    )

        return ExperimentResult.failure(
            task=experiment.task,
            model=experiment.model.name,
            dataset=experiment.dataset_name,
            scenario=experiment.scenario.name,
            seed=seed,
            exc=last_exc,  # type: ignore[arg-type]
        )

    def _execute(
        self,
        experiment: Experiment,
        seed: int,
        data_path: Path,
        allowed_phases: set[str],
        *,
        deterministic: bool | str | None = None,
    ) -> ExperimentResult:
        """Actually run the phases for one seed."""
        if deterministic is None:
            deterministic = self.config.deterministic

        seed_dir = self.config.output_root / experiment.run_id / str(seed)

        # Resume check
        skip, resume_from = detect_resume_point(seed_dir)
        if skip:
            logger.info("  seed=%d — all phases complete, skipping.", seed)
            return ExperimentResult(
                task=experiment.task,
                model=experiment.model.name,
                dataset=experiment.dataset_name,
                scenario=experiment.scenario.name,
                seed=seed,
                success=True,
                phases=[],
            )

        # Merge scenario overrides with ad-hoc CLI overrides (CLI wins).
        merged_overrides = {**experiment.scenario.overrides}
        merged_overrides.update(self.config.ad_hoc_overrides)

        merged_train_kwargs = {**experiment.scenario.train_kwargs}
        merged_train_kwargs.update(self.config.ad_hoc_train_kwargs)

        executor = ExperimentExecutor(
            recipe_path=experiment.recipe_path,
            data_path=data_path,
            work_dir=seed_dir,
            accelerator=self.config.accelerator,
            scenario_overrides=merged_overrides,
            train_kwargs=merged_train_kwargs,
            seed=seed,
            deterministic=deterministic,
            max_epochs=self.config.max_epochs,
        )

        # Determine which phases to run (respecting resume point)
        resuming = resume_from is not None
        phase_results: list[PhaseResult] = []

        for phase_name, method_name in _PHASE_CHAIN:
            if phase_name not in allowed_phases:
                continue
            if resuming and phase_name != resume_from:
                # Still skipping phases until we reach the resume point
                continue
            resuming = False  # From here on, execute everything

            logger.info("  seed=%d  phase=%s", seed, phase_name)
            method = getattr(executor, method_name)
            result = method()
            phase_results.append(result)

        return ExperimentResult(
            task=experiment.task,
            model=experiment.model.name,
            dataset=experiment.dataset_name,
            scenario=experiment.scenario.name,
            seed=seed,
            success=True,
            phases=phase_results,
        )
