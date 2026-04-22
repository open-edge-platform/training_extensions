# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experiment executor - thin wrapper around OTX/OV engines with timing."""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback as _traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Metrics collected from a single execution phase."""

    phase: str  # e.g. "train", "test/torch", "test/export", "export", "optimize"
    metrics: dict[str, float] = field(default_factory=dict)
    wall_time: float = 0.0


@dataclass
class ExperimentResult:
    """Aggregated result for one ``(experiment, seed)`` run."""

    task: str
    model: str
    dataset: str
    scenario: str
    seed: int
    success: bool
    phases: list[PhaseResult] = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None

    def all_metrics(self) -> dict[str, float]:
        """Merge metrics from all phases into a single dict."""
        merged: dict[str, float] = {}
        for phase in self.phases:
            merged.update(phase.metrics)
        return merged

    def total_wall_time(self) -> float:
        """Sum of wall-clock seconds across every phase of this seed run."""
        return float(sum(p.wall_time for p in self.phases))

    @classmethod
    def failure(
        cls,
        *,
        task: str,
        model: str,
        dataset: str,
        scenario: str,
        seed: int,
        exc: BaseException,
    ) -> ExperimentResult:
        """Construct a failed result from an exception."""
        tb_str = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
        return cls(
            task=task,
            model=model,
            dataset=dataset,
            scenario=scenario,
            seed=seed,
            success=False,
            phases=[],
            error=f"{type(exc).__name__}: {exc}",
            traceback=tb_str,
        )


# ---------------------------------------------------------------------------
# Override resolution
# ---------------------------------------------------------------------------


def resolve_overrides(scenario_overrides: dict[str, Any]) -> dict[str, Any]:
    """Convert scenario overrides into kwargs for ``OTXEngine.from_config()``.

    Complex values (dicts, lists) are JSON-serialized so that jsonargparse can
    parse them on the engine side.
    """
    resolved: dict[str, Any] = {}
    for dotpath, value in scenario_overrides.items():
        if isinstance(value, (dict, list)):
            resolved[dotpath] = json.dumps(value)
        else:
            resolved[dotpath] = value
    return resolved


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------

# Maps (phase_name -> marker file relative to seed dir) for resume checks.
_PHASE_MARKERS: list[tuple[str, str]] = [
    ("train", "train/metrics.csv"),
    ("test/torch", "test/torch/result.json"),
    ("export", "export/exported_model.xml"),
    ("test/export", "test/export/result.json"),
    ("optimize", "optimize/optimized_model.xml"),
    ("test/optimize", "test/optimize/result.json"),
]


def detect_resume_point(seed_dir: Path) -> tuple[bool, str | None]:
    """Determine whether an experiment can be skipped or partially resumed.

    Returns:
        ``(True, None)`` - all phases complete, skip entirely.
        ``(False, None)`` - start from scratch.
        ``(False, <phase_name>)`` - training done, resume from this phase.
    """
    train_marker = seed_dir / "train" / "metrics.csv"
    if not train_marker.exists() or train_marker.stat().st_size == 0:
        # Training not done or corrupt -> start over
        if seed_dir.exists():
            shutil.rmtree(seed_dir)
        return False, None

    checkpoint = seed_dir / "train" / "best_checkpoint.ckpt"
    if not checkpoint.exists():
        # metrics exist but checkpoint missing -> corrupt
        shutil.rmtree(seed_dir)
        return False, None

    # Training is complete. Walk later phases to find the first incomplete one.
    for phase_name, marker_rel in _PHASE_MARKERS[1:]:  # skip "train"
        marker = seed_dir / marker_rel
        if not marker.exists():
            return False, phase_name

    return True, None


# ---------------------------------------------------------------------------
# Metric scraping helpers
# ---------------------------------------------------------------------------


def _scrape_csv_metrics(csv_path: Path, prefix: str) -> dict[str, float]:
    """Read a Lightning ``metrics.csv`` and extract key aggregates.

    The prefix is prepended to each metric key (e.g. ``"training:"``).
    """
    if not csv_path.exists():
        return {}
    try:
        raw_metrics = pd.read_csv(csv_path)
    except Exception:
        logger.warning("Could not parse %s", csv_path)
        return {}

    metrics: dict[str, float] = {}
    for col in raw_metrics.columns:
        series = raw_metrics[col].dropna()
        if series.empty:
            continue
        # For val metrics, take the max (best epoch).
        # For timing metrics, take the mean (skip the first warmup step).
        if "val/" in col:
            metrics[f"{prefix}{col}"] = float(series.max())
        elif "iter_time" in col:
            trimmed = series.iloc[min(1, len(series) - 1) :]
            metrics[f"{prefix}{col}"] = float(trimmed.mean())
        elif "epoch" in col:
            # Lightning records ``epoch`` as a 0-indexed counter, so the max
            # is ``num_epochs - 1``.  Report the human-readable count instead.
            metrics[f"{prefix}{col}"] = float(series.max()) + 1.0
        elif "gpu_mem" in col or "gpu" in col.lower():
            metrics[f"{prefix}{col}"] = float(series.max())
    return metrics


def _find_csv_metrics(csv_dir: Path) -> Path | None:
    """Locate the Lightning CSV logger's ``metrics.csv`` under *csv_dir*.

    Lightning writes CSV files to ``csv/version_N/metrics.csv``, where *N*
    increments on each run.  This helper returns the ``metrics.csv`` inside
    the highest ``version_*`` directory, so it works even when multiple
    training sessions have run (e.g. during resume).

    Falls back to ``csv_dir / "metrics.csv"`` if the ``version_*`` layout
    is not present (for forward compatibility).
    """
    csv_parent = csv_dir / "csv"
    if csv_parent.is_dir():
        # Find all version_* directories and pick the highest number
        version_dirs = sorted(
            (d for d in csv_parent.iterdir() if d.is_dir() and d.name.startswith("version_")),
            key=lambda d: int(d.name.split("_", 1)[1]) if d.name.split("_", 1)[1].isdigit() else -1,
        )
        if version_dirs:
            candidate = version_dirs[-1] / "metrics.csv"
            if candidate.exists():
                return candidate
    # Fallback: direct metrics.csv (e.g. train/metrics.csv)
    direct = csv_dir / "metrics.csv"
    if direct.exists():
        return direct
    return None


def _get_peak_gpu_memory_mb() -> float:
    """Best-effort peak GPU memory reading (returns 0.0 if unavailable)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        logger.debug("Could not read peak GPU memory.", exc_info=True)
    return 0.0


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class ExperimentExecutor:
    """Run train / test / export / optimize for a single experiment + seed.

    The executor is intentionally stateless with respect to tracking -- it
    returns structured :class:`PhaseResult` objects that the runner can
    forward to any tracking backend.
    """

    def __init__(
        self,
        *,
        recipe_path: Path,
        data_path: Path,
        work_dir: Path,
        accelerator: str = "gpu",
        scenario_overrides: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
        deterministic: bool | str = True,
        max_epochs: int | None = None,
    ) -> None:
        self.recipe_path = recipe_path
        self.data_path = data_path
        self.work_dir = work_dir
        self.accelerator = accelerator
        self.scenario_overrides = scenario_overrides or {}
        self.extra_train_kwargs = train_kwargs or {}
        self.seed = seed
        self.deterministic = deterministic
        self.max_epochs = max_epochs

    # -- phases ------------------------------------------------------------

    def train(self) -> PhaseResult:
        """Train the model and return scraped metrics."""
        from getitune.backend.native.engine import OTXEngine

        overrides = resolve_overrides(self.scenario_overrides)
        engine = OTXEngine.from_config(
            config_path=self.recipe_path,
            data_root=self.data_path,
            work_dir=self.work_dir / "train",
            device=self.accelerator,
            **overrides,
        )

        kwargs: dict[str, Any] = {
            "seed": self.seed,
            "deterministic": self.deterministic,
        }
        if self.max_epochs is not None and self.max_epochs > 0:
            kwargs["max_epochs"] = self.max_epochs
        kwargs.update(self.extra_train_kwargs)

        start = time.monotonic()
        engine.train(**kwargs)
        wall = time.monotonic() - start

        # Scrape metrics from the CSV that Lightning writes
        csv_path = _find_csv_metrics(self.work_dir / "train")
        csv_metrics = _scrape_csv_metrics(csv_path, prefix="training:") if csv_path else {}
        csv_metrics["training:e2e_time"] = wall
        csv_metrics["training:gpu_mem"] = _get_peak_gpu_memory_mb()

        del engine
        return PhaseResult(phase="train", metrics=csv_metrics, wall_time=wall)

    def test_torch(self) -> PhaseResult:
        """Test the PyTorch checkpoint and return metrics."""
        from getitune.backend.native.engine import OTXEngine

        overrides = resolve_overrides(self.scenario_overrides)
        engine = OTXEngine.from_config(
            config_path=self.recipe_path,
            data_root=self.data_path,
            work_dir=self.work_dir / "test" / "torch",
            device=self.accelerator,
            **overrides,
        )
        ckpt = self.work_dir / "train" / "best_checkpoint.ckpt"

        start = time.monotonic()
        engine.test(checkpoint=ckpt)
        wall = time.monotonic() - start

        num_samples = max(len(engine.datamodule.subsets.get("test", [])), 1)
        latency = wall / num_samples

        csv_path = _find_csv_metrics(Path(engine.work_dir))
        csv_metrics = _scrape_csv_metrics(csv_path, prefix="torch:") if csv_path else {}
        csv_metrics["torch:test/e2e_time"] = wall
        csv_metrics["torch:test/latency"] = latency

        # Write a marker for resume detection
        result_json = self.work_dir / "test" / "torch" / "result.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(csv_metrics, indent=2))

        del engine
        return PhaseResult(phase="test/torch", metrics=csv_metrics, wall_time=wall)

    def export(self) -> PhaseResult:
        """Export the trained model to OpenVINO IR."""
        from getitune.backend.native.engine import OTXEngine

        overrides = resolve_overrides(self.scenario_overrides)
        engine = OTXEngine.from_config(
            config_path=self.recipe_path,
            data_root=self.data_path,
            work_dir=self.work_dir / "export",
            device=self.accelerator,
            **overrides,
        )
        ckpt = self.work_dir / "train" / "best_checkpoint.ckpt"

        start = time.monotonic()
        engine.export(checkpoint=ckpt)
        wall = time.monotonic() - start

        del engine
        return PhaseResult(phase="export", metrics={"export:e2e_time": wall}, wall_time=wall)

    def test_export(self) -> PhaseResult:
        """Test the exported OpenVINO model and return metrics."""
        from getitune.backend.openvino.engine import OVEngine

        exported = self._find_exported_model()
        engine = OVEngine(
            work_dir=self.work_dir / "test" / "export",
            data=self.data_path,
            model=exported,
        )

        start = time.monotonic()
        engine.test(checkpoint=exported)
        wall = time.monotonic() - start

        num_samples = max(len(engine.datamodule.subsets.get("test", [])), 1)
        latency = wall / num_samples

        csv_path = _find_csv_metrics(Path(engine.work_dir))
        csv_metrics = _scrape_csv_metrics(csv_path, prefix="export:") if csv_path else {}
        csv_metrics["export:test/e2e_time"] = wall
        csv_metrics["export:test/latency"] = latency

        result_json = self.work_dir / "test" / "export" / "result.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(csv_metrics, indent=2))

        del engine
        return PhaseResult(phase="test/export", metrics=csv_metrics, wall_time=wall)

    def optimize(self) -> PhaseResult:
        """Optimize the exported model with NNCF/POT."""
        from getitune.backend.openvino.engine import OVEngine

        exported = self._find_exported_model()
        engine = OVEngine(
            work_dir=self.work_dir / "optimize",
            data=self.data_path,
            model=exported,
        )

        start = time.monotonic()
        engine.optimize(checkpoint=exported)
        wall = time.monotonic() - start

        del engine
        return PhaseResult(phase="optimize", metrics={"optimize:e2e_time": wall}, wall_time=wall)

    def test_optimize(self) -> PhaseResult:
        """Test the optimized model and return metrics."""
        from getitune.backend.openvino.engine import OVEngine

        optimized = self.work_dir / "optimize" / "optimized_model.xml"
        if not optimized.exists():
            msg = f"Optimized model not found: {optimized}"
            raise FileNotFoundError(msg)

        engine = OVEngine(
            work_dir=self.work_dir / "test" / "optimize",
            data=self.data_path,
            model=optimized,
        )

        start = time.monotonic()
        engine.test(checkpoint=optimized)
        wall = time.monotonic() - start

        num_samples = max(len(engine.datamodule.subsets.get("test", [])), 1)
        latency = wall / num_samples

        csv_path = _find_csv_metrics(Path(engine.work_dir))
        csv_metrics = _scrape_csv_metrics(csv_path, prefix="optimize:") if csv_path else {}
        csv_metrics["optimize:test/e2e_time"] = wall
        csv_metrics["optimize:test/latency"] = latency

        result_json = self.work_dir / "test" / "optimize" / "result.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(csv_metrics, indent=2))

        del engine
        return PhaseResult(phase="test/optimize", metrics=csv_metrics, wall_time=wall)

    # -- helpers -----------------------------------------------------------

    def _find_exported_model(self) -> Path:
        """Locate the exported OpenVINO IR XML file."""
        candidates = [
            self.work_dir / "export" / "exported_model.xml",
            self.work_dir / ".latest" / "export" / "exported_model_decoder.xml",
        ]
        for p in candidates:
            if p.exists():
                return p
        msg = f"Exported model not found in any of: {candidates}"
        raise FileNotFoundError(msg)
