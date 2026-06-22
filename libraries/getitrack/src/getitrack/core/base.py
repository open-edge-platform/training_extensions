# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tracker interface and algorithm registry.

Concrete trackers register themselves with the `register_algorithm`
decorator, and `BaseTracker.from_config` dispatches on the registered
name. The registry is populated once the algorithm modules are imported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from getitrack.config import TrackerConfig
from getitrack.core.registry import ALGORITHM_REGISTRY
from getitrack.logger import LOGGER, enable_logging

if TYPE_CHECKING:
    import numpy as np

    from getitrack.core.detection import Detections, TrackedDetections


class BaseTracker(ABC):
    """Abstract base class for multi-object trackers.

    Concrete subclasses implement `_update_impl`; the public `update`
    wraps it with cross-cutting concerns such as verbose logging. The
    base owns the monotonic id allocator and `_frame_id` bookkeeping so
    subclasses focus on association logic. Track ids are scoped per
    instance (not per process), so parallel tracker instances do not
    collide.
    """

    algorithm_name: ClassVar[str] = ""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._next_id: int = 1
        self._frame_id: int | None = None
        if config.verbose:
            enable_logging()

    def update(self, detections: Detections) -> TrackedDetections:
        """Process one frame's detections and return the tracker output.

        Applies ``association.class_filter`` before the algorithm runs,
        so excluded classes never spawn or match tracks. ``det_indices``
        on the output always index into the unfiltered ``detections``
        passed here.
        """
        class_filter = self.config.association.class_filter
        if class_filter is None:
            filtered, source_rows = detections, None
        else:
            filtered, source_rows = detections.filter_by_class(class_filter)
        tracked = self._update_impl(filtered)
        if source_rows is not None and tracked.det_indices is not None:
            remapped = self._remap_det_indices(source_rows, tracked.det_indices)
            tracked = replace(tracked, det_indices=remapped)
        if self.config.verbose:
            self._log_update(filtered, tracked)
        return tracked

    @staticmethod
    def _remap_det_indices(source_rows: np.ndarray, det_indices: np.ndarray) -> np.ndarray:
        """Map ``det_indices`` from filtered-row space back to input rows."""
        remapped = det_indices.copy()
        matched = remapped >= 0
        remapped[matched] = source_rows[remapped[matched]]
        return remapped

    @abstractmethod
    def _update_impl(self, detections: Detections) -> TrackedDetections:
        """Algorithm-specific tracking step for one frame."""

    def _log_update(self, detections: Detections, tracked: TrackedDetections) -> None:
        """Emit a one-line per-frame summary on the ``getitrack`` logger."""
        n_high = int((detections.scores >= self.config.association.high_score_threshold).sum())
        pairs = ", ".join(
            f"{tid}:{cls}:{score:.2f}"
            for tid, cls, score in zip(
                tracked.track_ids.tolist(),
                tracked.class_ids.tolist(),
                tracked.scores.tolist(),
                strict=True,
            )
        )
        LOGGER.info(
            "frame {:4}: {} detections ({} high-score), {} tracks [id:class:score {}]",
            detections.frame_id,
            len(detections),
            n_high,
            len(tracked),
            pairs,
        )

    def reset(self) -> None:
        """Clear internal state between videos or sequences."""
        self._next_id = 1
        self._frame_id = None

    def _allocate_id(self) -> int:
        """Return a fresh monotonic id for a new track."""
        new_id = self._next_id
        self._next_id += 1
        return new_id

    @classmethod
    def from_config(cls, config: TrackerConfig | dict[str, Any] | str | Path) -> BaseTracker:
        """Instantiate a tracker dispatched on ``config.algorithm``.

        Accepts a `TrackerConfig`, a dict, or a path to a YAML file.
        """
        if isinstance(config, TrackerConfig):
            resolved = config
        elif isinstance(config, dict):
            resolved = TrackerConfig.model_validate(config)
        elif isinstance(config, str | Path):
            resolved = TrackerConfig.from_yaml(config)
        else:
            msg = f"unsupported config type: {type(config).__name__}"
            raise TypeError(msg)

        name = resolved.algorithm.value
        if name not in ALGORITHM_REGISTRY:
            known = sorted(ALGORITHM_REGISTRY) or ["<none registered>"]
            msg = f"unknown algorithm '{name}'; registered: {known}"
            raise KeyError(msg)
        return ALGORITHM_REGISTRY[name](resolved)
