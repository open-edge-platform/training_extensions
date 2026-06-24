# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shared Pydantic configuration models for getitrack.

`TrackerConfig` holds the parameters common to every algorithm. Each algorithm
defines its own subclass (e.g. `ByteTrackConfig`) and registers it under an
``algorithm`` name; loading dispatches on that name so a config validates against
the matching algorithm and rejects parameters that do not apply.

Sub-configs (lifecycle, motion, interpolation) group related fields. Field
docstrings become schema ``description`` strings via ``use_attribute_docstrings``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AlgorithmType(StrEnum):
    """Identifier resolving to a registered `BaseTracker` subclass."""

    BYTETRACK = "bytetrack"
    OCSORT = "ocsort"
    BOTSORT = "botsort"
    MEMORY = "memory"


class InterpolationMethod(StrEnum):
    """Strategy used by the post-processing interpolator."""

    LINEAR = "linear"
    KALMAN = "kalman"
    SPLINE = "spline"


class _StrictModel(BaseModel):
    """Base for config models: reject unknown keys and expose field docstrings."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class LifecycleConfig(_StrictModel):
    """Track creation, confirmation, and removal parameters."""

    max_age: Annotated[int, Field(ge=1)] = 30
    """Consecutive missed frames a LOST track may accumulate before removal."""

    min_hits: Annotated[int, Field(ge=1)] = 2
    """Observed detections required to promote a TENTATIVE track to ACTIVE.
    Trackers may bypass this on the first frame of a sequence."""

    tentative_max_age: Annotated[int, Field(ge=0)] = 0
    """Consecutive missed frames a TENTATIVE track tolerates before removal.
    0 removes on the first miss (reference ByteTrack behavior)."""


class MotionConfig(_StrictModel):
    """Kalman filter and motion model parameters."""

    process_noise: Annotated[float, Field(gt=0.0)] = 1.0
    """Multiplier on the process-noise covariance (Q). Larger values weight observations over the motion prior."""

    measurement_noise: Annotated[float, Field(gt=0.0)] = 1.0
    """Multiplier on the measurement-noise covariance (R). Larger values weight the motion prior over observations."""

    velocity_decay: Annotated[float, Field(gt=0.0, le=1.0)] = 0.99
    """Per-frame velocity damping in ``(0, 1]``. Values below 1.0 simulate gradual deceleration."""


class InterpolationConfig(_StrictModel):
    """Bbox interpolation and smoothing parameters."""

    enabled: bool = True
    """Enable the interpolation post-processing stage."""

    method: InterpolationMethod = InterpolationMethod.LINEAR
    """Interpolation strategy used when ``enabled`` is true."""

    max_gap: Annotated[int, Field(ge=1)] = 5
    """Maximum consecutive missing frames bridged by interpolation."""

    smoothing_window: Annotated[int, Field(ge=1)] = 5
    """Window size for spline or moving-average smoothing."""

    online_buffer: Annotated[int, Field(ge=0)] = 0
    """Frames of lookahead permitted in online mode. 0 is strictly causal (zero latency)."""


class TrackerConfig(_StrictModel):
    """Parameters shared by all tracking algorithms.

    Each algorithm subclasses this with its own parameters and pins
    ``algorithm`` to a single value. Load any variant from YAML via
    `TrackerConfig.from_yaml`, which dispatches on the ``algorithm`` key.
    """

    algorithm: AlgorithmType
    """Algorithm identifier; each subclass defaults it to a single value."""

    verbose: bool = False
    """Log a one-line tracking summary per frame at INFO level on the ``getitrack`` logger."""

    class_filter: list[int] | None = None
    """Track only detections whose class id is in this list; None tracks all
    classes. Applied before association, so excluded classes never spawn or
    match tracks."""

    score_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    """Drop detections with confidence below this value before tracking runs."""

    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    """Track creation, confirmation, and removal parameters."""

    motion: MotionConfig = Field(default_factory=MotionConfig)
    """Kalman filter and motion model parameters."""

    interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)
    """Bbox interpolation and smoothing parameters."""

    @model_validator(mode="after")
    def _algorithm_matches_variant(self) -> TrackerConfig:
        """Reject an ``algorithm`` value that differs from the one a subclass pins."""
        field = type(self).model_fields["algorithm"]
        if not field.is_required() and self.algorithm != field.default:
            msg = f"{type(self).__name__} pins algorithm={field.default!r}; got {self.algorithm!r}"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrackerConfig:
        """Load and validate a tracker configuration from a YAML file.

        Dispatches on the ``algorithm`` key (default ``bytetrack``) and returns
        the matching algorithm's config variant.
        """
        from getitrack.core.registry import resolve_tracker_config

        with Path(path).open(encoding="utf-8") as f:
            raw: Any = yaml.safe_load(f)
        return resolve_tracker_config(raw or {})

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this configuration to a YAML file."""
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
