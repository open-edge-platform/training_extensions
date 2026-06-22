# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pydantic configuration models for getitrack.

`TrackerConfig` is the top-level entry point. Sub-configs group fields by
responsibility (association, lifecycle, motion, appearance, interpolation)
so a YAML file can tune one concern in isolation. Field docstrings surface
as ``description`` strings in ``model_json_schema()`` because the shared
`_StrictModel` base sets ``use_attribute_docstrings=True``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


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


class AssociationConfig(_StrictModel):
    """Detection-to-track matching parameters."""

    match_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    """Maximum assignment cost accepted when matching detections to tracks.
    The cost is ``1 - IoU``, score-fused to ``1 - IoU * score`` where fusion
    applies, so larger values accept weaker overlaps."""

    score_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    """Drop detections with confidence below this value before matching runs."""

    high_score_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6
    """Boundary used by two-stage association (ByteTrack) to split detections into high-score and low-score buckets."""

    match_class_only: bool = True
    """Restrict matching to detection-track pairs that share a class id."""

    class_filter: list[int] | None = None
    """Track only detections whose class id is in this list; None tracks all
    classes. Applied before association, so excluded classes never spawn or
    match tracks."""


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


class AppearanceConfig(_StrictModel):
    """Appearance and embedding-based matching parameters."""

    enabled: bool = False
    """Fuse appearance similarity into the matching cost when true."""

    embedding_dim: Annotated[int, Field(ge=1)] = 512
    """Feature-vector dimensionality expected from a detector or re-identification head."""

    similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    """Minimum cosine similarity accepted as an appearance match."""

    memory_bank_size: Annotated[int, Field(ge=1)] = 10
    """Number of recent embeddings retained per track."""

    weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    """Blend between motion and appearance cost: 0.0 is pure motion, 1.0 is pure appearance."""


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
    """Top-level tracker configuration.

    Construct from a YAML file via `TrackerConfig.from_yaml`, from a dict
    via `TrackerConfig.model_validate`, or programmatically.
    """

    algorithm: AlgorithmType = AlgorithmType.BYTETRACK
    """Algorithm identifier. Each value resolves to a registered `BaseTracker` subclass."""

    verbose: bool = False
    """Log a one-line tracking summary per frame at INFO level on the ``getitrack`` logger."""

    association: AssociationConfig = Field(default_factory=AssociationConfig)
    """Detection-to-track matching parameters."""

    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    """Track creation, confirmation, and removal parameters."""

    motion: MotionConfig = Field(default_factory=MotionConfig)
    """Kalman filter and motion model parameters."""

    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    """Appearance and embedding-based matching parameters."""

    interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)
    """Bbox interpolation and smoothing parameters."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrackerConfig:
        """Load a validated configuration from a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            raw: Any = yaml.safe_load(f)
        return cls.model_validate(raw or {})

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this configuration to a YAML file."""
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
