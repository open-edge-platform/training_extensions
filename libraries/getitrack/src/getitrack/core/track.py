# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Per-track lifecycle state machine.

A `Track` captures the lifecycle of one tracked object across frames.
Concrete trackers hold a collection of `Track` instances and decide
associations; the transitions implemented here are algorithm-agnostic.

State transitions::

    TENTATIVE --(min_hits hits)--> ACTIVE
    ACTIVE    --(miss)----------> LOST
    LOST      --(hit)-----------> ACTIVE
    LOST      --(time_since_update > max_age)--> REMOVED
    TENTATIVE --(time_since_update > tentative_max_age)--> REMOVED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from getitrack.core.detection import TrackState

if TYPE_CHECKING:
    import numpy as np

    from getitrack.config import LifecycleConfig


@dataclass
class Track:
    """Per-id state machine maintained by a concrete tracker.

    Algorithm-specific extensions (e.g. Kalman state) live alongside the
    `Track`, typically as parallel dicts keyed by ``track_id``, so the
    lifecycle transitions stay algorithm-agnostic.

    Attributes:
        track_id: Stable identifier assigned at creation.
        class_id: Class label associated with this track.
        bbox: Last observed bbox in ``[x1, y1, x2, y2]`` (float32).
        score: Last observed detection score.
        state: Current lifecycle state.
        age: Total frames since the track was created.
        hits: Total observed detections.
        time_since_update: Consecutive frames missed (0 on a hit).
    """

    track_id: int
    class_id: int
    bbox: np.ndarray
    score: float
    state: TrackState = TrackState.TENTATIVE
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    _start_frame: int = field(default=0, repr=False)

    def mark_hit(self, bbox: np.ndarray, score: float, lifecycle: LifecycleConfig) -> None:
        """Record an observed detection on this frame and update state."""
        self.bbox = bbox
        self.score = float(score)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        if (self.state == TrackState.TENTATIVE and self.hits >= lifecycle.min_hits) or (self.state == TrackState.LOST):
            self.state = TrackState.ACTIVE

    def mark_miss(self, lifecycle: LifecycleConfig) -> None:
        """Record a missed observation on this frame and advance state."""
        self.age += 1
        self.time_since_update += 1
        if self.state == TrackState.TENTATIVE and self.time_since_update > lifecycle.tentative_max_age:
            self.state = TrackState.REMOVED
            return
        if self.state == TrackState.ACTIVE:
            self.state = TrackState.LOST
        if self.state == TrackState.LOST and self.time_since_update > lifecycle.max_age:
            self.state = TrackState.REMOVED

    @property
    def should_remove(self) -> bool:
        """True once this track has transitioned to ``REMOVED``."""
        return self.state == TrackState.REMOVED
