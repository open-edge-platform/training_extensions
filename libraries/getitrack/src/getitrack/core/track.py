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
from typing import TYPE_CHECKING, assert_never

from getitrack.core.detection import TrackState
from getitrack.logger import LOGGER

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
        prev_state = self.state
        self.bbox = bbox
        self.score = float(score)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        match self.state:
            case TrackState.TENTATIVE:
                if self.hits >= lifecycle.min_hits:
                    self.state = TrackState.ACTIVE
            case TrackState.LOST:
                self.state = TrackState.ACTIVE
            case TrackState.ACTIVE | TrackState.REMOVED:
                pass
            case _:
                assert_never(self.state)
        if self.state != prev_state:
            LOGGER.debug("track {}: {} -> {} on hit (hits={})", self.track_id, prev_state, self.state, self.hits)

    def mark_miss(self, lifecycle: LifecycleConfig) -> None:
        """Record a missed observation on this frame and advance state."""
        prev_state = self.state
        self.age += 1
        self.time_since_update += 1
        match self.state:
            case TrackState.TENTATIVE:
                if self.time_since_update > lifecycle.tentative_max_age:
                    self.state = TrackState.REMOVED
            case TrackState.ACTIVE:
                self.state = TrackState.LOST
            case TrackState.LOST:
                if self.time_since_update > lifecycle.max_age:
                    self.state = TrackState.REMOVED
            case TrackState.REMOVED:
                pass
            case _:
                assert_never(self.state)
        if self.state != prev_state:
            LOGGER.debug(
                "track {}: {} -> {} on miss (time_since_update={})",
                self.track_id,
                prev_state,
                self.state,
                self.time_since_update,
            )

    @property
    def should_remove(self) -> bool:
        """True once this track has transitioned to ``REMOVED``."""
        return self.state == TrackState.REMOVED
