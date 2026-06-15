# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-track lifecycle state machine."""

import numpy as np
import pytest

from getitrack.config import LifecycleConfig
from getitrack.core.detection import TrackState
from getitrack.core.track import Track


@pytest.fixture
def lifecycle() -> LifecycleConfig:
    return LifecycleConfig(max_age=5, min_hits=3, tentative_max_age=2)


@pytest.fixture
def track() -> Track:
    return Track(
        track_id=1,
        class_id=0,
        bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        score=0.9,
    )


class TestTrackInit:
    def test_initial_state_is_tentative(self, track: Track):
        assert track.state == TrackState.TENTATIVE
        assert track.hits == 1
        assert track.age == 0
        assert track.time_since_update == 0


class TestPromotion:
    def test_promotes_to_active_at_min_hits(self, track: Track, lifecycle: LifecycleConfig):
        track.mark_hit(np.array([1, 1, 11, 11], dtype=np.float32), 0.8, lifecycle)
        assert track.state == TrackState.TENTATIVE
        track.mark_hit(np.array([2, 2, 12, 12], dtype=np.float32), 0.8, lifecycle)
        assert track.state == TrackState.ACTIVE
        assert track.hits == 3

    def test_hit_resets_time_since_update(self, track: Track, lifecycle: LifecycleConfig):
        track.mark_miss(lifecycle)
        assert track.time_since_update == 1
        track.mark_hit(np.zeros(4, dtype=np.float32), 0.9, lifecycle)
        assert track.time_since_update == 0


class TestLossAndRemoval:
    def test_active_miss_transitions_to_lost(self, track: Track, lifecycle: LifecycleConfig):
        track.mark_hit(np.zeros(4, dtype=np.float32), 0.9, lifecycle)
        track.mark_hit(np.zeros(4, dtype=np.float32), 0.9, lifecycle)
        assert track.state == TrackState.ACTIVE
        track.mark_miss(lifecycle)
        assert track.state == TrackState.LOST

    def test_lost_track_recovers_on_hit(self, track: Track, lifecycle: LifecycleConfig):
        track.mark_hit(np.zeros(4, dtype=np.float32), 0.9, lifecycle)
        track.mark_hit(np.zeros(4, dtype=np.float32), 0.9, lifecycle)
        track.mark_miss(lifecycle)
        assert track.state == TrackState.LOST
        track.mark_hit(np.zeros(4, dtype=np.float32), 0.9, lifecycle)
        assert track.state == TrackState.ACTIVE

    def test_lost_track_removed_after_max_age(self, track: Track, lifecycle: LifecycleConfig):
        track.mark_hit(np.zeros(4, dtype=np.float32), 0.9, lifecycle)
        track.mark_hit(np.zeros(4, dtype=np.float32), 0.9, lifecycle)
        for _ in range(lifecycle.max_age + 1):
            track.mark_miss(lifecycle)
        assert track.state == TrackState.REMOVED
        assert track.should_remove is True

    def test_tentative_removed_after_tentative_max_age(self, track: Track, lifecycle: LifecycleConfig):
        for _ in range(lifecycle.tentative_max_age + 1):
            track.mark_miss(lifecycle)
        assert track.state == TrackState.REMOVED
        assert track.should_remove is True

    def test_age_increments_on_hit_and_miss(self, track: Track, lifecycle: LifecycleConfig):
        track.mark_hit(np.zeros(4, dtype=np.float32), 0.9, lifecycle)
        track.mark_miss(lifecycle)
        assert track.age == 2
