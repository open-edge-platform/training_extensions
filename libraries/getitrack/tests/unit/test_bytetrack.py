# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit + integration tests for the ByteTrack algorithm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import getitrack.algorithms  # noqa: F401  -> registers ByteTrack
from getitrack.algorithms import ByteTrackTracker
from getitrack.algorithms.bytetrack import ByteTrackConfig
from getitrack.config import LifecycleConfig, TrackerConfig
from getitrack.core.base import BaseTracker
from getitrack.core.detection import Detections
from getitrack.core.track import Track, TrackState


def _dets(
    boxes: list[list[float]], scores: list[float], frame_id: int, class_ids: list[int] | None = None
) -> Detections:
    n = len(boxes)
    return Detections(
        bboxes=np.asarray(boxes, dtype=np.float32).reshape(n, 4),
        scores=np.asarray(scores, dtype=np.float32),
        class_ids=np.asarray(class_ids if class_ids is not None else [0] * n, dtype=np.int64),
        frame_id=frame_id,
    )


@pytest.fixture
def fast_confirm_config() -> ByteTrackConfig:
    # Confirm after 1 hit so single-frame tests don't have to spam frames.
    return ByteTrackConfig(lifecycle=LifecycleConfig(min_hits=1, tentative_max_age=1, max_age=5))


class TestRegistry:
    def test_from_config_dispatches_to_bytetrack(self):
        cfg = ByteTrackConfig()
        tracker = BaseTracker.from_config(cfg)
        assert isinstance(tracker, ByteTrackTracker)


class TestSingleObject:
    def test_id_persists_across_frames(self, fast_confirm_config):
        bt = ByteTrackTracker(fast_confirm_config)
        out0 = bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=0))
        out = bt.update(_dets([[12, 12, 52, 52]], [0.9], frame_id=1))
        assert out0.track_ids.tolist() == [1]
        assert len(out) == 1
        assert out.track_ids[0] == 1


class TestMultiObject:
    def test_class_only_matching_keeps_classes_separate(self):
        cfg = ByteTrackConfig(lifecycle=LifecycleConfig(min_hits=1, tentative_max_age=1))
        bt = ByteTrackTracker(cfg)
        bt.update(_dets([[0, 0, 20, 20]], [0.9], frame_id=0, class_ids=[0]))
        # Same location, different class -> a new id, not a match.
        out = bt.update(_dets([[0, 0, 20, 20]], [0.9], frame_id=1, class_ids=[1]))
        assert sorted(out.track_ids.tolist()) == [2]


class TestLowScoreRecovery:
    def test_low_score_detection_keeps_track_active(self):
        cfg = ByteTrackConfig(lifecycle=LifecycleConfig(min_hits=1, tentative_max_age=1))
        bt = ByteTrackTracker(cfg)
        bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=0))
        # Only a low-score detection on frame 1; first-pass would miss it but second-pass picks it up.
        out = bt.update(_dets([[12, 12, 52, 52]], [0.3], frame_id=1))
        assert len(out) == 1
        assert out.track_ids[0] == 1


class TestConfigValidation:
    def test_low_floor_above_high_split_rejected(self):
        with pytest.raises(ValidationError, match="score_threshold must be below"):
            ByteTrackConfig(score_threshold=0.7, high_score_threshold=0.5)

    def test_high_split_without_room_for_margin_rejected(self):
        with pytest.raises(ValidationError, match="new-track margin"):
            ByteTrackConfig(high_score_threshold=0.95)

    def test_default_yaml_matches_programmatic_defaults(self):
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        assert TrackerConfig.from_yaml(path) == ByteTrackConfig()


class TestThresholdSemantics:
    def test_mid_score_detection_reactivates_lost_track(self):
        # A 0.55 detection is above the 0.5 high/low split, so it reaches the
        # first-pass association and recovers a LOST track, even though it is
        # below the 0.6 new-track threshold.
        bt = ByteTrackTracker(ByteTrackConfig(lifecycle=LifecycleConfig(min_hits=1, max_age=5)))
        bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=0))
        bt.update(_dets([], [], frame_id=1))  # miss -> LOST
        out = bt.update(_dets([[12, 12, 52, 52]], [0.55], frame_id=2))
        assert out.track_ids.tolist() == [1]

    def test_high_detection_below_margin_does_not_spawn(self):
        # 0.55 clears the split but not the split + 0.1 new-track margin, so with
        # no track to match it spawns nothing.
        bt = ByteTrackTracker(ByteTrackConfig(lifecycle=LifecycleConfig(min_hits=1)))
        out = bt.update(_dets([[10, 10, 50, 50]], [0.55], frame_id=0))
        assert len(out) == 0


class TestDuplicateSuppression:
    def test_active_duplicated_against_multiple_lost_does_not_crash(self):
        bt = ByteTrackTracker(ByteTrackConfig())
        box = np.array([0, 0, 40, 40], dtype=np.float32)
        bt._tracks = {
            1: Track(track_id=1, class_id=0, bbox=box, score=0.9, state=TrackState.LOST, age=10),
            2: Track(track_id=2, class_id=0, bbox=box.copy(), score=0.9, state=TrackState.LOST, age=10),
            3: Track(track_id=3, class_id=0, bbox=box.copy(), score=0.9, state=TrackState.ACTIVE, age=1),
        }
        bt._remove_duplicate_tracks()
        # Track 3 duplicates both LOST tracks; it is dropped once, no KeyError.
        assert set(bt._tracks) == {1, 2}

    def test_equal_age_tie_drops_active_track(self):
        bt = ByteTrackTracker(ByteTrackConfig())
        box = np.array([0, 0, 40, 40], dtype=np.float32)
        bt._tracks = {
            1: Track(track_id=1, class_id=0, bbox=box, score=0.9, state=TrackState.LOST, age=5),
            2: Track(track_id=2, class_id=0, bbox=box.copy(), score=0.9, state=TrackState.ACTIVE, age=5),
        }
        bt._remove_duplicate_tracks()
        # On an age tie the reference drops the active-side track.
        assert set(bt._tracks) == {1}

    def test_overlapping_pair_of_different_classes_is_kept(self):
        bt = ByteTrackTracker(ByteTrackConfig())
        box = np.array([0, 0, 40, 40], dtype=np.float32)
        bt._tracks = {
            1: Track(track_id=1, class_id=0, bbox=box, score=0.9, state=TrackState.LOST, age=10),
            2: Track(track_id=2, class_id=1, bbox=box.copy(), score=0.9, state=TrackState.ACTIVE, age=1),
        }
        bt._remove_duplicate_tracks()
        assert set(bt._tracks) == {1, 2}


class TestEmpty:
    def test_reset_clears_state(self, fast_confirm_config):
        bt = ByteTrackTracker(fast_confirm_config)
        bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=0))
        bt.reset()
        out = bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=0))
        # Id allocator is back to 1 after reset.
        assert out.track_ids[0] == 1


class TestLifecycle:
    def test_aging_out_removes_track(self):
        cfg = ByteTrackConfig(lifecycle=LifecycleConfig(min_hits=1, tentative_max_age=1, max_age=2))
        bt = ByteTrackTracker(cfg)
        bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=0))
        # Three empty frames: state goes ACTIVE -> LOST (frame 1) -> still LOST (2) -> REMOVED (3).
        for f in range(1, 4):
            bt.update(_dets([], [], frame_id=f))
        # A fresh detection now should spawn track id 2, not reuse 1.
        out = bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=4))
        assert out.track_ids[0] != 1


class TestStandardLifecycleDefaults:
    """Default config reproduces reference ByteTrack confirmation behavior."""

    def test_mid_sequence_track_confirms_on_second_match(self):
        bt = ByteTrackTracker(ByteTrackConfig())
        # min_hits=2 default, but tracks born on the first frame bypass TENTATIVE.
        out0 = bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=0))
        assert len(out0) == 1
        # New object on frame 1 starts TENTATIVE: not output yet.
        out1 = bt.update(_dets([[10, 10, 50, 50], [200, 200, 240, 240]], [0.9, 0.9], frame_id=1))
        assert len(out1) == 1
        # Its second match confirms it.
        out2 = bt.update(_dets([[10, 10, 50, 50], [202, 202, 242, 242]], [0.9, 0.9], frame_id=2))
        assert len(out2) == 2

    def test_tentative_removed_on_first_miss(self):
        # tentative_max_age=0 default: one missed frame kills an unconfirmed track.
        bt = ByteTrackTracker(ByteTrackConfig())
        bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=0))
        bt.update(_dets([[10, 10, 50, 50], [200, 200, 240, 240]], [0.9, 0.9], frame_id=1))
        bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=2))
        # The object reappears and confirms; it must carry a fresh id, not 2.
        bt.update(_dets([[10, 10, 50, 50], [200, 200, 240, 240]], [0.9, 0.9], frame_id=3))
        out = bt.update(_dets([[10, 10, 50, 50], [200, 200, 240, 240]], [0.9, 0.9], frame_id=4))
        assert sorted(out.track_ids.tolist()) == [1, 3]


class TestDetIndices:
    def test_det_indices_point_to_original_rows_for_low_score_matches(self):
        cfg = ByteTrackConfig(lifecycle=LifecycleConfig(min_hits=1, tentative_max_age=1))
        bt = ByteTrackTracker(cfg)
        bt.update(_dets([[10, 10, 50, 50]], [0.9], frame_id=0))
        # Row 0 is a low-score continuation; row 1 is an unrelated high-score det.
        dets1 = _dets([[12, 12, 52, 52], [300, 300, 340, 340]], [0.3, 0.9], frame_id=1)
        out1 = bt.update(dets1)
        assert out1.det_indices is not None
        idx_of_track_1 = int(out1.det_indices[out1.track_ids.tolist().index(1)])
        assert idx_of_track_1 == 0

    def test_empty_frame_has_empty_det_indices(self, fast_confirm_config):
        bt = ByteTrackTracker(fast_confirm_config)
        out = bt.update(_dets([], [], frame_id=0))
        assert out.det_indices is not None
        assert len(out.det_indices) == 0
