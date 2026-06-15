# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for Detections / TrackedDetections / TrackState."""

import numpy as np
import pytest

from getitrack.core.detection import Detections, TrackedDetections, TrackState


def _dets(n: int = 3, frame_id: int = 0) -> Detections:
    return Detections(
        bboxes=np.array([[i, i, i + 10, i + 10] for i in range(n)], dtype=np.float32),
        scores=np.array([0.9, 0.5, 0.2][:n], dtype=np.float32),
        class_ids=np.array([0, 1, 0][:n], dtype=np.int64),
        frame_id=frame_id,
    )


class TestTrackState:
    def test_ordinals_are_dense_zero_based(self):
        ordinals = [s.ordinal() for s in TrackState]
        assert ordinals == list(range(len(TrackState)))

    def test_from_ordinal_roundtrip(self):
        for s in TrackState:
            assert TrackState.from_ordinal(s.ordinal()) is s


class TestDetections:
    def test_construct_valid(self):
        d = _dets(3)
        assert len(d) == 3
        assert d.frame_id == 0
        assert d.embeddings is None

    def test_create_empty(self):
        d = Detections.create_empty(frame_id=7)
        assert len(d) == 0
        assert d.frame_id == 7

    def test_bbox_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="bboxes must have shape"):
            Detections(
                bboxes=np.array([1, 2, 3, 4], dtype=np.float32),
                scores=np.array([0.5], dtype=np.float32),
                class_ids=np.array([0], dtype=np.int64),
                frame_id=0,
            )

    def test_row_mismatch_raises(self):
        with pytest.raises(ValueError, match="scores has"):
            Detections(
                bboxes=np.zeros((3, 4), dtype=np.float32),
                scores=np.array([0.5, 0.5], dtype=np.float32),
                class_ids=np.array([0, 0, 0], dtype=np.int64),
                frame_id=0,
            )

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="scores must be in"):
            Detections(
                bboxes=np.zeros((1, 4), dtype=np.float32),
                scores=np.array([1.5], dtype=np.float32),
                class_ids=np.array([0], dtype=np.int64),
                frame_id=0,
            )

    def test_filter_by_score(self):
        d = _dets(3).filter_by_score(0.4)
        assert len(d) == 2
        assert d.scores.min() >= 0.4

    def test_split_by_score(self):
        high, low = _dets(3).split_by_score(0.4)
        assert len(high) == 2
        assert len(low) == 1
        assert low.scores[0] == pytest.approx(0.2)

    def test_filter_by_class(self):
        d = _dets(3).filter_by_class([1])
        assert len(d) == 1
        assert d.class_ids[0] == 1

    def test_embeddings_roundtrip_through_filter(self):
        embeds = np.random.default_rng(0).random((3, 8), dtype=np.float32)
        d = Detections(
            bboxes=np.zeros((3, 4), dtype=np.float32),
            scores=np.array([0.9, 0.1, 0.9], dtype=np.float32),
            class_ids=np.array([0, 0, 0], dtype=np.int64),
            frame_id=0,
            embeddings=embeds,
        )
        kept = d.filter_by_score(0.5)
        assert kept.embeddings is not None
        assert kept.embeddings.shape == (2, 8)
        np.testing.assert_array_equal(kept.embeddings, embeds[[0, 2]])

    def test_embeddings_row_mismatch_raises(self):
        with pytest.raises(ValueError, match="embeddings has"):
            Detections(
                bboxes=np.zeros((2, 4), dtype=np.float32),
                scores=np.zeros(2, dtype=np.float32),
                class_ids=np.zeros(2, dtype=np.int64),
                frame_id=0,
                embeddings=np.zeros((3, 4), dtype=np.float32),
            )

    def test_bboxes_wrong_dtype_raises(self):
        with pytest.raises(TypeError, match="bboxes must have dtype float32"):
            Detections(
                bboxes=np.zeros((1, 4), dtype=np.float64),
                scores=np.zeros(1, dtype=np.float32),
                class_ids=np.zeros(1, dtype=np.int64),
                frame_id=0,
            )

    def test_scores_wrong_dtype_raises(self):
        with pytest.raises(TypeError, match="scores must have dtype float32"):
            Detections(
                bboxes=np.zeros((1, 4), dtype=np.float32),
                scores=np.zeros(1, dtype=np.float64),
                class_ids=np.zeros(1, dtype=np.int64),
                frame_id=0,
            )

    def test_class_ids_wrong_dtype_raises(self):
        with pytest.raises(TypeError, match="class_ids must have dtype int64"):
            Detections(
                bboxes=np.zeros((1, 4), dtype=np.float32),
                scores=np.zeros(1, dtype=np.float32),
                class_ids=np.zeros(1, dtype=np.int32),
                frame_id=0,
            )

    def test_embeddings_wrong_dtype_raises(self):
        with pytest.raises(TypeError, match="embeddings must have dtype float32"):
            Detections(
                bboxes=np.zeros((1, 4), dtype=np.float32),
                scores=np.zeros(1, dtype=np.float32),
                class_ids=np.zeros(1, dtype=np.int64),
                frame_id=0,
                embeddings=np.zeros((1, 8), dtype=np.float64),
            )


class TestTrackedDetections:
    def _td(self, n: int = 2) -> TrackedDetections:
        return TrackedDetections(
            bboxes=np.zeros((n, 4), dtype=np.float32),
            scores=np.array([0.9, 0.5][:n], dtype=np.float32),
            class_ids=np.array([0, 1][:n], dtype=np.int64),
            track_ids=np.array([1, 2][:n], dtype=np.int64),
            track_states=np.array([TrackState.ACTIVE.ordinal(), TrackState.LOST.ordinal()][:n], dtype=np.int8),
            frame_id=0,
        )

    def test_construct_valid(self):
        td = self._td(2)
        assert len(td) == 2

    def test_create_empty(self):
        td = TrackedDetections.create_empty(frame_id=4)
        assert len(td) == 0
        assert td.frame_id == 4

    def test_active_only(self):
        td = self._td(2).active_only()
        assert len(td) == 1
        assert td.track_ids[0] == 1

    def test_to_string_states(self):
        td = self._td(2)
        assert td.to_string_states() == ["active", "lost"]

    def test_invalid_state_ordinal_raises(self):
        with pytest.raises(ValueError, match="track_states ordinals"):
            TrackedDetections(
                bboxes=np.zeros((1, 4), dtype=np.float32),
                scores=np.zeros(1, dtype=np.float32),
                class_ids=np.zeros(1, dtype=np.int64),
                track_ids=np.zeros(1, dtype=np.int64),
                track_states=np.array([99], dtype=np.int8),
                frame_id=0,
            )

    def test_track_ids_wrong_dtype_raises(self):
        with pytest.raises(TypeError, match="track_ids must have dtype int64"):
            TrackedDetections(
                bboxes=np.zeros((1, 4), dtype=np.float32),
                scores=np.zeros(1, dtype=np.float32),
                class_ids=np.zeros(1, dtype=np.int64),
                track_ids=np.zeros(1, dtype=np.int32),
                track_states=np.zeros(1, dtype=np.int8),
                frame_id=0,
            )

    def test_track_states_wrong_dtype_raises(self):
        with pytest.raises(TypeError, match="track_states must have dtype int8"):
            TrackedDetections(
                bboxes=np.zeros((1, 4), dtype=np.float32),
                scores=np.zeros(1, dtype=np.float32),
                class_ids=np.zeros(1, dtype=np.int64),
                track_ids=np.zeros(1, dtype=np.int64),
                track_states=np.zeros(1, dtype=np.int16),
                frame_id=0,
            )

    def test_interpolated_propagates_through_active_only(self):
        td = TrackedDetections(
            bboxes=np.zeros((2, 4), dtype=np.float32),
            scores=np.array([0.9, 0.5], dtype=np.float32),
            class_ids=np.array([0, 1], dtype=np.int64),
            track_ids=np.array([1, 2], dtype=np.int64),
            track_states=np.array([TrackState.ACTIVE.ordinal(), TrackState.LOST.ordinal()], dtype=np.int8),
            frame_id=0,
            interpolated=np.array([False, True]),
        )
        active = td.active_only()
        assert active.interpolated is not None
        assert active.interpolated.tolist() == [False]


class TestDetIndices:
    def test_det_indices_wrong_dtype_raises(self):
        with pytest.raises(TypeError, match="det_indices must have dtype int64"):
            TrackedDetections(
                bboxes=np.zeros((1, 4), dtype=np.float32),
                scores=np.zeros(1, dtype=np.float32),
                class_ids=np.zeros(1, dtype=np.int64),
                track_ids=np.zeros(1, dtype=np.int64),
                track_states=np.zeros(1, dtype=np.int8),
                frame_id=0,
                det_indices=np.zeros(1, dtype=np.int32),
            )

    def test_det_indices_row_mismatch_raises(self):
        with pytest.raises(ValueError, match="det_indices"):
            TrackedDetections(
                bboxes=np.zeros((2, 4), dtype=np.float32),
                scores=np.zeros(2, dtype=np.float32),
                class_ids=np.zeros(2, dtype=np.int64),
                track_ids=np.zeros(2, dtype=np.int64),
                track_states=np.zeros(2, dtype=np.int8),
                frame_id=0,
                det_indices=np.zeros(3, dtype=np.int64),
            )

    def test_det_indices_propagate_through_active_only(self):
        td = TrackedDetections(
            bboxes=np.zeros((2, 4), dtype=np.float32),
            scores=np.array([0.9, 0.5], dtype=np.float32),
            class_ids=np.array([0, 1], dtype=np.int64),
            track_ids=np.array([1, 2], dtype=np.int64),
            track_states=np.array([TrackState.ACTIVE.ordinal(), TrackState.LOST.ordinal()], dtype=np.int8),
            frame_id=0,
            det_indices=np.array([5, -1], dtype=np.int64),
        )
        active = td.active_only()
        assert active.det_indices is not None
        assert active.det_indices.tolist() == [5]
