# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for getitrack.visualization.annotator."""

import numpy as np

from getitrack.core.detection import TrackedDetections
from getitrack.core.track import TrackState
from getitrack.io import VideoReader
from getitrack.visualization import TrackAnnotator, VideoAnnotator, color_for_track


def _frame(w=128, h=96) -> np.ndarray:
    return np.full((h, w, 3), 30, dtype=np.uint8)


def _tracked(boxes, frame_id=1) -> TrackedDetections:
    n = len(boxes)
    return TrackedDetections(
        bboxes=np.asarray(boxes, dtype=np.float32),
        scores=np.full(n, 0.9, dtype=np.float32),
        class_ids=np.zeros(n, dtype=np.int64),
        track_ids=np.arange(1, n + 1, dtype=np.int64),
        track_states=np.full(n, TrackState.ACTIVE, dtype=np.int8),
        frame_id=frame_id,
    )


class TestTrackAnnotator:
    def test_returns_modified_copy(self):
        frame = _frame()
        original = frame.copy()
        out = TrackAnnotator().annotate(frame, _tracked([[20, 20, 60, 60]]))
        assert out is not frame
        assert np.array_equal(frame, original)
        assert not np.array_equal(out, frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_empty_tracks_changes_nothing(self):
        frame = _frame()
        out = TrackAnnotator().annotate(frame, TrackedDetections.create_empty(frame_id=1))
        assert np.array_equal(out, frame)

    def test_show_score_false(self):
        out = TrackAnnotator(show_score=False).annotate(_frame(), _tracked([[10, 10, 40, 40]]))
        assert not np.array_equal(out, _frame())

    def test_multiple_tracks(self):
        out = TrackAnnotator().annotate(_frame(), _tracked([[5, 5, 30, 30], [60, 40, 100, 80]]))
        assert not np.array_equal(out, _frame())


class TestColorForTrack:
    def test_deterministic(self):
        assert color_for_track(7) == color_for_track(7)

    def test_distinct_for_consecutive_ids(self):
        colors = {color_for_track(i) for i in range(1, 11)}
        assert len(colors) == 10

    def test_valid_bgr_range(self):
        for i in range(1, 50):
            assert all(0 <= c <= 255 for c in color_for_track(i))


class TestClassNames:
    def test_label_includes_class_name(self):
        ann = TrackAnnotator(class_names=["background", "person", "car"])
        assert ann._label(7, 0.86, 1) == "person #7 0.86"

    def test_unknown_class_id_falls_back_to_numeric(self):
        ann = TrackAnnotator(class_names=["background"])
        assert ann._label(3, 0.5, 42) == "42 #3 0.50"

    def test_no_class_names_keeps_id_only_label(self):
        assert TrackAnnotator(show_score=False)._label(5, 0.9, 1) == "#5"

    def test_annotate_with_class_names_draws(self):
        ann = TrackAnnotator(class_names=["background", "person"])
        out = ann.annotate(_frame(), _tracked([[10, 10, 60, 60]]))
        assert not np.array_equal(out, _frame())


class TestVideoAnnotator:
    def test_writes_annotated_frames(self, tmp_path):
        path = tmp_path / "out.mp4"
        with VideoAnnotator(path, fps=30.0, frame_size=(128, 96)) as out:
            for _ in range(5):
                out.write(_frame(), _tracked([[20, 20, 60, 60]]))
            assert out.frames_written == 5
        with VideoReader(path) as reader:
            frames = list(reader)
        assert len(frames) == 5
        # The stored frames must differ from the raw input: boxes were drawn.
        assert not np.array_equal(frames[0], _frame())

    def test_style_params_forwarded(self, tmp_path):
        with VideoAnnotator(
            tmp_path / "out.mp4",
            fps=30.0,
            frame_size=(128, 96),
            class_names={0: "thing"},
            show_score=False,
            thickness=3,
        ) as out:
            assert out.annotator.class_names == {0: "thing"}
            assert out.annotator.show_score is False
            assert out.annotator.thickness == 3
            out.write(_frame(), _tracked([[10, 10, 50, 50]]))

    def test_path_property(self, tmp_path):
        with VideoAnnotator(tmp_path / "a" / "out.mp4", fps=30.0, frame_size=(128, 96)) as out:
            assert out.path == tmp_path / "a" / "out.mp4"
