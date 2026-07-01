# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for getitrack.io.video."""

from pathlib import Path

import numpy as np
import pytest

from getitrack.io import VideoReader, VideoWriter

_W, _H = 64, 48


def _solid_frame(value: int) -> np.ndarray:
    return np.full((_H, _W, 3), value, dtype=np.uint8)


def _write_video(path, n_frames=10, fps=30.0) -> Path:
    with VideoWriter(path, fps=fps, frame_size=(_W, _H)) as writer:
        for i in range(n_frames):
            writer.write(_solid_frame((i * 20) % 255))
    return path


class TestVideoReader:
    def test_roundtrip_frame_count_and_shape(self, tmp_path):
        path = _write_video(tmp_path / "clip.mp4", n_frames=10)
        with VideoReader(path) as reader:
            frames = list(reader)
        assert len(frames) == 10
        assert all(f.shape == (_H, _W, 3) for f in frames)
        assert all(f.dtype == np.uint8 for f in frames)

    def test_metadata_properties(self, tmp_path):
        path = _write_video(tmp_path / "clip.mp4", n_frames=10, fps=30.0)
        with VideoReader(path) as reader:
            assert reader.width == _W
            assert reader.height == _H
            assert reader.frame_count == 10
            assert reader.fps == pytest.approx(30.0, abs=1.0)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            VideoReader(tmp_path / "nope.mp4")

    def test_unreadable_file_raises(self, tmp_path):
        bogus = tmp_path / "bogus.mp4"
        bogus.write_text("this is not a video")
        with pytest.raises(ValueError, match="could not open"):
            VideoReader(bogus)


class TestVideoWriter:
    def test_counts_written_frames(self, tmp_path):
        with VideoWriter(tmp_path / "out.mp4", fps=30.0, frame_size=(_W, _H)) as writer:
            writer.write(_solid_frame(0))
            writer.write(_solid_frame(100))
            assert writer.frames_written == 2

    def test_wrong_frame_shape_raises(self, tmp_path):
        with VideoWriter(tmp_path / "out.mp4", fps=30.0, frame_size=(_W, _H)) as writer:
            bad = np.zeros((_H + 1, _W, 3), dtype=np.uint8)
            with pytest.raises(ValueError, match="does not match"):
                writer.write(bad)

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "out.mp4"
        with VideoWriter(nested, fps=30.0, frame_size=(_W, _H)) as writer:
            writer.write(_solid_frame(50))
        assert nested.is_file()
