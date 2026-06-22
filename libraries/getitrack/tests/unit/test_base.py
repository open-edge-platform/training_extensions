# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BaseTracker ABC and the algorithm registry."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from getitrack.config import AlgorithmType, TrackerConfig
from getitrack.core.base import BaseTracker
from getitrack.core.detection import Detections, TrackedDetections
from getitrack.core.registry import ALGORITHM_REGISTRY, register_algorithm


class _Recording(BaseTracker):
    """Trivial tracker that echoes inputs back with sequential ids."""

    def _update_impl(self, detections: Detections) -> TrackedDetections:
        n = len(detections)
        return TrackedDetections(
            bboxes=detections.bboxes,
            scores=detections.scores,
            class_ids=detections.class_ids,
            track_ids=np.array([self._allocate_id() for _ in range(n)], dtype=np.int64),
            track_states=np.zeros((n,), dtype=np.int8),
            frame_id=detections.frame_id,
            det_indices=np.arange(n, dtype=np.int64),
        )


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Snapshot and restore the registry around each test.

    Also restores loguru's default-disabled state for ``getitrack``, since
    a ``verbose`` tracker enables it process-wide.
    """
    snapshot = dict(ALGORITHM_REGISTRY)
    ALGORITHM_REGISTRY.clear()
    logger.disable("getitrack")
    yield
    ALGORITHM_REGISTRY.clear()
    ALGORITHM_REGISTRY.update(snapshot)
    logger.disable("getitrack")


def _register_dummy(name: str = "bytetrack") -> type[BaseTracker]:
    @register_algorithm(name)
    class _Dummy(_Recording):
        algorithm_name = name

    return _Dummy


class TestRegistry:
    def test_register_and_lookup(self):
        cls = _register_dummy("bytetrack")
        assert ALGORITHM_REGISTRY["bytetrack"] is cls

    def test_duplicate_registration_raises(self):
        _register_dummy("bytetrack")
        with pytest.raises(ValueError, match="already registered"):
            _register_dummy("bytetrack")


class TestFromConfig:
    def test_from_trackerconfig(self):
        _register_dummy("bytetrack")
        tracker = BaseTracker.from_config(TrackerConfig())
        assert isinstance(tracker, BaseTracker)

    def test_from_dict(self):
        _register_dummy("bytetrack")
        tracker = BaseTracker.from_config({"algorithm": "bytetrack"})
        assert isinstance(tracker, BaseTracker)

    def test_from_yaml_path(self, tmp_path: Path):
        _register_dummy("bytetrack")
        p = tmp_path / "c.yaml"
        TrackerConfig().to_yaml(p)
        tracker = BaseTracker.from_config(p)
        assert isinstance(tracker, BaseTracker)

    def test_from_yaml_str_path(self, tmp_path: Path):
        _register_dummy("bytetrack")
        p = tmp_path / "c.yaml"
        TrackerConfig().to_yaml(p)
        tracker = BaseTracker.from_config(str(p))
        assert isinstance(tracker, BaseTracker)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(KeyError, match="unknown algorithm"):
            BaseTracker.from_config(TrackerConfig(algorithm=AlgorithmType.OCSORT))

    def test_bad_config_type_raises(self):
        with pytest.raises(TypeError, match="unsupported config type"):
            BaseTracker.from_config(123)  # type: ignore[arg-type]


class TestUpdateAndReset:
    def test_update_assigns_monotonic_ids(self):
        _register_dummy("bytetrack")
        t = BaseTracker.from_config(TrackerConfig())
        dets = Detections(
            bboxes=np.zeros((2, 4), dtype=np.float32),
            scores=np.array([0.9, 0.5], dtype=np.float32),
            class_ids=np.array([0, 0], dtype=np.int64),
            frame_id=0,
        )
        out = t.update(dets)
        assert out.track_ids.tolist() == [1, 2]
        out2 = t.update(dets)
        assert out2.track_ids.tolist() == [3, 4]

    def test_reset_resets_id_counter(self):
        _register_dummy("bytetrack")
        t = BaseTracker.from_config(TrackerConfig())
        t.update(Detections.create_empty(0))
        t._next_id = 7
        t.reset()
        assert t._next_id == 1
        assert t._frame_id is None

    def test_abstract_update_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseTracker(TrackerConfig())  # type: ignore[abstract]


class TestClassFilter:
    def _dets(self) -> Detections:
        return Detections(
            bboxes=np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]], dtype=np.float32),
            scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
            class_ids=np.array([1, 2, 1], dtype=np.int64),
            frame_id=0,
        )

    def _tracker(self, class_filter) -> BaseTracker:
        _register_dummy("bytetrack")
        cfg = TrackerConfig()
        cfg.association.class_filter = class_filter
        return BaseTracker.from_config(cfg)

    def test_excluded_classes_never_reach_algorithm(self):
        out = self._tracker([1]).update(self._dets())
        assert len(out) == 2
        assert out.class_ids.tolist() == [1, 1]

    def test_det_indices_remap_to_input_rows(self):
        out = self._tracker([1]).update(self._dets())
        assert out.det_indices is not None
        assert out.det_indices.tolist() == [0, 2]

    def test_none_filter_tracks_all(self):
        out = self._tracker(None).update(self._dets())
        assert len(out) == 3
        assert out.det_indices is not None
        assert out.det_indices.tolist() == [0, 1, 2]

    def test_remap_preserves_unmatched_rows(self):
        source_rows = np.array([0, 2], dtype=np.int64)
        remapped = BaseTracker._remap_det_indices(source_rows, np.array([0, -1, 1], dtype=np.int64))
        assert remapped.tolist() == [0, -1, 2]


class TestVerboseLogging:
    def _dets(self) -> Detections:
        return Detections(
            bboxes=np.zeros((2, 4), dtype=np.float32),
            scores=np.array([0.9, 0.3], dtype=np.float32),
            class_ids=np.array([3, 8], dtype=np.int64),
            frame_id=4,
        )

    def _capture(self, level: str = "INFO") -> tuple[list[str], int]:
        messages: list[str] = []
        sink_id = logger.add(messages.append, level=level, format="{message}")
        return messages, sink_id

    def test_verbose_logs_frame_summary(self):
        _register_dummy("bytetrack")
        messages, sink_id = self._capture()
        try:
            BaseTracker.from_config(TrackerConfig(verbose=True)).update(self._dets())
        finally:
            logger.remove(sink_id)
        assert len(messages) == 1
        msg = messages[0]
        assert "frame    4" in msg
        assert "2 detections (1 high-score)" in msg
        assert "1:3:0.90, 2:8:0.30" in msg

    def test_default_is_silent(self):
        _register_dummy("bytetrack")
        messages, sink_id = self._capture(level="DEBUG")
        try:
            BaseTracker.from_config(TrackerConfig()).update(self._dets())
        finally:
            logger.remove(sink_id)
        assert not messages

    def test_verbose_enables_package_logging(self):
        _register_dummy("bytetrack")
        messages, sink_id = self._capture()
        try:
            # A non-verbose tracker stays suppressed even with a sink attached.
            BaseTracker.from_config(TrackerConfig()).update(self._dets())
            assert not messages
            # A verbose tracker lifts the package-level suppression.
            BaseTracker.from_config(TrackerConfig(verbose=True)).update(self._dets())
            assert len(messages) == 1
        finally:
            logger.remove(sink_id)
