# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BaseTracker ABC and the algorithm registry."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from getitrack.config import AlgorithmType, TrackerConfig
from getitrack.core.base import ALGORITHM_REGISTRY, BaseTracker, register_algorithm
from getitrack.core.detection import Detections, TrackedDetections


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
    """Snapshot and restore the registry around each test."""
    snapshot = dict(ALGORITHM_REGISTRY)
    ALGORITHM_REGISTRY.clear()
    yield
    ALGORITHM_REGISTRY.clear()
    ALGORITHM_REGISTRY.update(snapshot)


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
        remapped = BaseTracker._remap_det_indices(self._dets(), [1], np.array([0, -1, 1], dtype=np.int64))
        assert remapped.tolist() == [0, -1, 2]


class TestVerboseLogging:
    def _dets(self) -> Detections:
        return Detections(
            bboxes=np.zeros((2, 4), dtype=np.float32),
            scores=np.array([0.9, 0.3], dtype=np.float32),
            class_ids=np.array([3, 8], dtype=np.int64),
            frame_id=4,
        )

    def test_verbose_logs_frame_summary(self, caplog):
        _register_dummy("bytetrack")
        t = BaseTracker.from_config(TrackerConfig(verbose=True))
        with caplog.at_level("INFO", logger="getitrack"):
            t.update(self._dets())
        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "frame    4" in msg
        assert "2 detections (1 high-score)" in msg
        assert "1:3:0.90, 2:8:0.30" in msg

    def test_default_is_silent(self, caplog):
        _register_dummy("bytetrack")
        t = BaseTracker.from_config(TrackerConfig())
        with caplog.at_level("INFO", logger="getitrack"):
            t.update(self._dets())
        assert not caplog.records


class TestVerboseHandler:
    def _clean(self) -> None:
        pkg = logging.getLogger("getitrack")
        for h in list(pkg.handlers):
            pkg.removeHandler(h)
        pkg.propagate = True
        pkg.setLevel(logging.NOTSET)

    def test_attaches_handler_when_logging_unconfigured(self, monkeypatch):
        self._clean()
        monkeypatch.setattr(logging.getLogger(), "handlers", [])
        _register_dummy("bytetrack")
        BaseTracker.from_config(TrackerConfig(verbose=True))
        pkg = logging.getLogger("getitrack")
        assert len(pkg.handlers) == 1
        assert pkg.level == logging.INFO
        assert pkg.propagate is False
        self._clean()

    def test_respects_application_logging_config(self, monkeypatch):
        self._clean()
        monkeypatch.setattr(logging.getLogger(), "handlers", [logging.NullHandler()])
        _register_dummy("bytetrack")
        BaseTracker.from_config(TrackerConfig(verbose=True))
        assert not logging.getLogger("getitrack").handlers
        self._clean()

    def test_no_handler_when_not_verbose(self, monkeypatch):
        self._clean()
        monkeypatch.setattr(logging.getLogger(), "handlers", [])
        _register_dummy("bytetrack")
        BaseTracker.from_config(TrackerConfig())
        assert not logging.getLogger("getitrack").handlers
        self._clean()
