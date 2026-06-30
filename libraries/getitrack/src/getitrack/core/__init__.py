# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Core data types and tracker interface."""

from getitrack.core.base import BaseTracker
from getitrack.core.detection import Detections, TrackedDetections
from getitrack.core.registry import ALGORITHM_REGISTRY, register_algorithm
from getitrack.core.track import Track, TrackState

__all__ = [
    "ALGORITHM_REGISTRY",
    "BaseTracker",
    "Detections",
    "Track",
    "TrackState",
    "TrackedDetections",
    "register_algorithm",
]
