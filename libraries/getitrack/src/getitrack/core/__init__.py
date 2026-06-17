# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Core data types and tracker interface."""

from getitrack.core.base import ALGORITHM_REGISTRY, BaseTracker, register_algorithm
from getitrack.core.detection import Detections, TrackedDetections, TrackState
from getitrack.core.track import Track

__all__ = [
    "ALGORITHM_REGISTRY",
    "BaseTracker",
    "Detections",
    "Track",
    "TrackState",
    "TrackedDetections",
    "register_algorithm",
]
