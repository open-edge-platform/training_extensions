# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Concrete tracker algorithms.

Importing this package registers every algorithm into
`getitrack.core.registry.ALGORITHM_REGISTRY` as a side effect, so that
`BaseTracker.from_config` can dispatch by name without explicit wiring.
"""

from getitrack.algorithms.bytetrack import ByteTrackTracker

__all__ = ["ByteTrackTracker"]
