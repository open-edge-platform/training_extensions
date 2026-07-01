# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Adapters between third-party detector frameworks and getitrack types.

Each adapter is a `DetectionAdapter` subclass wrapping one detector
instance. Framework imports stay lazy or duck-typed, so no adapter adds
a hard dependency to getitrack.
"""

from getitrack.adapters.base import DetectionAdapter
from getitrack.adapters.geti import GetiAdapter

__all__ = ["DetectionAdapter", "GetiAdapter"]
