# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Adapter interface between detector frameworks and getitrack.

A `DetectionAdapter` owns one detector instance and translates between
raw BGR frames and getitrack `Detections`, so trackers and pipelines
consume every framework through the same two members: `detect` and
`class_names`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from getitrack.core.detection import Detections


class DetectionAdapter(ABC):
    """Wraps one detector behind a framework-agnostic interface.

    Concrete adapters keep their framework imports lazy or duck-typed so
    getitrack stays installable without the framework.
    """

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, frame_id: int) -> Detections:
        """Run the detector on one BGR frame.

        Args:
            frame_bgr: ``(H, W, 3)`` uint8 frame in BGR order.
            frame_id: Frame index to stamp on the returned `Detections`.

        Returns:
            `Detections` in original frame coordinates.
        """

    @property
    def class_names(self) -> dict[int, str] | None:
        """Class-id-to-name mapping carried by the detector, if any.

        Returns None when the framework does not expose meaningful names,
        in which case callers supply their own table
        """
        return None
