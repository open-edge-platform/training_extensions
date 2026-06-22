# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Per-frame detection and tracker-output containers.

`Detections` and `TrackedDetections` are framework-agnostic numpy
dataclasses with eager shape, dtype, and value validation in
``__post_init__``.

Bounding boxes are always ``(N, 4)`` in ``xyxy`` order
(``[x1, y1, x2, y2]`` with ``x1 < x2`` and ``y1 < y2``) in absolute pixel
coordinates of the source image. Other formats (``cxcywh``, normalised,
etc.) are converted at the boundary.

Required dtypes (enforced, not coerced, so silent casts cannot mask
precision drift downstream):

- ``bboxes``: float32
- ``scores``: float32
- ``class_ids``: int64
- ``track_ids``: int64
- ``track_states``: int8
- ``embeddings``: float32
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from getitrack.core.validation import (
    BBOX_COLS,
    validate_bboxes,
    validate_dtypes,
    validate_row_aligned,
    validate_scores,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class TrackState(StrEnum):
    """Lifecycle states of a single track.

    State transitions are defined in `getitrack.core.track`.
    `TrackedDetections.track_states` stores values as int8 ordinals where
    the ordinal is the member's 0-based position in declaration order.
    """

    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"

    @classmethod
    def from_ordinal(cls, ordinal: int) -> TrackState:
        """Return the member at a given 0-based ordinal."""
        return list(cls)[ordinal]

    def ordinal(self) -> int:
        """Return this member's 0-based ordinal."""
        return list(type(self)).index(self)


@dataclass
class Detections:
    """One frame's detector output, before tracking.

    A tracker consumes a `Detections` and emits a `TrackedDetections`
    that adds track ids and lifecycle states.

    Attributes:
        bboxes: ``(N, 4)`` float32 array in ``xyxy`` order.
        scores: ``(N,)`` float32 confidence scores in ``[0, 1]``.
        class_ids: ``(N,)`` int64 class identifiers.
        frame_id: Index of the frame this batch belongs to.
        embeddings: Optional ``(N, D)`` float32 appearance features.
    """

    bboxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    frame_id: int
    embeddings: np.ndarray | None = None

    def __post_init__(self) -> None:
        validate_bboxes(self.bboxes)
        validate_row_aligned(
            n=self.bboxes.shape[0],
            scores=self.scores,
            class_ids=self.class_ids,
            embeddings=self.embeddings,
        )
        validate_scores(self.scores)
        validate_dtypes(
            bboxes=(self.bboxes, np.float32),
            scores=(self.scores, np.float32),
            class_ids=(self.class_ids, np.int64),
            embeddings=(self.embeddings, np.float32) if self.embeddings is not None else None,
        )

    def __len__(self) -> int:
        return int(self.bboxes.shape[0])

    def filter_by_score(self, threshold: float) -> Detections:
        """Return rows whose score is at least ``threshold``."""
        keep = self.scores >= threshold
        return self._index(keep)

    def split_by_score(self, threshold: float) -> tuple[Detections, Detections]:
        """Return ``(high, low)`` partitions split at ``threshold``.

        Used by ByteTrack's two-stage association.
        """
        high = self.scores >= threshold
        return self._index(high), self._index(~high)

    def filter_by_class(self, class_ids: Sequence[int]) -> tuple[Detections, np.ndarray]:
        """Return ``(filtered, source_rows)`` for rows whose class is in ``class_ids``.

        ``source_rows`` holds the indices of the kept rows in this
        `Detections`, letting callers map filtered-row positions back to
        input rows without recomputing the membership test.
        """
        wanted = np.asarray(list(class_ids), dtype=np.int64)
        source_rows = np.flatnonzero(np.isin(self.class_ids, wanted))
        return self._index(source_rows), source_rows

    def _index(self, mask: np.ndarray) -> Detections:
        return Detections(
            bboxes=self.bboxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            frame_id=self.frame_id,
            embeddings=None if self.embeddings is None else self.embeddings[mask],
        )

    @classmethod
    def create_empty(cls, frame_id: int) -> Detections:
        """Construct an empty `Detections` for the given frame."""
        return cls(
            bboxes=np.empty((0, BBOX_COLS), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=np.int64),
            frame_id=frame_id,
        )


@dataclass
class TrackedDetections:
    """One frame's tracker output.

    Each row carries a track id and a lifecycle state in addition to the
    detection fields. The ``interpolated`` flag is set by the
    post-processing interpolator.

    Attributes:
        bboxes: ``(N, 4)`` float32 array in ``xyxy`` order.
        scores: ``(N,)`` float32 confidence scores in ``[0, 1]``.
        class_ids: ``(N,)`` int64 class identifiers.
        track_ids: ``(N,)`` int64 stable per-object identifiers.
        track_states: ``(N,)`` int8 ordinals of `TrackState`.
        frame_id: Index of the frame this batch belongs to.
        det_indices: Optional ``(N,)`` int64 row indices into the frame's
            input `Detections`, identifying the detection that produced
            each row. -1 marks rows without a source detection (e.g.
            interpolated boxes). Lets callers re-attach per-detection data
            such as embeddings or masks to tracks.
        interpolated: Optional ``(N,)`` bool flag marking interpolated bboxes.
    """

    bboxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    track_ids: np.ndarray
    track_states: np.ndarray
    frame_id: int
    det_indices: np.ndarray | None = field(default=None)
    interpolated: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        validate_bboxes(self.bboxes)
        validate_row_aligned(
            n=self.bboxes.shape[0],
            scores=self.scores,
            class_ids=self.class_ids,
            track_ids=self.track_ids,
            track_states=self.track_states,
            det_indices=self.det_indices,
            interpolated=self.interpolated,
        )
        validate_scores(self.scores)
        _validate_track_states(self.track_states)
        validate_dtypes(
            bboxes=(self.bboxes, np.float32),
            scores=(self.scores, np.float32),
            class_ids=(self.class_ids, np.int64),
            track_ids=(self.track_ids, np.int64),
            track_states=(self.track_states, np.int8),
            det_indices=(self.det_indices, np.int64) if self.det_indices is not None else None,
            interpolated=(self.interpolated, np.bool_) if self.interpolated is not None else None,
        )

    def __len__(self) -> int:
        return int(self.bboxes.shape[0])

    def active_only(self) -> TrackedDetections:
        """Return rows whose state is ``TrackState.ACTIVE``."""
        keep = self.track_states == TrackState.ACTIVE.ordinal()
        return TrackedDetections(
            bboxes=self.bboxes[keep],
            scores=self.scores[keep],
            class_ids=self.class_ids[keep],
            track_ids=self.track_ids[keep],
            track_states=self.track_states[keep],
            frame_id=self.frame_id,
            det_indices=None if self.det_indices is None else self.det_indices[keep],
            interpolated=None if self.interpolated is None else self.interpolated[keep],
        )

    def to_string_states(self) -> list[str]:
        """Return per-row state names as lowercase strings."""
        return [TrackState.from_ordinal(int(o)).value for o in self.track_states]

    @classmethod
    def create_empty(cls, frame_id: int) -> TrackedDetections:
        """Construct an empty `TrackedDetections` for the given frame."""
        return cls(
            bboxes=np.empty((0, BBOX_COLS), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=np.int64),
            track_ids=np.empty((0,), dtype=np.int64),
            track_states=np.empty((0,), dtype=np.int8),
            frame_id=frame_id,
        )


def _validate_track_states(states: np.ndarray) -> None:
    """Raise if any track-state ordinal is outside the `TrackState` range."""
    if states.size == 0:
        return
    n_states = len(TrackState)
    if states.min() < 0 or states.max() >= n_states:
        msg = f"track_states ordinals must be in [0, {n_states}); got min={states.min()} max={states.max()}"
        raise ValueError(msg)
