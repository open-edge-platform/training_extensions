# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""ByteTrack: two-stage detection-to-track association.

Tracks are held in one `_tracks` dict keyed by id, with lifecycle state on each
`Track` and Kalman state on the tracker in `_kalman_states`. Duplicate
suppression is class-aware when `match_class_only` is set.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box" (ECCV 2022).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from getitrack.config import AlgorithmType, TrackerConfig
from getitrack.core.base import BaseTracker
from getitrack.core.detection import Detections, TrackedDetections
from getitrack.core.registry import register_algorithm
from getitrack.core.track import Track, TrackState
from getitrack.matching import fuse_score, iou_distance, linear_assignment
from getitrack.motion import KalmanFilter, xyah_to_xyxy, xyxy_to_xyah

if TYPE_CHECKING:
    from getitrack.config import LifecycleConfig

_UNMATCHABLE_COST = np.nextafter(np.float32(1.0), np.float32(2.0))

# Cost limits for the second-stage and tentative association passes.
_SECOND_STAGE_COST_LIMIT = 0.5
_TENTATIVE_COST_LIMIT = 0.7

# Margin above the high/low split a detection must clear to spawn a new track.
_NEW_TRACK_MARGIN = 0.1
# IoU distance below which an ACTIVE and a LOST track are treated as duplicates.
_DUPLICATE_IOU_DIST = 0.15


class ByteTrackConfig(TrackerConfig):
    """ByteTrack-specific configuration."""

    # Literal pins the value in the JSON schema and typing.
    algorithm: Literal[AlgorithmType.BYTETRACK] = AlgorithmType.BYTETRACK  # pyrefly: ignore[bad-override]
    """Algorithm identifier; fixed to ``bytetrack``."""

    match_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    """Maximum assignment cost accepted when matching detections to tracks.
    The cost is ``1 - IoU``, score-fused to ``1 - IoU * score`` where fusion
    applies, so larger values accept weaker overlaps."""

    high_score_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    """High/low detection split for two-stage association (ByteTrack's ``track_thresh``).
    Spawning a new track additionally requires a score 0.1 above this value."""

    match_class_only: bool = True
    """Restrict matching to detection-track pairs that share a class id."""

    @model_validator(mode="after")
    def _check_thresholds(self) -> ByteTrackConfig:
        """Reject thresholds that contradict the high/low split or new-track gate."""
        if self.score_threshold >= self.high_score_threshold:
            msg = "score_threshold must be below high_score_threshold"
            raise ValueError(msg)
        if self.high_score_threshold + _NEW_TRACK_MARGIN > 1.0:
            msg = f"high_score_threshold must be <= {1.0 - _NEW_TRACK_MARGIN} to leave room for the new-track margin"
            raise ValueError(msg)
        return self


@register_algorithm("bytetrack", config=ByteTrackConfig)
class ByteTrackTracker(BaseTracker[ByteTrackConfig]):
    """ByteTrack multi-object tracker.

    Two-stage association: confirmed tracks (ACTIVE + LOST) are matched
    against high-score detections with score fusion, then unmatched ACTIVE
    tracks are matched against low-score detections to recover from brief
    mis-detections. TENTATIVE tracks compete for leftover high-score
    detections; a remaining high-score detection spawns a new track once its
    score clears the high/low split by 0.1. ``update`` returns the ACTIVE
    tracks for the frame.
    """

    algorithm_name: ClassVar[str] = "bytetrack"

    def __init__(self, config: ByteTrackConfig) -> None:
        super().__init__(config)
        self._kalman = KalmanFilter.from_config(config.motion)
        self._tracks: dict[int, Track] = {}
        self._kalman_states: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._first_frame_id: int | None = None
        # track_id -> row index into this frame's input Detections.
        self._frame_det_index: dict[int, int] = {}

    def reset(self) -> None:  # noqa: D102
        super().reset()
        self._tracks.clear()
        self._kalman_states.clear()
        self._first_frame_id = None
        self._frame_det_index.clear()

    def _update_impl(self, detections: Detections) -> TrackedDetections:
        """Run one ByteTrack iteration and return the active set."""
        self._frame_id = detections.frame_id
        if self._first_frame_id is None:
            self._first_frame_id = detections.frame_id
        self._frame_det_index.clear()
        cfg = self.config
        lifecycle = cfg.lifecycle

        # high = score > high split; low = score_threshold < score < high split.
        # Scores exactly on a bound are dropped.
        scores = detections.scores
        high_src = np.flatnonzero(scores > cfg.high_score_threshold)
        low_src = np.flatnonzero((scores > cfg.score_threshold) & (scores < cfg.high_score_threshold))
        high_dets = _subset(detections, high_src)
        low_dets = _subset(detections, low_src)

        self._predict_all()

        tentative_ids = [tid for tid, t in self._tracks.items() if t.state == TrackState.TENTATIVE]
        confirmed_ids = [tid for tid, t in self._tracks.items() if t.state in {TrackState.ACTIVE, TrackState.LOST}]

        # First association: confirmed tracks (ACTIVE + LOST) vs high-score detections.
        matches_a, unmatched_track_a, unmatched_det_a = self._associate(
            confirmed_ids,
            high_dets,
            cfg.match_threshold,
            apply_fuse_score=True,
        )
        matched_high_dets: set[int] = set()
        for ti, di in matches_a:
            self._apply_hit(confirmed_ids[ti], high_dets, di, lifecycle, src_index=int(high_src[di]))
            matched_high_dets.add(di)

        # Second association: unmatched ACTIVE tracks vs low-score detections (no fuse_score).
        remaining_confirmed_ids = [
            confirmed_ids[i] for i in unmatched_track_a if self._tracks[confirmed_ids[i]].state == TrackState.ACTIVE
        ]
        matches_b, unmatched_track_b, _ = self._associate(
            remaining_confirmed_ids,
            low_dets,
            cost_limit=_SECOND_STAGE_COST_LIMIT,
            apply_fuse_score=False,
        )
        for ti, di in matches_b:
            self._apply_hit(remaining_confirmed_ids[ti], low_dets, di, lifecycle, src_index=int(low_src[di]))

        missed_after_b = {remaining_confirmed_ids[i] for i in unmatched_track_b}
        confirmed_ids_set_b = set(remaining_confirmed_ids)
        for i in unmatched_track_a:
            tid = confirmed_ids[i]
            if tid not in confirmed_ids_set_b or tid in missed_after_b:
                self._tracks[tid].mark_miss(lifecycle)

        # TENTATIVE tracks compete for leftover high-score detections.
        leftover_high_idx = [i for i in unmatched_det_a if i not in matched_high_dets]
        matches_c, unmatched_track_c, unmatched_det_c = self._associate(
            tentative_ids,
            _subset(high_dets, leftover_high_idx),
            _TENTATIVE_COST_LIMIT,
            apply_fuse_score=True,
        )
        for ti, di in matches_c:
            real_di = leftover_high_idx[di]
            self._apply_hit(tentative_ids[ti], high_dets, real_di, lifecycle, src_index=int(high_src[real_di]))
        for i in unmatched_track_c:
            tid = tentative_ids[i]
            self._tracks[tid].mark_miss(lifecycle)

        # A new track needs a score clear of the high/low split by the margin.
        new_track_floor = cfg.high_score_threshold + _NEW_TRACK_MARGIN
        for di in unmatched_det_c:
            real_di = leftover_high_idx[di]
            if float(high_dets.scores[real_di]) >= new_track_floor:
                self._spawn(high_dets, real_di, src_index=int(high_src[real_di]))

        for tid in list(self._tracks):
            if self._tracks[tid].should_remove:
                del self._tracks[tid]
                self._kalman_states.pop(tid, None)

        self._remove_duplicate_tracks()
        return self._compose_output(detections.frame_id)

    def _predict_all(self) -> None:
        if not self._kalman_states:
            return
        tids = list(self._kalman_states)
        means = np.stack([self._kalman_states[tid][0] for tid in tids], axis=0)
        covs = np.stack([self._kalman_states[tid][1] for tid in tids], axis=0)
        # Lost tracks: zero out the height velocity so missing observations do not blow up the box.
        for i, tid in enumerate(tids):
            if self._tracks[tid].state != TrackState.ACTIVE:
                means[i, 7] = 0.0
        means, covs = self._kalman.multi_predict(means, covs)
        for i, tid in enumerate(tids):
            self._kalman_states[tid] = (means[i], covs[i])
            self._tracks[tid].bbox = xyah_to_xyxy(means[i, :4][None, :])[0].astype(np.float32)

    def _associate(
        self,
        track_ids: list[int],
        dets: Detections,
        cost_limit: float,
        *,
        apply_fuse_score: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not track_ids or len(dets) == 0:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.arange(len(track_ids), dtype=np.int64),
                np.arange(len(dets), dtype=np.int64),
            )
        track_boxes = np.stack([self._tracks[tid].bbox for tid in track_ids], axis=0)
        cost = iou_distance(track_boxes, dets.bboxes)
        cls_mismatch: np.ndarray | None = None
        if self.config.match_class_only:
            track_classes = np.array([self._tracks[tid].class_id for tid in track_ids])
            cls_mismatch = track_classes[:, None] != dets.class_ids[None, :]
        if apply_fuse_score:
            cost = fuse_score(cost, dets.scores)
        if cls_mismatch is not None:
            cost[cls_mismatch] = _UNMATCHABLE_COST
        return linear_assignment(cost, cost_limit)

    def _apply_hit(
        self,
        track_id: int,
        dets: Detections,
        det_idx: int,
        lifecycle: LifecycleConfig,
        *,
        src_index: int,
    ) -> None:
        track = self._tracks[track_id]
        bbox = dets.bboxes[det_idx]
        score = float(dets.scores[det_idx])
        track.mark_hit(bbox, score, lifecycle)
        self._frame_det_index[track_id] = src_index
        measurement = xyxy_to_xyah(bbox[None, :])[0]
        mean, covariance = self._kalman_states[track_id]
        self._kalman_states[track_id] = self._kalman.update(mean, covariance, measurement)

    def _spawn(self, dets: Detections, det_idx: int, *, src_index: int) -> None:
        track_id = self._allocate_id()
        bbox = dets.bboxes[det_idx].astype(np.float32)
        class_id = int(dets.class_ids[det_idx])
        score = float(dets.scores[det_idx])
        # Tracks activate immediately when min_hits <= 1 or on the first frame.
        skip_tentative = self.config.lifecycle.min_hits <= 1 or dets.frame_id == self._first_frame_id
        initial_state = TrackState.ACTIVE if skip_tentative else TrackState.TENTATIVE
        track = Track(
            track_id=track_id,
            class_id=class_id,
            bbox=bbox,
            score=score,
            state=initial_state,
            _start_frame=dets.frame_id,
        )
        self._tracks[track_id] = track
        self._frame_det_index[track_id] = src_index
        measurement = xyxy_to_xyah(bbox[None, :])[0]
        self._kalman_states[track_id] = self._kalman.initiate(measurement)

    def _remove_duplicate_tracks(self) -> None:
        """Drop the younger of any ACTIVE/LOST track pair with near-identical boxes.

        Mirrors the reference ``remove_duplicate_stracks``, keeping the
        longer-lived track. Cross-class pairs are skipped when
        ``match_class_only`` is set, since they are distinct objects.
        """
        active = [tid for tid, t in self._tracks.items() if t.state == TrackState.ACTIVE]
        lost = [tid for tid, t in self._tracks.items() if t.state == TrackState.LOST]
        if not active or not lost:
            return
        dist = iou_distance(
            np.stack([self._tracks[tid].bbox for tid in active], axis=0),
            np.stack([self._tracks[tid].bbox for tid in lost], axis=0),
        )
        # Collect duplicates first, then remove, so a track shared across
        # several pairs is not popped mid-iteration.
        drop_ids: set[int] = set()
        for ai, li in np.argwhere(dist < _DUPLICATE_IOU_DIST):
            a_track, l_track = self._tracks[active[ai]], self._tracks[lost[li]]
            if self.config.match_class_only and a_track.class_id != l_track.class_id:
                continue
            # Reference tie-break: keep the strictly longer-lived track, so an
            # equal-age pair drops the active-side track.
            drop_ids.add(lost[li] if a_track.age > l_track.age else active[ai])
        for tid in drop_ids:
            self._tracks.pop(tid, None)
            self._kalman_states.pop(tid, None)

    def _compose_output(self, frame_id: int) -> TrackedDetections:
        # Output is ACTIVE tracks only; LOST tracks coast internally.
        active = [t for t in self._tracks.values() if t.state == TrackState.ACTIVE]
        if not active:
            empty = TrackedDetections.create_empty(frame_id=frame_id)
            return replace(empty, det_indices=np.empty((0,), dtype=np.int64))
        return TrackedDetections(
            bboxes=np.stack([t.bbox for t in active], axis=0).astype(np.float32),
            scores=np.array([t.score for t in active], dtype=np.float32),
            class_ids=np.array([t.class_id for t in active], dtype=np.int64),
            track_ids=np.array([t.track_id for t in active], dtype=np.int64),
            track_states=np.array([int(t.state) for t in active], dtype=np.int8),
            frame_id=frame_id,
            det_indices=np.array([self._frame_det_index.get(t.track_id, -1) for t in active], dtype=np.int64),
        )


def _subset(dets: Detections, indices: list[int] | np.ndarray) -> Detections:
    idx = np.asarray(indices, dtype=np.int64)
    return Detections(
        bboxes=dets.bboxes[idx] if idx.size else np.empty((0, 4), dtype=np.float32),
        scores=dets.scores[idx] if idx.size else np.empty((0,), dtype=np.float32),
        class_ids=dets.class_ids[idx] if idx.size else np.empty((0,), dtype=np.int64),
        frame_id=dets.frame_id,
        embeddings=None if dets.embeddings is None or not idx.size else dets.embeddings[idx],
    )
