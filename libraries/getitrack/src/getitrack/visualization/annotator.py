# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Bounding-box and track-id overlay drawing.

`TrackAnnotator` draws one frame's `TrackedDetections` onto a BGR image.
Colors are deterministic per track id (golden-ratio hue stepping), so the
same track keeps the same color across frames and across runs.
`VideoAnnotator` composes a `TrackAnnotator` with a `VideoWriter` for the
common case of writing the annotated frames straight to a video file.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import cv2
import numpy as np

from getitrack.io import VideoWriter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from types import TracebackType

    from getitrack.core.detection import TrackedDetections

_GOLDEN_RATIO_CONJUGATE = 0.61803398875
_HUE_MAX = 179
_LABEL_PADDING = 3
_LUMINANCE_THRESHOLD = 140.0


def color_for_track(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a track id.

    Hues are stepped by the golden-ratio conjugate so consecutive ids get
    visually distant colors.

    Args:
        track_id: Stable track identifier.

    Returns:
        ``(B, G, R)`` tuple with components in ``[0, 255]``.
    """
    hue = (track_id * _GOLDEN_RATIO_CONJUGATE) % 1.0
    hsv = np.array([[[int(hue * _HUE_MAX), 200, 255]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


class TrackAnnotator:
    """Draws tracked bounding boxes, ids, class names, and scores onto frames.

    Labels render as ``<class> #<id> <score>``; the class part appears only
    when ``class_names`` is provided, the score only when ``show_score`` is
    true.

    Attributes:
        thickness: Box outline thickness in pixels.
        font_scale: OpenCV font scale for labels.
        show_score: Append the detection score to each label when true.
        class_names: Optional class-id-to-name lookup, either a sequence
            indexed by class id or a (possibly sparse) mapping of class id
            to name. Unknown class ids fall back to the numeric id.
    """

    def __init__(
        self,
        thickness: int = 2,
        font_scale: float = 0.4,
        show_score: bool = True,
        class_names: Sequence[str] | Mapping[int, str] | None = None,
    ) -> None:
        self.thickness = thickness
        self.font_scale = font_scale
        self.show_score = show_score
        self.class_names = class_names

    def annotate(self, frame: np.ndarray, tracked: TrackedDetections) -> np.ndarray:
        """Return a copy of ``frame`` with all tracks drawn on it.

        Args:
            frame: ``(H, W, 3)`` BGR uint8 image. Not modified.
            tracked: One frame's tracker output.

        Returns:
            Annotated copy of the input frame.
        """
        out = frame.copy()
        for bbox, track_id, score, class_id in zip(
            tracked.bboxes,
            tracked.track_ids,
            tracked.scores,
            tracked.class_ids,
            strict=True,
        ):
            color = color_for_track(int(track_id))
            x1, y1, x2, y2 = (round(float(v)) for v in bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.thickness)
            label = self._label(int(track_id), float(score), int(class_id))
            self._draw_label(out, label, x1, y1, color)
        return out

    def _label(self, track_id: int, score: float, class_id: int) -> str:
        parts = []
        if isinstance(self.class_names, Mapping):
            parts.append(self.class_names.get(class_id, str(class_id)))
        elif self.class_names is not None:
            parts.append(self.class_names[class_id] if 0 <= class_id < len(self.class_names) else str(class_id))
        parts.append(f"#{track_id}")
        if self.show_score:
            parts.append(f"{score:.2f}")
        return " ".join(parts)

    def _draw_label(self, image: np.ndarray, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
        top = max(y - text_h - baseline - _LABEL_PADDING, 0)
        bottom_right = (x + text_w + _LABEL_PADDING, top + text_h + baseline + _LABEL_PADDING)
        cv2.rectangle(image, (x, top), bottom_right, color, -1)
        # Black or white text, whichever contrasts with the box color.
        luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        text_color = (0, 0, 0) if luminance > _LUMINANCE_THRESHOLD else (255, 255, 255)
        cv2.putText(
            image,
            label,
            (x + _LABEL_PADDING, top + text_h + _LABEL_PADDING // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )


class VideoAnnotator:
    """Writes annotated tracking output straight to a video file.

    Composes a `TrackAnnotator` with a `VideoWriter` for the common case.
    Use the two pieces directly when the annotated frames are needed for
    anything other than a video file (live preview, image dumps).

    Example:
        >>> with VideoAnnotator("out.mp4", fps=30.0, frame_size=(640, 480), class_names=names) as out:
        ...     out.write(frame, tracked)
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
        thickness: int = 2,
        font_scale: float = 0.4,
        show_score: bool = True,
        class_names: Sequence[str] | Mapping[int, str] | None = None,
    ) -> None:
        """Open a video file for annotated writing.

        Args:
            path: Destination video path.
            fps: Output frame rate.
            frame_size: ``(width, height)`` of every frame.
            codec: FourCC codec identifier.
            thickness: Box outline thickness in pixels.
            font_scale: OpenCV font scale for labels.
            show_score: Append the detection score to each label when true.
            class_names: Optional class-id-to-name lookup for labels.
        """
        self.annotator = TrackAnnotator(
            thickness=thickness,
            font_scale=font_scale,
            show_score=show_score,
            class_names=class_names,
        )
        self._writer = VideoWriter(path, fps=fps, frame_size=frame_size, codec=codec)

    @property
    def path(self) -> Path:
        """Destination video path."""
        return self._writer.path

    @property
    def frames_written(self) -> int:
        """Number of frames written so far."""
        return self._writer.frames_written

    def write(self, frame: np.ndarray, tracked: TrackedDetections) -> None:
        """Annotate one frame and append it to the video."""
        self._writer.write(self.annotator.annotate(frame, tracked))

    def close(self) -> None:
        """Finalise the container and release the writer handle."""
        self._writer.close()

    def __enter__(self) -> VideoAnnotator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
