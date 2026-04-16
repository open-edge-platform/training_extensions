# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .interface import IVideoService, VideoMetadata


class VideoService(IVideoService):
    """Direct video frame extraction using OpenCV without caching."""

    def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """
        Extract video metadata (dimensions, frame count, FPS) from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMetadata with width, height, frame_count and fps.

        Raises:
            RuntimeError: If the video cannot be opened or metadata extraction fails.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
        except Exception as e:
            logger.error(f"Failed getting metadata for video {video_path}", exc_info=e)
            raise RuntimeError("Error occurred while getting video metadata")
        finally:
            cap.release()
        return VideoMetadata(width=width, height=height, frame_count=frame_count, fps=fps)

    def extract_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        """
        Extract a single frame from a video file.

        Args:
            video_path: Path to the video file.
            frame_index: Zero-based index of the frame to extract.

        Returns:
            Extracted frame as a numpy array in RGB format.

        Raises:
            RuntimeError: If the video cannot be opened or the frame cannot be read.
        """
        frames = self.extract_frames(video_path=video_path, frame_indexes=[frame_index])
        return frames[frame_index]

    def extract_frames(self, video_path: Path, frame_indexes: list[int]) -> dict[int, np.ndarray]:
        """
        Extract multiple frames from a video file in a single pass.

        Frames are read in ascending index order to minimise seeking overhead.

        Args:
            video_path: Path to the video file.
            frame_indexes: List of zero-based frame indexes to extract.

        Returns:
            Dictionary mapping each requested frame index to its numpy array in RGB format.

        Raises:
            RuntimeError: If the video cannot be opened or any frame cannot be read.
        """
        if not frame_indexes:
            return {}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        try:
            result: dict[int, np.ndarray] = {}
            for frame_index in sorted(set(frame_indexes)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                read_success, frame = cap.read()
                if not read_success:
                    raise RuntimeError(f"Cannot read frame at {frame_index} index from video: {video_path}")
                result[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return result
        finally:
            cap.release()
