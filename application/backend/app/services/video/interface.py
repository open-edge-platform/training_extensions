# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    width: int = Field(..., description="Video width", ge=0)
    height: int = Field(..., description="Video height", ge=0)
    frame_count: int = Field(..., description="Video frames number", ge=0)
    fps: float = Field(..., description="Video frames per second", ge=0)


class IVideoService(ABC):
    """Interface for video frame extraction and metadata retrieval."""

    @abstractmethod
    def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract video metadata."""

    @abstractmethod
    def extract_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        """Extract a single video frame (RGB format)."""

    @abstractmethod
    def extract_frames(self, video_path: Path, frame_indexes: list[int]) -> dict[int, np.ndarray]:
        """Extract multiple video frames (RGB format)."""

    def close(self) -> None:
        """Release resources. Default is no-op."""
