# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.schema import VideoFrameDB


class VideoFrameRepository:
    """Repository for video frame-related database operations."""

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _normalize_timestamp(timestamp: float) -> float:
        """Normalize timestamps to a fixed precision to avoid float equality issues."""
        # Using millisecond precision
        return round(timestamp, 3)

    def save(self, video_frame_db: VideoFrameDB) -> VideoFrameDB:
        video_frame_db.timestamp = self._normalize_timestamp(video_frame_db.timestamp)
        video_frame_db.updated_at = datetime.now(UTC)
        self.db.add(video_frame_db)
        self.db.flush()
        return video_frame_db

    def get_by_video_id_and_timestamp(self, video_id: str, timestamp: float) -> VideoFrameDB | None:
        normalized_timestamp = self._normalize_timestamp(timestamp)
        stmt = select(VideoFrameDB).where(
            VideoFrameDB.video_id == video_id, VideoFrameDB.timestamp == normalized_timestamp
        )
        return self.db.scalar(stmt)

    def get_by_video_id(self, video_id: str, timestamp_from: float = 0, timestamp_to: int = 10) -> list[VideoFrameDB]:
        stmt = select(VideoFrameDB).where(
            VideoFrameDB.video_id == video_id,
            VideoFrameDB.timestamp >= self._normalize_timestamp(timestamp_from),
            VideoFrameDB.timestamp < self._normalize_timestamp(timestamp_to),
        )
        return list(self.db.scalars(stmt).all())
