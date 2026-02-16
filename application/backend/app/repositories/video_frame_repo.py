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

    def save(self, video_frame_db: VideoFrameDB) -> VideoFrameDB:
        video_frame_db.updated_at = datetime.now(UTC)
        self.db.add(video_frame_db)
        self.db.flush()
        return video_frame_db

    def get_by_video_id_and_timestamp(self, video_id: str, timestamp: float) -> VideoFrameDB | None:
        stmt = select(VideoFrameDB).where(VideoFrameDB.video_id == video_id, VideoFrameDB.timestamp == timestamp)
        return self.db.scalar(stmt)

    def get_by_video_id(self, video_id: str) -> list[VideoFrameDB]:
        stmt = select(VideoFrameDB).where(VideoFrameDB.video_id == video_id)
        return list(self.db.scalars(stmt).all())
