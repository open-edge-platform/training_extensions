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

    def get_by_video_id_and_index(self, video_id: str, frame_index: int) -> VideoFrameDB | None:
        stmt = select(VideoFrameDB).where(VideoFrameDB.video_id == video_id, VideoFrameDB.frame_index == frame_index)
        return self.db.scalar(stmt)

    def get_by_video_id(self, video_id: str, frame_index_from: int = 0, frame_index_to: int = 10) -> list[VideoFrameDB]:
        stmt = select(VideoFrameDB).where(
            VideoFrameDB.video_id == video_id,
            VideoFrameDB.frame_index >= frame_index_from,
            VideoFrameDB.frame_index < frame_index_to,
        )
        return list(self.db.scalars(stmt).all())
