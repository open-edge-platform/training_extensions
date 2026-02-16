# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from app.db.schema import VideoFrameDB
from app.models import Media, VideoFrame
from app.repositories.video_frame_repo import VideoFrameRepository
from app.services import BaseSessionManagedService


class VideoFrameService(BaseSessionManagedService):
    def create_video_frame(
        self,
        video_frame_media: Media,
        video: Media,
        timestamp: float,
    ) -> VideoFrame:
        """Creates a new video frame"""

        db_video_frame = VideoFrameDB(
            id=str(video_frame_media.id),
            video_id=str(video.id),
            timestamp=timestamp,
        )

        repo = VideoFrameRepository(db=self.db_session)
        db_video_frame = repo.save(db_video_frame)
        return VideoFrame.model_validate(db_video_frame)

    def get_frame_by_video_id_and_timestamp(self, video_id: UUID, timestamp: float) -> VideoFrame | None:
        repo = VideoFrameRepository(db=self.db_session)
        db_video_frame = repo.get_by_video_id_and_timestamp(video_id=str(video_id), timestamp=timestamp)
        return VideoFrame.model_validate(db_video_frame) if db_video_frame else None

    def get_frames_by_video_id(self, video_id: UUID) -> list[VideoFrame]:
        repo = VideoFrameRepository(db=self.db_session)
        db_video_frames = repo.get_by_video_id(video_id=str(video_id))
        return [VideoFrame.model_validate(frame) for frame in db_video_frames]
