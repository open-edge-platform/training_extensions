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
        frame_index: int,
    ) -> VideoFrame:
        """Creates a new video frame"""

        db_video_frame = VideoFrameDB(
            id=str(video_frame_media.id),
            video_id=str(video.id),
            frame_index=frame_index,
        )

        repo = VideoFrameRepository(db=self.db_session)
        db_video_frame = repo.save(db_video_frame)
        return VideoFrame.model_validate(db_video_frame)

    def get_frame_by_video_id_and_index(self, video_id: UUID, frame_index: int) -> VideoFrame | None:
        """
        Returns annotated video frame by video ID and frame index.

        Args:
            video_id: Video identifier
            frame_index: Frame index

        Returns:
            Video frame data if such frame has been annotated, None otherwise.
        """
        repo = VideoFrameRepository(db=self.db_session)
        db_video_frame = repo.get_by_video_id_and_index(video_id=str(video_id), frame_index=frame_index)
        return VideoFrame.model_validate(db_video_frame) if db_video_frame else None

    def get_frames_by_video_id(
        self, video_id: UUID, frame_index_from: int = 0, frame_index_to: int = 10
    ) -> list[VideoFrame]:
        """
        Returns all annotated video frame falling into the specified index range.

        Args:
            video_id: Video identifier
            frame_index_from: Frame index range start, default is 0
            frame_index_to: Frame index range end, default is 10

        Returns:
            Annotated video frames list.
        """
        repo = VideoFrameRepository(db=self.db_session)
        db_video_frames = repo.get_by_video_id(
            video_id=str(video_id), frame_index_from=frame_index_from, frame_index_to=frame_index_to
        )
        return [VideoFrame.model_validate(frame) for frame in db_video_frames]
