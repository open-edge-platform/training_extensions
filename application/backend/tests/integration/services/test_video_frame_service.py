# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from datetime import datetime
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from app.db.schema import MediaDB, ProjectDB, VideoFrameDB
from app.models import Media
from app.services import VideoFrameService
from tests.integration.project_factory import ProjectTestDataFactory


@pytest.fixture
def fxt_video_frame_service(db_session: Session) -> VideoFrameService:
    """Fixture to create a VideoFrameService instance."""
    return VideoFrameService(db_session=db_session)


@pytest.fixture(autouse=True)
def setup_project(fxt_db_projects: list[ProjectDB], db_session: Session) -> None:
    """Fixture to set up a project."""

    (ProjectTestDataFactory(db_session).with_project(fxt_db_projects[0]).build())


@pytest.fixture
def fxt_video_media(fxt_project_id: UUID, db_session) -> MediaDB:
    db_media = MediaDB(
        type="video",
        name="test4",
        format="avi",
        size=1024,
        width=1024,
        height=768,
        fps=25.0,
        frame_count=100,
    )
    db_media.project_id = str(fxt_project_id)
    db_media.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")

    db_session.add(db_media)
    db_session.flush()

    return db_media


@pytest.fixture
def fxt_video_frame(
    fxt_project_id: UUID, fxt_video_media: MediaDB, db_session
) -> Callable[[float], tuple[MediaDB, VideoFrameDB]]:
    def _create_video_frame(timestamp: float) -> tuple[MediaDB, VideoFrameDB]:
        db_media = MediaDB(
            type="video_frame", name=f"test4_frame_{timestamp:.3f}", format="jpg", size=1024, width=1024, height=768
        )
        db_media.project_id = str(fxt_project_id)
        db_media.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")

        db_session.add(db_media)
        db_session.flush()

        db_video_frame = VideoFrameDB(id=db_media.id, video_id=fxt_video_media.id, timestamp=timestamp)
        db_session.add(db_video_frame)
        db_session.flush()

        return db_media, db_video_frame

    return _create_video_frame


class TestVideoFrameServiceIntegration:
    """Integration tests for VideoFrameService."""

    def test_create_video_frame(
        self,
        fxt_video_frame_service: VideoFrameService,
        fxt_project_id: UUID,
        fxt_video_media: MediaDB,
        db_session: Session,
    ) -> None:
        """Test creating a video frame."""
        video_media = Media.model_validate(fxt_video_media, from_attributes=True)

        db_video_frame_media = MediaDB(
            type="video_frame", name="test4_10", format="jpg", size=1024, width=1024, height=768
        )
        db_video_frame_media.project_id = str(fxt_project_id)
        db_video_frame_media.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")

        db_session.add(db_video_frame_media)
        db_session.flush()

        video_frame_media = Media.model_validate(db_video_frame_media, from_attributes=True)

        created_video_frame = fxt_video_frame_service.create_video_frame(
            video_frame_media=video_frame_media,
            video=video_media,
            timestamp=10.0,
        )

        video_frame = db_session.get(VideoFrameDB, str(created_video_frame.id))
        assert video_frame is not None
        assert (
            video_frame.id == str(created_video_frame.id)
            and video_frame.video_id == str(created_video_frame.video_id)
            and video_frame.timestamp == 10.0
        )

    def test_get_frame_by_video_id_and_timestamp(
        self,
        fxt_video_frame_service: VideoFrameService,
        fxt_video_media: MediaDB,
        fxt_video_frame: Callable[[float], tuple[MediaDB, VideoFrameDB]],
    ) -> None:
        """Test getting a video frame by video ID and timestamp."""
        fxt_video_frame(10.0)

        video_frame = fxt_video_frame_service.get_frame_by_video_id_and_timestamp(
            video_id=UUID(fxt_video_media.id), timestamp=10.0
        )
        assert video_frame is not None

    def test_get_non_existing_frame_by_video_id_and_timestamp(
        self,
        fxt_video_frame_service: VideoFrameService,
        fxt_video_media: MediaDB,
        fxt_video_frame: Callable[[float], tuple[MediaDB, VideoFrameDB]],
    ) -> None:
        """Test getting a non extracted video frame by video ID and timestamp."""
        fxt_video_frame(10.0)

        video_frame = fxt_video_frame_service.get_frame_by_video_id_and_timestamp(
            video_id=UUID(fxt_video_media.id), timestamp=20.0
        )
        assert video_frame is None

    def test_get_frames_by_video_id(
        self,
        fxt_video_frame_service: VideoFrameService,
        fxt_video_media: MediaDB,
        fxt_video_frame: Callable[[float], tuple[MediaDB, VideoFrameDB]],
    ) -> None:
        """Test getting a list of video frames by video ID."""
        fxt_video_frame(10.0)
        fxt_video_frame(20.0)

        video_frames = fxt_video_frame_service.get_frames_by_video_id(video_id=UUID(fxt_video_media.id))
        assert video_frames is not None
        assert len(video_frames) == 2
