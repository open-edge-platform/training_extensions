# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import os.path
import tempfile
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

import numpy as np
from loguru import logger
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.db.schema import MediaDB
from app.models import DatasetItemAnnotationStatus, Media, MediaType, Project, VideoFrame
from app.models.media import ImageFormat, MediaAdapter, VideoFormat
from app.repositories import MediaRepository
from app.services.video import extract_video_frame, get_video_metadata
from app.utils.images import crop_to_thumbnail

from .base import BaseSessionManagedService, ResourceNotFoundError, ResourceType

DEFAULT_THUMBNAIL_SIZE = 256

VIDEO_WRITE_CHUNK_SIZE = 100 * 1024 * 1024  # 100 MB


class InvalidImageError(Exception):
    """Exception raised when invalid image is used to create a media."""

    def __init__(self, message: str | None = None):
        msg = message or "Invalid image has been passed while creating a media."
        super().__init__(msg)


@dataclass(frozen=True)
class MediaFilters:
    limit: int = 20
    offset: int = 0
    start_date: datetime | None = None
    end_date: datetime | None = None
    annotation_status: DatasetItemAnnotationStatus | None = None
    label_ids: list[UUID] | None = None
    subset: str | None = None


class MediaService(BaseSessionManagedService):
    def __init__(self, data_dir: Path, db_session: Session | None = None) -> None:
        super().__init__(db_session)
        self.projects_dir = data_dir / "projects"

    @staticmethod
    def _read_image_from_ndarray(data: np.ndarray) -> Image.Image:
        return Image.fromarray(data)

    @staticmethod
    def _read_image_from_binary(data: BinaryIO | BytesIO) -> Image.Image:
        data.seek(0)
        try:
            return Image.open(data)
        except UnidentifiedImageError:
            raise InvalidImageError

    @staticmethod
    def _generate_and_save_thumbnail(image: Image.Image, path: Path) -> None:
        try:
            thumbnail_image = crop_to_thumbnail(
                image=image, target_width=DEFAULT_THUMBNAIL_SIZE, target_height=DEFAULT_THUMBNAIL_SIZE
            )
            if thumbnail_image.mode in ("RGBA", "P"):
                thumbnail_image = thumbnail_image.convert("RGB")
            thumbnail_image.save(path)
        except Exception:
            logger.exception("Failed to generate thumbnail image")

    def create_image(
        self,
        project: Project,
        name: str,
        format: ImageFormat,
        data: Image.Image | np.ndarray | BinaryIO | BytesIO,
        source_id: UUID | None = None,
    ) -> Media:
        """Creates a new media (image)"""
        media_id = uuid4()
        match data:
            case Image.Image():
                image = data
            case np.ndarray():
                image = self._read_image_from_ndarray(data)
            case _:
                image = self._read_image_from_binary(data)

        dataset_dir = self.projects_dir / f"{project.id}/dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        binary_path = dataset_dir / f"{media_id}.{format}"
        image.save(binary_path)

        MediaService._generate_and_save_thumbnail(image, dataset_dir / f"{media_id}-thumb.jpg")

        media = MediaDB(
            id=str(media_id),
            project_id=str(project.id),
            type=MediaType.IMAGE,
            name=name,
            format=str(format),
            width=image.width,
            height=image.height,
            size=os.path.getsize(binary_path),
            source_id=str(source_id) if source_id is not None else None,
        )

        repo = MediaRepository(project_id=str(project.id), db=self.db_session)
        db_media = repo.save(media)
        return MediaAdapter.validate_python(db_media)

    def create_video(
        self,
        project: Project,
        name: str,
        format: VideoFormat,
        data: BinaryIO,
        source_id: UUID | None = None,
    ) -> Media:
        """Creates a new media (video)"""
        media_id = uuid4()

        dataset_dir = self.projects_dir / f"{project.id}/dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        binary_path = dataset_dir / f"{media_id}.{format}"

        data.seek(0)
        with open(binary_path, "wb") as f:
            # Read in chunks to avoid memory issues for large files
            while chunk := data.read(VIDEO_WRITE_CHUNK_SIZE):
                f.write(chunk)

        try:
            video_metadata = get_video_metadata(video_path=binary_path)
            media = MediaDB(
                id=str(media_id),
                project_id=str(project.id),
                type=MediaType.VIDEO,
                name=name,
                format=str(format),
                width=video_metadata.width,
                height=video_metadata.height,
                frame_count=video_metadata.frame_count,
                fps=video_metadata.fps,
                size=os.path.getsize(binary_path),
                source_id=str(source_id) if source_id is not None else None,
            )

            repo = MediaRepository(project_id=str(project.id), db=self.db_session)
            db_media = repo.save(media)
        except Exception as e:
            binary_path.unlink(missing_ok=True)
            raise e
        return MediaAdapter.validate_python(db_media)

    def count_media(
        self,
        project: Project,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[UUID] | None = None,
        subset: str | None = None,
        exclude_types: list[MediaType] | None = None,
    ) -> int:
        """Get number of available media (within date range if specified)"""
        repo = MediaRepository(project_id=str(project.id), db=self.db_session)
        label_ids_str = [str(label_id) for label_id in label_ids] if label_ids else None
        return repo.count(
            start_date=start_date,
            end_date=end_date,
            annotation_status=annotation_status,
            label_ids=label_ids_str,
            subset=subset,
            exclude_types=exclude_types,
        )

    def list_media(
        self,
        project_id: UUID,
        filters: MediaFilters | None = None,
        exclude_types: list[MediaType] | None = None,
    ) -> list[Media]:
        """Get information about available media"""
        if filters is None:
            filters = MediaFilters()
        repo = MediaRepository(project_id=str(project_id), db=self.db_session)
        label_ids_str = [str(label_id) for label_id in filters.label_ids] if filters.label_ids else None
        return [
            MediaAdapter.validate_python(db)
            for db in repo.list_items(
                limit=filters.limit,
                offset=filters.offset,
                start_date=filters.start_date,
                end_date=filters.end_date,
                annotation_status=filters.annotation_status,
                label_ids=label_ids_str,
                subset=filters.subset,
                exclude_types=exclude_types,
            )
        ]

    def get_media_by_id(self, project_id: UUID, media_id: UUID) -> Media:
        """Get a media by its ID"""
        repo = MediaRepository(project_id=str(project_id), db=self.db_session)
        db_media = repo.get_by_id(str(media_id))
        if not db_media:
            raise ResourceNotFoundError(ResourceType.MEDIA, str(media_id))
        return MediaAdapter.validate_python(db_media)

    def get_media_binary_path(self, project_id: UUID, media: MediaDB | Media) -> Path:
        dataset_dir = self.projects_dir / f"{project_id}/dataset"
        return dataset_dir / f"{media.id}.{media.format}"

    def get_media_binary_path_by_id(self, project_id: UUID, media_id: UUID) -> Path | str:
        """Get a media binary content by its ID"""
        media = self.get_media_by_id(project_id=project_id, media_id=media_id)
        return self.get_media_binary_path(project_id=project_id, media=media)

    def get_media_thumbnail_path_by_id(self, project: Project, media_id: UUID) -> Path | str:
        """Get a media thumbnail binary content by its ID"""
        media = self.get_media_by_id(project_id=project.id, media_id=media_id)
        return self.projects_dir / f"{project.id}/dataset/{media.id}-thumb.jpg"

    def generate_media_thumbnail(self, project: Project, media_id: UUID) -> Image.Image:
        """Regenerate a media thumbnail by its ID"""
        media = self.get_media_by_id(project_id=project.id, media_id=media_id)
        dataset_dir = self.projects_dir / f"{project.id}/dataset"
        binary_path = dataset_dir / f"{media.id}.{media.format}"

        if media.type == MediaType.IMAGE:
            return MediaService.generate_image_thumbnail(binary_path)
        if media.type == MediaType.VIDEO:
            return MediaService.generate_video_thumbnail(
                binary_path=binary_path,
                frame_index=media.frame_count // 2,  # pyrefly: ignore
            )
        raise RuntimeError(f"Unknown media type: {media.type}")

    @staticmethod
    def generate_image_thumbnail(binary_path: Path) -> Image.Image:
        """Regenerate an image thumbnail by its path"""
        try:
            with Image.open(binary_path) as image:
                thumbnail = crop_to_thumbnail(
                    image=image, target_width=DEFAULT_THUMBNAIL_SIZE, target_height=DEFAULT_THUMBNAIL_SIZE
                )
                if thumbnail.mode not in ("RGB", "L", "CMYK"):
                    thumbnail = thumbnail.convert("RGB")
        except UnidentifiedImageError:
            logger.error("Failed to open image {} for thumbnail generation", binary_path)
            raise InvalidImageError("Failed to open image for thumbnail generation.")
        return thumbnail

    @staticmethod
    def generate_video_thumbnail(binary_path: Path, frame_index: int) -> Image.Image:
        """Regenerate a video thumbnail by its path"""
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

        frame_file_path = Path(temp_path)
        try:
            extract_video_frame(video_path=binary_path, video_frame_path=frame_file_path, frame_index=frame_index)
            return MediaService.generate_image_thumbnail(frame_file_path)
        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

    def delete_media(self, project: Project, media_id: UUID) -> None:
        """Delete a media by its ID"""
        media = self.get_media_by_id(project_id=project.id, media_id=media_id)
        repo = MediaRepository(project_id=str(project.id), db=self.db_session)

        dataset_dir = self.projects_dir / f"{project.id}/dataset"
        try:
            os.remove(dataset_dir / f"{media.id}.{media.format}")
        except FileNotFoundError:
            logger.warning("Media {} binary was not found during deletion", media_id)
        try:
            os.remove(dataset_dir / f"{media_id}-thumb.jpg")
        except FileNotFoundError:
            logger.warning("Media {} thumbnail was not found during deletion", media_id)

        repo.delete(obj_id=str(media.id))

    def extract_video_frame(self, project: Project, video_id: UUID, frame_index: int) -> Media:
        """Extract video frame by video ID and index"""
        video = self.get_media_by_id(project_id=project.id, media_id=video_id)
        if video.type != MediaType.VIDEO:
            raise RuntimeError(f"Media {video_id} is not a video")

        media_id = uuid4()
        format = ImageFormat.JPG

        dataset_dir = self.projects_dir / f"{project.id}/dataset"
        video_path = self.get_media_binary_path(project_id=project.id, media=video)
        video_frame_path = dataset_dir / f"{media_id}.{format}"

        video_frame_numpy = extract_video_frame(
            video_path=video_path, video_frame_path=video_frame_path, frame_index=frame_index
        )
        image = self._read_image_from_ndarray(video_frame_numpy)

        db_video_frame = MediaDB(
            id=str(media_id),
            project_id=str(project.id),
            type=MediaType.VIDEO_FRAME,
            name=f"{video.name}_frame_{frame_index}",
            format=str(format),
            width=image.width,
            height=image.height,
            video_id=str(video.id),
            frame_index=frame_index,
            size=os.path.getsize(video_frame_path),
            source_id=None,
        )

        repo = MediaRepository(project_id=str(project.id), db=self.db_session)
        db_video_frame = repo.save(db_video_frame)
        return MediaAdapter.validate_python(db_video_frame)

    def get_video_frame_by_video_id_and_index(
        self, project: Project, video_id: UUID, frame_index: int
    ) -> VideoFrame | None:
        """
        Returns annotated video frame by video ID and frame index.

        Args:
            project: Project
            video_id: Video identifier
            frame_index: Frame index

        Returns:
            Video frame data if such frame has been annotated, None otherwise.
        """
        repo = MediaRepository(project_id=str(project.id), db=self.db_session)
        db_media = repo.get_video_frame_by_video_id_and_index(video_id=str(video_id), frame_index=frame_index)
        return VideoFrame.model_validate(db_media, from_attributes=True) if db_media else None

    def get_video_frames_by_video_id(
        self, project: Project, video_id: UUID, frame_index_from: int = 0, frame_index_to: int = 10
    ) -> list[VideoFrame]:
        """
        Returns all annotated video frames falling into the specified frame index range.

        Args:
            project: Project
            video_id: Video identifier
            frame_index_from: Frame index range start, default is 0
            frame_index_to: Frame index range end, default is 10

        Returns:
            Annotated video frames list.
        """
        repo = MediaRepository(project_id=str(project.id), db=self.db_session)
        db_media_list = repo.get_video_frames_by_video_id(
            video_id=str(video_id), frame_index_from=frame_index_from, frame_index_to=frame_index_to
        )
        return [VideoFrame.model_validate(db_media, from_attributes=True) for db_media in db_media_list]
