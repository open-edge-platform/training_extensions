# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
from uuid import UUID

import cv2
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import ProjectDB, SourceDB
from app.models import Source, SourceType
from app.models.source import (
    ImagesFolderSourceConfig,
    IPCameraSourceConfig,
    SourceAdapter,
    SourceConfig,
    USBCameraSourceConfig,
    VideoFileSourceConfig,
)
from app.repositories import SourceRepository
from app.repositories.base import PrimaryKeyIntegrityError, UniqueConstraintIntegrityError
from app.repositories.pipeline_repo import PipelineRepository

from .base import (
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    ResourceWithIdAlreadyExistsError,
    ResourceWithNameAlreadyExistsError,
)
from .event.event_bus import EventBus, EventType
from .parent_process_guard import parent_process_only


class SourceService:
    def __init__(self, db_session: Session):
        self._db_session = db_session

    @parent_process_only
    def create_source(
        self,
        name: str,
        source_type: SourceType,
        config_data: SourceConfig,
        source_id: UUID | None = None,
    ) -> Source:
        try:
            db_source = SourceRepository(self._db_session).save(
                SourceDB(
                    id=str(source_id) if source_id is not None else None,
                    name=name,
                    source_type=source_type,
                    config_data=config_data.model_dump(mode="json"),
                )
            )
            return SourceAdapter.validate_python(db_source, from_attributes=True)
        except PrimaryKeyIntegrityError:
            raise ResourceWithIdAlreadyExistsError(ResourceType.SOURCE, str(source_id))
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SOURCE, name)

    def get_by_id(self, source_id: UUID) -> Source:
        db_source = SourceRepository(self._db_session).get_by_id(str(source_id))
        if not db_source:
            raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
        return SourceAdapter.validate_python(db_source, from_attributes=True)

    def list_all(self) -> list[Source]:
        return [
            SourceAdapter.validate_python(db_source, from_attributes=True)
            for db_source in SourceRepository(self._db_session).list_all()
        ]

    @parent_process_only
    def delete_source(self, source: Source) -> None:
        # Check for pipelines using this source before attempting deletion
        pipelines = PipelineRepository(self._db_session).get_by_source_id(str(source.id))
        if pipelines:
            project_details = []
            for p in pipelines:
                project = self._db_session.get(ProjectDB, p.project_id)
                project_name = project.name if project else p.project_id
                state = "running" if p.is_running else "configured"
                project_details.append(f"'{project_name}' ({state})")
            projects_str = ", ".join(project_details)
            msg = (
                f"Source '{source.name}' cannot be deleted because it is used by "
                f"a pipeline in project: {projects_str}. "
                f"Please stop and remove the pipeline configuration in that project first."
            )
            raise ResourceInUseError(ResourceType.SOURCE, str(source.id), msg)
        try:
            deleted = SourceRepository(self._db_session).delete(str(source.id))
            if not deleted:
                raise ResourceNotFoundError(ResourceType.SOURCE, str(source.id))
        except IntegrityError:
            raise ResourceInUseError(ResourceType.SOURCE, str(source.id))

    def get_active_source(self) -> Source | None:
        db_source = SourceRepository(self._db_session).get_active_source()
        return SourceAdapter.validate_python(db_source, from_attributes=True) if db_source else None

    def get_active_source_id(self) -> UUID | None:
        id = SourceRepository(self._db_session).get_active_source_id()
        return UUID(id) if id else None


class SourceUpdateService(SourceService):
    def __init__(self, event_bus: EventBus, db_session: Session):
        self._event_bus: EventBus = event_bus
        super().__init__(db_session)

    @parent_process_only
    def update_source(
        self,
        source: Source,
        new_name: str,
        new_config_data: SourceConfig,
    ) -> Source:
        try:
            source_repo = SourceRepository(self._db_session)
            db_source = source_repo.update(
                SourceDB(
                    id=str(source.id),
                    name=new_name,
                    config_data=new_config_data.model_dump(mode="json"),
                )
            )
            active_source_id = self.get_active_source_id()
            if active_source_id == UUID(db_source.id):
                self._event_bus.emit_event(EventType.SOURCE_CHANGED)
            return SourceAdapter.validate_python(db_source, from_attributes=True)
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SOURCE, new_name)

    _TEST_TIMEOUT_MS = 5000

    def test_source(self, source: Source) -> dict:
        """Perform a connectivity check on the source.

        Verifies that the source can be opened based on its type:
        - USB camera: opens the device and verifies capture is functional
        - IP camera: opens the RTSP/HTTP stream and verifies frames can be read
        - Video file: verifies the file exists and can be opened as a video
        - Images folder: verifies the directory exists and is accessible
        """
        start = time.monotonic()

        try:
            match source:
                case USBCameraSourceConfig():
                    reachable, error = self._test_usb_camera(source)
                case IPCameraSourceConfig():
                    reachable, error = self._test_ip_camera(source)
                case VideoFileSourceConfig():
                    reachable, error = self._test_video_file(source)
                case ImagesFolderSourceConfig():
                    reachable, error = self._test_images_folder(source)
                case _:
                    return {"reachable": False, "error": f"Unsupported source type: {source.source_type}"}
        except Exception as e:
            return {"reachable": False, "error": str(e)}

        if not reachable:
            return {"reachable": False, "error": error}

        elapsed_ms = (time.monotonic() - start) * 1000
        return {"reachable": True, "latency_ms": round(elapsed_ms, 1)}

    def _test_usb_camera(self, source: USBCameraSourceConfig) -> tuple[bool, str | None]:
        cap = cv2.VideoCapture(source.config_data.device_id)
        try:
            if not cap.isOpened():
                return False, f"Cannot open USB camera device {source.config_data.device_id}"
            ret, _ = cap.read()
            if not ret:
                return False, f"USB camera device {source.config_data.device_id} opened but cannot read frames"
        finally:
            cap.release()
        return True, None

    def _test_ip_camera(self, source: IPCameraSourceConfig) -> tuple[bool, str | None]:
        stream_url = source.config_data.get_configured_stream_url()
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._TEST_TIMEOUT_MS)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._TEST_TIMEOUT_MS)
        try:
            if not cap.isOpened():
                return False, f"Cannot open stream at {source.config_data.stream_url}"
            ret, _ = cap.read()
            if not ret:
                return False, f"Stream at {source.config_data.stream_url} opened but cannot read frames"
        finally:
            cap.release()
        return True, None

    def _test_video_file(self, source: VideoFileSourceConfig) -> tuple[bool, str | None]:
        video_path = source.config_data.video_path
        if not os.path.isfile(video_path):
            return False, f"Video file not found: {video_path}"
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return False, f"File exists but cannot be opened as video: {video_path}"
        finally:
            cap.release()
        return True, None

    def _test_images_folder(self, source: ImagesFolderSourceConfig) -> tuple[bool, str | None]:
        folder_path = source.config_data.images_folder_path
        if not os.path.isdir(folder_path):
            return False, f"Directory not found: {folder_path}"
        if not os.access(folder_path, os.R_OK):
            return False, f"Directory is not accessible: {folder_path}"
        return True, None
