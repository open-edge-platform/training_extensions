# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Generator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.api.validators import ProjectID, SinkID, SourceID
from app.core.jobs.control_plane import JobQueue
from app.db import get_db_session
from app.models import Sink, Source
from app.scheduler import Scheduler
from app.schemas import ProjectView
from app.services import (
    BaseWeightsService,
    DatasetService,
    LabelService,
    MetricsService,
    ModelService,
    PipelineMetricsService,
    PipelineService,
    ProjectService,
    ResourceNotFoundError,
    SinkService,
    SourceUpdateService,
    SystemService,
)
from app.services.data_collect import DataCollector
from app.services.event.event_bus import EventBus
from app.services.training_configuration_service import TrainingConfigurationService
from app.webrtc.manager import WebRTCManager


def get_file_name_and_extension(file: UploadFile) -> tuple[str, str]:
    """Return the file name and extension"""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="File name cannot be empty.")
    full_name = file.filename.strip()
    if not full_name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="File name cannot be empty.")
    file_name, file_ext = os.path.splitext(full_name)
    file_name = file_name.strip()  # remove whitespace characters between the basename and the extension
    file_ext = file_ext[1:]  # remove leading dot in the extension
    if not file_ext:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="File extension cannot be empty.")
    return file_name, file_ext


def get_file_size(file: UploadFile) -> int:
    """Return the file size in bytes"""
    if not file.size:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="File size should be defined.")
    return file.size


def get_db() -> Generator[Session]:
    """Provides a database session."""
    with get_db_session() as session:
        yield session


def get_scheduler(request: Request) -> Scheduler:
    """Provides the global Scheduler instance."""
    return request.app.state.scheduler


def get_data_dir(request: Request) -> Path:
    """Provides the data directory path from settings."""
    return request.app.state.settings.data_dir


def get_job_dir(request: Request) -> Path:
    """Provides the job log directory path from settings."""
    return request.app.state.settings.job_dir


def get_event_bus(request: Request) -> EventBus:
    """Provides an EventBus instance."""
    return request.app.state.event_bus


def get_data_collector(request: Request) -> DataCollector:
    """Provides an DataCollector instance."""
    return request.app.state.data_collector


def get_metrics_service(scheduler: Annotated[Scheduler, Depends(get_scheduler)]) -> MetricsService:
    """Provides a MetricsService instance for collecting and retrieving metrics."""
    return MetricsService(scheduler.shm_metrics.name, scheduler.shm_metrics_lock)


def get_sink_service(
    event_bus: Annotated[EventBus, Depends(get_event_bus)], db: Annotated[Session, Depends(get_db)]
) -> SinkService:
    """Provides a SinkService instance."""
    return SinkService(event_bus=event_bus, db_session=db)


def get_source_update_service(
    event_bus: Annotated[EventBus, Depends(get_event_bus)], db: Annotated[Session, Depends(get_db)]
) -> SourceUpdateService:
    """Provides a SourceUpdateService instance."""
    return SourceUpdateService(event_bus=event_bus, db_session=db)


def get_pipeline_service(
    event_bus: Annotated[EventBus, Depends(get_event_bus)],
    db: Annotated[Session, Depends(get_db)],
) -> PipelineService:
    """Provides a PipelineService instance ."""
    return PipelineService(event_bus=event_bus, db_session=db)


def get_pipeline_metrics_service(
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
) -> PipelineMetricsService:
    """Provides a PipelineMetricsService instance."""
    return PipelineMetricsService(
        pipeline_service=pipeline_service,
        metrics_service=metrics_service,
    )


def get_system_service() -> SystemService:
    """Provides a SystemService instance for system-level operations."""
    return SystemService()


def get_model_service(
    data_dir: Annotated[Path, Depends(get_data_dir)],
    db: Annotated[Session, Depends(get_db)],
) -> ModelService:
    """Provides a ModelService instance with the model reload event from the scheduler."""
    return ModelService(data_dir=data_dir, db_session=db)


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager


def get_label_service(db: Annotated[Session, Depends(get_db)]) -> LabelService:
    """Provides a LabelService instance for managing labels."""
    return LabelService(db_session=db)


def get_project_service(
    data_dir: Annotated[Path, Depends(get_data_dir)],
    db: Annotated[Session, Depends(get_db)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
    label_service: Annotated[LabelService, Depends(get_label_service)],
) -> ProjectService:
    """Provides a ProjectService instance for managing projects."""
    return ProjectService(
        data_dir=data_dir, label_service=label_service, pipeline_service=pipeline_service, db_session=db
    )


def get_dataset_service(
    data_dir: Annotated[Path, Depends(get_data_dir)],
    label_service: Annotated[LabelService, Depends(get_label_service)],
    db: Annotated[Session, Depends(get_db)],
) -> DatasetService:
    """Provides a DatasetService instance."""
    return DatasetService(data_dir=data_dir, label_service=label_service, db_session=db)


def get_project(
    project_id: ProjectID,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectView:
    """Provides a ProjectView instance for request scoped project."""
    try:
        return project_service.get_project_by_id(project_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


def get_sink(
    sink_id: SinkID,
    sink_service: Annotated[SinkService, Depends(get_sink_service)],
) -> Sink:
    """Provides a Sink instance for request scoped sink."""
    try:
        return sink_service.get_by_id(sink_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


def get_source(
    source_id: SourceID,
    source_update_service: Annotated[SourceUpdateService, Depends(get_source_update_service)],
) -> Source:
    """Provides a Source instance for request scoped source."""
    try:
        return source_update_service.get_by_id(source_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


def get_base_weights_service(data_dir: Annotated[Path, Depends(get_data_dir)]) -> BaseWeightsService:
    """Provides a BaseWeightsService instance for managing base weights."""
    return BaseWeightsService(data_dir)


def get_job_queue(request: Request) -> JobQueue:
    """Provides the global JobQueue instance from FastAPI application's state."""
    return request.app.state.job_queue


def get_training_configuration_service(db: Annotated[Session, Depends(get_db)]) -> TrainingConfigurationService:
    """Provides a TrainingConfigurationService instance for managing training configurations."""
    return TrainingConfigurationService(db_session=db)
