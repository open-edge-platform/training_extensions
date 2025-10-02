# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Generator
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.core import Scheduler
from app.db import get_db_session
from app.services import (
    ActivePipelineService,
    ConfigurationService,
    DatasetService,
    MetricsService,
    ModelService,
    PipelineService,
    ProjectService,
    SystemService,
)
from app.services.base_weights_service import BaseWeightsService
from app.services.data_collect import DataCollector
from app.services.label_service import LabelService
from app.webrtc.manager import WebRTCManager


def is_valid_uuid(identifier: str) -> bool:
    """
    Check if a given string identifier is formatted as a valid UUID

    :param identifier: String to check
    :return: True if valid UUID, False otherwise
    """
    try:
        UUID(identifier)
    except ValueError:
        return False
    return True


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


def get_source_id(source_id: str) -> UUID:
    """Initializes and validates a source ID"""
    if not is_valid_uuid(source_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid source ID")
    return UUID(source_id)


def get_project_id(project_id: str) -> UUID:
    """Initializes and validates a project ID"""
    if not is_valid_uuid(project_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project ID")
    return UUID(project_id)


def get_sink_id(sink_id: str) -> UUID:
    """Initializes and validates a sink ID"""
    if not is_valid_uuid(sink_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid sink ID")
    return UUID(sink_id)


def get_model_id(model_id: str) -> UUID:
    """Initializes and validates a model ID"""
    if not is_valid_uuid(model_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid model ID")
    return UUID(model_id)


def get_dataset_item_id(dataset_item_id: str) -> UUID:
    """Initializes and validates a dataset item ID"""
    if not is_valid_uuid(dataset_item_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid dataset item ID")
    return UUID(dataset_item_id)


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


def get_active_pipeline_service(request: Request) -> ActivePipelineService:
    """Provides an ActivePipelineService instance for managing the active pipeline."""
    return request.app.state.active_pipeline_service


def get_data_collector(request: Request) -> DataCollector:
    """Provides an DataCollector instance."""
    return request.app.state.data_collector


def get_metrics_service(scheduler: Annotated[Scheduler, Depends(get_scheduler)]) -> MetricsService:
    """Provides a MetricsService instance for collecting and retrieving metrics."""
    return MetricsService(scheduler.shm_metrics.name, scheduler.shm_metrics_lock)


def get_configuration_service(
    active_pipeline_service: Annotated[ActivePipelineService, Depends(get_active_pipeline_service)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
    db: Annotated[Session, Depends(get_db)],
) -> ConfigurationService:
    """Provides a ConfigurationService instance with the active pipeline service and config changed condition."""
    return ConfigurationService(
        active_pipeline_service=active_pipeline_service,
        db_session=db,
        config_changed_condition=scheduler.mp_config_changed_condition,
    )


def get_pipeline_service(
    active_pipeline_service: Annotated[ActivePipelineService, Depends(get_active_pipeline_service)],
    data_collector: Annotated[DataCollector, Depends(get_data_collector)],
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
    db: Annotated[Session, Depends(get_db)],
) -> PipelineService:
    """Provides a PipelineService instance with the active pipeline service and config changed condition."""
    return PipelineService(
        active_pipeline_service=active_pipeline_service,
        data_collector=data_collector,
        metrics_service=metrics_service,
        config_changed_condition=scheduler.mp_config_changed_condition,
        db_session=db,
    )


def get_system_service() -> SystemService:
    """Provides a SystemService instance for system-level operations."""
    return SystemService()


def get_model_service(
    db: Annotated[Session, Depends(get_db)],
) -> ModelService:
    """Provides a ModelService instance with the model reload event from the scheduler."""
    return ModelService(db_session=db)


def get_dataset_service(
    data_dir: Annotated[Path, Depends(get_data_dir)], db: Annotated[Session, Depends(get_db)]
) -> DatasetService:
    """Provides a DatasetService instance."""
    return DatasetService(data_dir=data_dir, db_session=db)


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager


def get_project_service(
    data_dir: Annotated[Path, Depends(get_data_dir)], db: Annotated[Session, Depends(get_db)]
) -> ProjectService:
    """Provides a ProjectService instance for managing projects."""
    return ProjectService(data_dir=data_dir, db_session=db)


def get_label_service(db: Annotated[Session, Depends(get_db)]) -> LabelService:
    """Provides a LabelService instance for managing labels."""
    return LabelService(db_session=db)


def get_base_weights_service(data_dir: Annotated[Path, Depends(get_data_dir)]) -> BaseWeightsService:
    """Provides a BaseWeightsService instance for managing base weights."""
    return BaseWeightsService(data_dir)
