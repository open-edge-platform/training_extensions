# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, UploadFile, status

from app.core import Scheduler
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
from app.settings import get_settings
from app.webrtc.manager import WebRTCManager

settings = get_settings()


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


def get_active_pipeline_service(request: Request) -> ActivePipelineService:
    """Provides an ActivePipelineService instance for managing the active pipeline."""
    return request.app.state.active_pipeline_service


def get_data_collector(request: Request) -> DataCollector:
    """Provides an DataCollector instance."""
    return request.app.state.data_collector


def get_scheduler(request: Request) -> Scheduler:
    """Provides the global Scheduler instance."""
    return request.app.state.scheduler


@lru_cache
def get_metrics_service(scheduler: Annotated[Scheduler, Depends(get_scheduler)]) -> MetricsService:
    """Provides a MetricsService instance for collecting and retrieving metrics."""
    return MetricsService(scheduler.shm_metrics.name, scheduler.shm_metrics_lock)


@lru_cache
def get_configuration_service(
    active_pipeline_service: Annotated[ActivePipelineService, Depends(get_active_pipeline_service)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> ConfigurationService:
    """Provides a ConfigurationService instance with the active pipeline service and config changed condition."""
    return ConfigurationService(
        active_pipeline_service=active_pipeline_service,
        config_changed_condition=scheduler.mp_config_changed_condition,
    )


@lru_cache
def get_pipeline_service(
    active_pipeline_service: Annotated[ActivePipelineService, Depends(get_active_pipeline_service)],
    data_collector: Annotated[DataCollector, Depends(get_data_collector)],
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> PipelineService:
    """Provides a PipelineService instance with the active pipeline service and config changed condition."""
    return PipelineService(
        active_pipeline_service=active_pipeline_service,
        data_collector=data_collector,
        metrics_service=metrics_service,
        config_changed_condition=scheduler.mp_config_changed_condition,
    )


@lru_cache
def get_system_service() -> SystemService:
    """Provides a SystemService instance for system-level operations."""
    return SystemService()


@lru_cache
def get_model_service(scheduler: Annotated[Scheduler, Depends(get_scheduler)]) -> ModelService:
    """Provides a ModelService instance with the model reload event from the scheduler."""
    return ModelService(
        data_dir=settings.data_dir,
        mp_model_reload_event=scheduler.mp_model_reload_event,
    )


@lru_cache
def get_dataset_service() -> DatasetService:
    """Provides a DatasetService instance."""
    return DatasetService(settings.data_dir)


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager


@lru_cache
def get_project_service() -> ProjectService:
    """Provides a ProjectService instance for managing projects."""
    return ProjectService(settings.data_dir)


def get_label_service() -> type[LabelService]:
    """Provides a LabelService instance for managing labels."""
    return LabelService


@lru_cache
def get_base_weights_service() -> BaseWeightsService:
    """Provides a BaseWeightsService instance for managing base weights."""
    return BaseWeightsService(settings.data_dir)
