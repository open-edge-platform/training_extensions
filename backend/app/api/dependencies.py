# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status

from app.core import Scheduler
from app.services import ActivePipelineService, ConfigurationService, ModelService, PipelineService, SystemService
from app.webrtc.manager import WebRTCManager


def validate_uuid(identifier: str, error_detail: str) -> UUID:
    """Validates a string as UUID or raises HTTP 400 with a custom error message."""
    try:
        return UUID(identifier)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_detail)


def get_source_id(source_id: str) -> UUID:
    """Initializes and validates a source ID"""
    return validate_uuid(source_id, "Invalid source ID")


def get_sink_id(sink_id: str) -> UUID:
    """Initializes and validates a sink ID"""
    return validate_uuid(sink_id, "Invalid sink ID")


def get_model_id(model_id: str) -> UUID:
    """Initializes and validates a model ID"""
    return validate_uuid(model_id, "Invalid model ID")


def get_pipeline_id(pipeline_id: str) -> UUID:
    """Initializes and validates a pipeline ID"""
    return validate_uuid(pipeline_id, "Invalid pipeline ID")


def get_media_id(media_id: str) -> UUID:
    """Initializes and validates a media ID"""
    return validate_uuid(media_id, "Invalid media ID")


@lru_cache
def get_active_pipeline_service() -> ActivePipelineService:
    """Provides an ActivePipelineService instance for managing the active pipeline."""
    return ActivePipelineService()


def get_scheduler(request: Request) -> Scheduler:
    """Provides the global Scheduler instance."""
    return request.app.state.scheduler


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
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> PipelineService:
    """Provides a PipelineService instance with the active pipeline service and config changed condition."""
    return PipelineService(
        active_pipeline_service=active_pipeline_service,
        config_changed_condition=scheduler.mp_config_changed_condition,
    )


@lru_cache
def get_system_service() -> SystemService:
    """Provides a SystemService instance for system-level operations."""
    return SystemService()


@lru_cache
def get_model_service(
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> ModelService:
    """Provides a ModelService instance with the model reload event from the scheduler."""
    return ModelService(
        mp_model_reload_event=scheduler.mp_model_reload_event,
    )


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager
