# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from loguru import logger
from sqlalchemy.orm import Session

from app.db.schema import PipelineDB
from app.models import Pipeline, PipelineStatus
from app.models.model_revision import ModelFormat, ModelPrecision, TrainingStatus
from app.repositories import PipelineRepository
from app.repositories.model_revision_repo import ModelRevisionRepository
from app.repositories.model_variant_repo import ModelVariantRepository
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.event.event_bus import EventBus, EventType
from app.services.parent_process_guard import parent_process_only

from . import BaseSessionManagedService
from .system_service import DEFAULT_DEVICE, SystemService

MSG_ERR_DELETE_RUNNING_PIPELINE = "Cannot delete a running pipeline."


class OtherProjectActiveError(Exception):
    """
    Exception raised when trying to run a pipeline in one project, while a pipeline of another project is still running.
    """

    def __init__(self, requested_project_id: str, active_project_id: str):
        super().__init__(
            f"Attempted to enable a pipeline in project with ID {requested_project_id}, while a pipeline is still "
            f"enabled in another project with ID {active_project_id}. Please first disable pipeline in project with "
            f"ID {active_project_id}"
        )


class IncompatibleModelVariantError(ValueError):
    """Exception raised when a model variant is incompatible for pipeline inference (e.g., not OpenVINO format)."""


class DeviceInt8NotSupportedError(Exception):
    """Exception raised when INT8 inference is requested on a device that does not support it."""

    def __init__(self, device: str):
        super().__init__(
            f"INT8 inference is not supported on device '{device}'. "
            f"Please select a non-quantized (FP16/FP32) model variant or use a device that supports INT8."
        )


class PipelineService(BaseSessionManagedService):
    def __init__(
        self,
        system_service: SystemService | None = None,
        event_bus: EventBus | None = None,
        db_session: Session | None = None,
    ) -> None:
        super().__init__(db_session)
        self._event_bus: EventBus | None = event_bus
        self._system_service: SystemService | None = system_service

    def create_pipeline(self, project_id: UUID) -> Pipeline:
        pipeline_repo = PipelineRepository(self.db_session)
        pipeline_db = PipelineDB(
            project_id=str(project_id),
        )
        created = pipeline_repo.save(pipeline_db)
        return Pipeline.model_validate(created)

    def get_active_pipeline(self) -> Pipeline | None:
        """Retrieve an active pipeline."""
        if self._system_service is None:
            raise ValueError("System service is required to get active pipeline.")
        pipeline_repo = PipelineRepository(self.db_session)
        pipeline_db = pipeline_repo.get_active_pipeline()
        if pipeline_db is None:
            return None

        if not self._system_service.validate_device(pipeline_db.device):
            logger.warning(
                "The configured device '{}' is not available for pipeline '{}'. Falling back to 'cpu'.",
                pipeline_db.device,
                pipeline_db.project_id,
            )
            pipeline_db.device = DEFAULT_DEVICE
            pipeline_repo.update(pipeline_db)
        return Pipeline.model_validate(pipeline_db)

    def get_pipeline_by_id(self, project_id: UUID) -> Pipeline:
        """Retrieve a pipeline by project ID."""
        pipeline_repo = PipelineRepository(self.db_session)
        pipeline_db = pipeline_repo.get_by_id(str(project_id))
        if not pipeline_db:
            raise ResourceNotFoundError(ResourceType.PIPELINE, str(project_id))
        return Pipeline.model_validate(pipeline_db)

    def is_running(self, project_id: UUID) -> bool:
        """Retrieve a pipeline status by project ID."""
        pipeline_repo = PipelineRepository(self.db_session)
        return pipeline_repo.is_running(str(project_id))

    @parent_process_only
    def update_pipeline(self, project_id: UUID, partial_config: dict) -> Pipeline:
        """Update an existing pipeline."""
        pipeline = self.get_pipeline_by_id(project_id)
        base = pipeline.model_dump()

        # When model_id changes without an explicit model_variant_id, any existing model_variant_id should be cleared
        # before merging dicts so that _validate_model_and_resolve_variant can default to the FP16 OpenVINO variant.
        new_model_id = partial_config.get("model_id") or partial_config.get("model_revision_id")
        if new_model_id is not None and "model_variant_id" not in partial_config:
            base["model_variant_id"] = None

        to_update = type(pipeline).model_validate({**base, **partial_config})
        pipeline_repo = PipelineRepository(self.db_session)
        to_update_db = PipelineDB(
            project_id=str(to_update.project_id),
            source_id=str(to_update.source_id) if to_update.source_id else None,
            sink_id=str(to_update.sink_id) if to_update.sink_id else None,
            model_revision_id=str(to_update.model_id) if to_update.model_id else None,
            model_variant_id=str(to_update.model_variant_id) if to_update.model_variant_id else None,
            is_running=to_update.status.as_bool,
            data_collection=to_update.data_collection.model_dump(),
            device=to_update.device,
        )

        # Validate pipeline data
        if to_update_db.is_running:
            # Only one pipeline can run at the same time. Note that only one pipeline per project exists.
            active_pipeline_db = pipeline_repo.get_active_pipeline()
            if active_pipeline_db is not None and to_update_db.project_id != active_pipeline_db.project_id:
                raise OtherProjectActiveError(
                    requested_project_id=to_update_db.project_id, active_project_id=active_pipeline_db.project_id
                )
        if to_update_db.model_revision_id is not None:
            self._validate_model_and_resolve_variant(to_update_db)

        pipeline_db = pipeline_repo.update(to_update_db)
        updated = Pipeline.model_validate(pipeline_db)
        self.__emit_event(pipeline, updated)
        return updated

    def _validate_model_and_resolve_variant(self, pipeline_db: PipelineDB) -> None:
        """Validate the model revision and resolve/validate the model variant for inference.

        Ensures that:
        - The model revision exists and was successfully trained.
        - If a model_variant_id is provided, it belongs to the revision, is in OpenVINO format,
          and the device supports INT8 when the variant is quantized.
        - If no model_variant_id is provided, defaults to the FP16 OpenVINO variant.

        The pipeline_db.model_variant_id field may be mutated in-place when the default variant
        is resolved.

        Args:
            pipeline_db: The PipelineDB instance whose model_revision_id and model_variant_id to validate.

        Raises:
            ResourceNotFoundError: If the model revision or variant is not found.
            ValueError: If the model revision is not successfully trained.
            IncompatibleModelVariantError: If the variant is not OpenVINO or cannot be resolved.
            DeviceInt8NotSupportedError: If the device does not support INT8 inference.
        """
        model_revision_id: str = pipeline_db.model_revision_id  # type: ignore[union-attr]

        # Only successfully trained models can be part of a pipeline
        model_revision_repo = ModelRevisionRepository(project_id=pipeline_db.project_id, db=self.db_session)
        model_revision_db = model_revision_repo.get_by_id(model_revision_id)
        if model_revision_db is None:
            raise ResourceNotFoundError(resource_type=ResourceType.MODEL, resource_id=model_revision_id)
        if model_revision_db.training_status != TrainingStatus.SUCCESSFUL:
            raise ValueError(
                f"Provided model id ({model_revision_id}) points to a model that was not successfully "
                f"trained (status is {model_revision_db.training_status})."
            )

        # Validate and resolve model_variant_id
        model_variant_repo = ModelVariantRepository(db=self.db_session)
        if pipeline_db.model_variant_id is not None:
            # Explicit variant specified: validate it
            variant_db = model_variant_repo.get_by_id(pipeline_db.model_variant_id)
            if variant_db is None or variant_db.files_deleted:
                raise ResourceNotFoundError(resource_type=ResourceType.MODEL, resource_id=pipeline_db.model_variant_id)
            if variant_db.model_revision_id != model_revision_id:
                raise IncompatibleModelVariantError(
                    f"Model variant '{pipeline_db.model_variant_id}' does not belong to "
                    f"model revision '{model_revision_id}'."
                )
            if variant_db.format != ModelFormat.OPENVINO:
                raise IncompatibleModelVariantError(
                    f"Only OpenVINO model variants can be used for inference. "
                    f"The selected variant has format '{variant_db.format}'."
                )
            if variant_db.precision == ModelPrecision.INT8:
                self._validate_int8_support(pipeline_db.device)
        else:
            # No variant specified: default to FP16 OpenVINO variant
            default_variant = model_variant_repo.get_by_revision_and_format_and_precision(
                model_revision_id=model_revision_id,
                format=ModelFormat.OPENVINO,
                precision=ModelPrecision.FP16,
            )
            if default_variant is None or default_variant.files_deleted:
                raise IncompatibleModelVariantError(
                    f"No FP16 OpenVINO variant found for model revision '{model_revision_id}'. "
                    f"Please specify a model_variant_id explicitly."
                )
            pipeline_db.model_variant_id = default_variant.id

    def __emit_event(self, pipeline: Pipeline, updated: Pipeline) -> None:
        if self._event_bus is None:
            raise ValueError(
                "Event bus is required to update pipeline. This is because updating pipeline may trigger events that "
                "require other services to react."
            )
        if pipeline.status == PipelineStatus.RUNNING and updated.status == PipelineStatus.RUNNING:
            # If the pipeline source_id or sink_id is being updated while running
            if pipeline.source_id != updated.source_id:
                self._event_bus.emit_event(EventType.SOURCE_CHANGED)
            if pipeline.sink_id != updated.sink_id:
                # Sink may be None (disconnected): in that case predictions are only routed to WebRTC.
                self._event_bus.emit_event(EventType.SINK_CHANGED)
            if pipeline.data_collection != updated.data_collection:
                self._event_bus.emit_event(EventType.PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED)
            if pipeline.device != updated.device:
                self._event_bus.emit_event(EventType.INFERENCE_DEVICE_CHANGED)
            if pipeline.model_id != updated.model_id or pipeline.model_variant_id != updated.model_variant_id:
                self._event_bus.emit_event(EventType.MODEL_CHANGED)
        elif pipeline.status != updated.status:
            # If the pipeline is being activated or stopped
            self._event_bus.emit_event(EventType.PIPELINE_STATUS_CHANGED)

    def _validate_int8_support(self, device: str) -> None:
        """Validate that the device supports INT8 inference.

        Args:
            device: Device string (e.g., 'cpu', 'xpu', 'xpu-1').

        Raises:
            DeviceInt8NotSupportedError: If the device does not support INT8 inference.
        """
        if self._system_service is None:
            raise ValueError("System service is required to validate INT8 support.")
        device_info = self._system_service.get_device_info(device)
        if not self._system_service.supports_int8(device_info):
            raise DeviceInt8NotSupportedError(device)
