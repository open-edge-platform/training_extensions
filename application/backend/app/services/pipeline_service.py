# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from uuid import UUID

from loguru import logger
from sqlalchemy.orm import Session

from app.db.schema import PipelineDB
from app.models import FolderSinkConfig, Pipeline, PipelineStatus, SinkAdapter
from app.models.model_revision import ModelFormat, ModelPrecision, TrainingStatus
from app.repositories import PipelineRepository, SinkRepository
from app.repositories.model_revision_repo import ModelRevisionRepository
from app.repositories.model_variant_repo import ModelVariantRepository
from app.repositories.project_repo import ProjectRepository
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

    def __init__(
        self,
        requested_project_name: str,
        requested_project_id: str,
        active_project_name: str,
        active_project_id: str,
    ):
        super().__init__(
            f"Attempted to enable a pipeline in project '{requested_project_name}' (ID: {requested_project_id}), "
            f"while a pipeline is still enabled in another project '{active_project_name}' (ID: {active_project_id}). "
            f"Please first disable pipeline in project '{active_project_name}' (ID: {active_project_id})."
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


class FolderSinkNotAccessibleError(Exception):
    """Exception raised when the folder sink path cannot be created or written to."""

    def __init__(self, folder_path: str, reason: str):
        super().__init__(
            f"Folder sink path '{folder_path}' is not accessible: {reason}. "
            f"Please ensure the path exists or can be created and that write permissions are granted."
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

        self._validate_model_and_resolve_variant(pipeline=pipeline, partial_config=partial_config)

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
                project_repo = ProjectRepository(db=self.db_session)
                to_update_project = project_repo.get_by_id(to_update_db.project_id)
                active_project = project_repo.get_by_id(active_pipeline_db.project_id)
                raise OtherProjectActiveError(
                    requested_project_name=to_update_project.name if to_update_project is not None else "",
                    requested_project_id=to_update_db.project_id,
                    active_project_name=active_project.name if active_project is not None else "",
                    active_project_id=active_pipeline_db.project_id,
                )
            # Validate folder sink accessibility when activating the pipeline
            if to_update_db.sink_id:
                sink_id = to_update_db.sink_id
                db_sink = SinkRepository(self.db_session).get_by_id(sink_id)
                sink_config = SinkAdapter.validate_python(db_sink, from_attributes=True)
                if isinstance(sink_config, FolderSinkConfig):
                    self._validate_folder_sink(sink_config)

        pipeline_db = pipeline_repo.update(to_update_db)
        updated = Pipeline.model_validate(pipeline_db)
        self.__emit_event(pipeline, updated)
        return updated

    def _validate_model_and_resolve_variant(self, pipeline: Pipeline, partial_config: dict) -> None:  # noqa: C901
        """Validate the model revision and resolve/validate the model variant for inference.

        Ensures that:
        - At least the model revision id is provided, or both the model revision id and model variant id
        - The model revision exists and was successfully trained.
        - If a model_variant_id is provided, it belongs to the revision, is in OpenVINO format,
          and the device supports INT8 when the variant is quantized.
        - If no model_variant_id is provided, defaults to the FP16 OpenVINO variant.

        The ``partial_config`` dict may be mutated in-place: when no ``model_variant_id`` is
        supplied by the caller, this method resolves and writes the default FP16 OpenVINO
        variant id into ``partial_config["model_variant_id"]``.

        Args:
            pipeline: The current Pipeline entity, used as a fallback for fields not present
                in ``partial_config`` (e.g. ``model_id``, ``device``) and for the project id.
            partial_config: The partial update dict provided to ``update_pipeline``. Keys of
                interest are ``model_id``, ``model_variant_id`` and ``device``.

        Raises:
            ResourceNotFoundError: If the model revision or variant is not found, or only the model variant is passed.
            ValueError: If the model revision is not successfully trained.
            IncompatibleModelVariantError: If the variant is not OpenVINO or cannot be resolved.
            DeviceInt8NotSupportedError: If the device does not support INT8 inference.
        """
        model_revision_id = partial_config.get("model_id") or partial_config.get("model_revision_id")
        model_variant_id = partial_config.get("model_variant_id")
        device = partial_config.get("device", pipeline.device)

        if model_revision_id is None:
            if model_variant_id is not None:
                raise ValueError("It is not possible to provide only a model variant ID")
            return  # Nothing to validate if no model is configured (e.g. clearing the pipeline).

        model_revision_id = str(model_revision_id)
        model_variant_id = str(model_variant_id) if model_variant_id else None

        # Only successfully trained models can be part of a pipeline
        model_revision_repo = ModelRevisionRepository(project_id=str(pipeline.project_id), db=self.db_session)
        model_revision_db = model_revision_repo.get_by_id(model_revision_id)
        if model_revision_db is None:
            raise ResourceNotFoundError(resource_type=ResourceType.MODEL, resource_id=model_revision_id)
        if model_revision_db.training_status != TrainingStatus.SUCCESSFUL:
            raise ValueError(
                f"Provided model id ({model_revision_id}) points to a model that was not successfully "
                f"trained (status is {model_revision_db.training_status})."
            )

        # Validate and resolve model
        model_variant_repo = ModelVariantRepository(db=self.db_session)
        if model_revision_id and model_variant_id:
            # Explicit variant specified: validate it
            variant_db = model_variant_repo.get_by_id(model_variant_id)
            if variant_db is None or variant_db.files_deleted:
                raise ResourceNotFoundError(resource_type=ResourceType.MODEL_VARIANT, resource_id=model_variant_id)
            if variant_db.model_revision_id != model_revision_id:
                raise IncompatibleModelVariantError(
                    f"Model variant '{model_variant_id}' does not belong to model revision '{model_revision_id}'."
                )
            if variant_db.format != ModelFormat.OPENVINO:
                raise IncompatibleModelVariantError(
                    f"Only OpenVINO model variants can be used for inference. "
                    f"The selected variant has format '{variant_db.format}'."
                )
            if variant_db.precision == ModelPrecision.INT8:
                self._validate_int8_support(device)
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
            partial_config["model_variant_id"] = default_variant.id

    @staticmethod
    def _validate_folder_sink(sink_config: FolderSinkConfig) -> None:
        """
        Validate that the folder sink path can be created and written to.

        Args:
            sink_config: The FolderSinkConfig to validate.

        Raises:
            FolderSinkNotAccessibleError: If the folder cannot be created.
        """
        folder_path = sink_config.config_data.folder_path
        try:
            os.makedirs(folder_path, exist_ok=True)
        except OSError as e:
            raise FolderSinkNotAccessibleError(folder_path, str(e)) from e

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
