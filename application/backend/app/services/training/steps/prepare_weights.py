# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import UUID

from app.core.run import ExecutionContext
from app.services.base_weights_service import BaseWeightsService
from app.services.training.base import PipelineContext, TrainingParams, TrainingStep

MODEL_WEIGHTS_PATH = "model_weights_path"


class PrepareWeightsStep(TrainingStep):
    def __init__(self, weights_service: BaseWeightsService, data_dir: Path):
        self._weights_service = weights_service
        self._data_dir = data_dir

    def execute(self, _ctx: ExecutionContext, params: TrainingParams, pipeline_ctx: PipelineContext) -> None:
        """
        Prepare weights for training based on training parameters.

        If a parent model revision ID is provided, it fetches the weights from the parent model. Otherwise, it retrieves
        the base weights for the specified model architecture.

        Raises:
            ValueError: If project ID is not provided when parent model revision ID is specified.
        """
        if params.parent_model_revision_id is None:
            weights_path = self._weights_service.get_local_weights_path(
                task=params.task_type, model_manifest_id=params.model_architecture_id
            )
            pipeline_ctx.set(MODEL_WEIGHTS_PATH, weights_path)
            return
        if params.project_id is None:
            raise ValueError("Project ID must be provided for parent model weights preparation")
        weights_path = self._build_model_weights_path(
            self._data_dir, params.project_id, params.parent_model_revision_id
        )
        if not weights_path.exists():
            raise FileNotFoundError(f"Parent model weights not found at {weights_path}")
        pipeline_ctx.set(MODEL_WEIGHTS_PATH, weights_path)

    def get_name(self) -> str:
        return "Prepare Model Weights"

    @staticmethod
    def _build_model_weights_path(data_dir: Path, project_id: UUID, model_id: UUID) -> Path:
        return data_dir / "projects" / str(project_id) / "models" / str(model_id) / "model.pth"
