# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from app.core.run import ExecutionContext, Runnable
from app.services.base_weights_service import BaseWeightsService

from .models import TrainingParams


class Trainer(Runnable, ABC):
    def __init__(self, base_weights_service: BaseWeightsService):
        self._weights_service = base_weights_service

    @staticmethod
    def _build_model_weights_path(data_dir: Path, project_id: UUID, model_id: UUID) -> Path:
        return data_dir / "projects" / str(project_id) / "models" / str(model_id) / "model.pth"

    @staticmethod
    def _get_training_params(ctx: ExecutionContext) -> TrainingParams:
        return TrainingParams.model_validate_json(ctx.payload)

    @abstractmethod
    def _prepare_weights(self, data_dir: Path, training_params: TrainingParams, report_fn: Callable) -> Path: ...

    @abstractmethod
    def _train(self): ...

    @abstractmethod
    def _evaluate(self): ...
