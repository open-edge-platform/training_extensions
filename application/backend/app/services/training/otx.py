# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from sqlalchemy.orm import Session

from app.services.base_weights_service import BaseWeightsService

from .base import Trainer, TrainingPipeline
from .steps import AssignSubsetsStep, OTXTrainModelStep, PrepareWeightsStep
from .steps.subset_assignment import SubsetAssigner, SubsetService

logger = logging.getLogger(__name__)


class OTXTrainer(Trainer):
    def __init__(
        self,
        data_dir: Path,
        base_weights_service: BaseWeightsService,
        subset_service: SubsetService,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ):
        super().__init__()
        self._data_dir = data_dir
        self._base_weights_service = base_weights_service
        self._subset_service = subset_service
        self._db_session_factory = db_session_factory

    def _build_pipeline(self) -> TrainingPipeline:
        """Build OTX-specific training pipeline."""
        return TrainingPipeline(
            [
                PrepareWeightsStep(self._base_weights_service, self._data_dir),
                AssignSubsetsStep(self._subset_service, SubsetAssigner(), self._db_session_factory),
                OTXTrainModelStep(),
            ]
        )
