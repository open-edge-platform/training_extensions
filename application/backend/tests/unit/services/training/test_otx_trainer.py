# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pathlib
from collections.abc import Callable
from unittest.mock import Mock

import pytest

from app.services.training.base import TrainingPipeline
from app.services.training.otx import OTXTrainer
from app.services.training.steps import AssignSubsetsStep, OTXTrainModelStep, PrepareWeightsStep


@pytest.fixture
def fxt_otx_trainer(
    tmp_path: pathlib.Path,
    fxt_db_session_factory: Callable,
) -> OTXTrainer:
    """Create an OTXTrainer instance."""
    return OTXTrainer(
        data_dir=tmp_path, base_weights_service=Mock(), subset_service=Mock(), db_session_factory=fxt_db_session_factory
    )


class TestOTXTrainer:
    def test_build_pipeline_returns_training_pipeline(self, fxt_otx_trainer):
        """Test that _build_pipeline returns a TrainingPipeline instance."""
        pipeline = fxt_otx_trainer._build_pipeline()
        assert isinstance(pipeline, TrainingPipeline)
        steps = pipeline._steps
        assert len(steps) == 3
        assert isinstance(steps[0], PrepareWeightsStep)
        assert isinstance(steps[1], AssignSubsetsStep)
        assert isinstance(steps[2], OTXTrainModelStep)
