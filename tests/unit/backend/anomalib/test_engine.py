"""Test Anomalib engine."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
from anomalib import LearningType
from anomalib.data import AnomalibDataModule
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine as _AnomalibEngine
from anomalib.models.components import AnomalyModule
from lightning.pytorch import LightningDataModule, LightningModule

from otx.backend.anomalib.engine import AnomalyEngine


class TestAnomalyModel(AnomalyModule):
    """Test Anomalib model."""

    @property
    def learning_type(self) -> LearningType:
        """Learning type."""
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Trainer arguments."""
        return {}


class TestAnomalyDataModule(AnomalibDataModule):
    """Test Anomalib data module."""

    def __init__(self) -> None:
        super().__init__(
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=1,
            val_split_mode=ValSplitMode.FROM_TRAIN,
            val_split_ratio=0.2,
        )

    def _setup(self, _stage: str | None = None) -> None:
        """Setup the data module."""


class TestOtherModel(LightningModule):
    """Test other model."""


class TestOtherDataModule(LightningDataModule):
    """Test other data module."""


@pytest.fixture(scope="module")
def model():
    """Fixture for Anomalib model."""
    return TestAnomalyModel()


@pytest.fixture(scope="module")
def data():
    """Fixture for Anomalib data module."""
    return TestAnomalyDataModule()


@pytest.fixture(scope="module")
def other_model():
    """Fixture for other model."""
    return TestOtherModel()


@pytest.fixture(scope="module")
def other_data():
    """Fixture for other data module."""
    return TestOtherDataModule()


class TestAnomalibEngine:
    """Test Anomalib engine."""

    def test_create_engine(self, mocker):
        """Test create engine."""
        mock_model = mocker.Mock(spec=AnomalyModule)
        mock_data = mocker.Mock(spec=AnomalibDataModule)
        engine = AnomalyEngine(mock_model, mock_data)
        assert isinstance(engine._create_anomalib_engine(), _AnomalibEngine)

    def test_train(self, model, data):
        """Test train."""
        engine = AnomalyEngine(model, data)
        engine.train()

    def test_test(self):
        """Test test."""
        raise AssertionError()

    def test_predict(self):
        """Test predict."""
        raise AssertionError()

    def test_export(self):
        """Test export."""
        raise AssertionError()


    def test_is_supported(
        self,
        model: AnomalyModule,
        data: AnomalibDataModule,
        other_model: LightningModule,
        other_data: LightningDataModule,
    ):
        """Test is supported."""
        # Test with AnomalibModel and AnomalibDataModule
        assert AnomalyEngine.is_supported(model, data)

        # Test with non-AnomalibModel
        assert not AnomalyEngine.is_supported(other_model, data)

        # Test with non-AnomalibDataModule
        assert not AnomalyEngine.is_supported(model, other_data)

        # Test with non-AnomalibModel and non-AnomalibDataModule
        assert not AnomalyEngine.is_supported(other_model, other_data)

