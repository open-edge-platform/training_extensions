"""Integration tests for the Anomalib engine.

Tests if one of the model from Anomalib can be used with AnomalyEngine.
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from anomalib.data import AnomalibDataModule, Folder
from anomalib.models import Padim
from lightning.pytorch import LightningDataModule, LightningModule

from otx.backend.anomalib.engine import AnomalyEngine
from otx.data import OTXDataItem

if TYPE_CHECKING:
    from anomalib.models.components import AnomalyModule


class TestIncorrectModel(LightningModule):
    """Unsupported model."""


class TestIncorrectData(LightningDataModule):
    """Unsupported data."""


@pytest.fixture(scope="module")
def incorrect_model():
    """Fixture for the incorrect model."""
    return TestIncorrectModel()


@pytest.fixture(scope="module")
def incorrect_data():
    """Fixture for the incorrect data."""
    return TestIncorrectData()


@pytest.fixture(scope="module")
def model():
    """Fixture for the model."""
    return Padim()


@pytest.fixture(scope="module")
def data():
    """Fixture for the data."""
    return Folder(
        root="tests/assets/anomaly_hazelnut",
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/colour",
        name="hazelnut",
        mask_dir="ground_truth/colour",
    )


class TestAnomalibEngine:
    """Test the Anomalib engine."""

    def test_train(self, model: Padim, data: Folder, tmpdir: Path):
        """Test the train method."""
        engine = AnomalyEngine(model, data, work_dir=tmpdir)
        engine.train()

    def test_test(self, model: Padim, data: Folder, tmpdir: Path):
        """Test the test method."""
        engine = AnomalyEngine(model, data, work_dir=tmpdir)
        result_metrics = engine.test()
        assert result_metrics is not None
        assert isinstance(result_metrics, list)
        assert isinstance(result_metrics[0], dict)

    def test_predict(self, model: Padim, data: Folder, tmpdir: Path):
        """Test the predict method."""
        engine = AnomalyEngine(model, data, work_dir=tmpdir)
        result_annotations = engine.predict()
        assert result_annotations is not None
        assert isinstance(result_annotations, list)
        assert isinstance(result_annotations[0], OTXDataItem)

    def test_export(self, model: Padim, data: Folder, tmpdir: Path):
        """Test the export method."""
        engine = AnomalyEngine(model, data, work_dir=tmpdir)
        result_path = engine.export()
        assert result_path.exists()

    def test_is_supported(
        self,
        model: AnomalyModule,
        data: AnomalibDataModule,
        incorrect_model: LightningModule,
        incorrect_data: LightningDataModule,
    ):
        """Test is supported."""
        # Test with AnomalibModel and AnomalibDataModule
        assert AnomalyEngine.is_supported(model, data)

        # Test with non-AnomalibModel
        assert not AnomalyEngine.is_supported(incorrect_model, data)

        # Test with non-AnomalibDataModule
        assert not AnomalyEngine.is_supported(model, incorrect_data)

        # Test with non-AnomalibModel and non-AnomalibDataModule
        assert not AnomalyEngine.is_supported(incorrect_model, incorrect_data)
