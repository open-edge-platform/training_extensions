"""Integration tests for the Anomalib engine.

Tests if one of the model from Anomalib can be used with AnomalyEngine.
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from anomalib import TaskType
from anomalib.data import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import ValSplitMode
from anomalib.models import Padim
import pandas as pd

from otx.backend.anomalib.engine import AnomalyEngine


class DummyDataset(AnomalibDataset):
    """Dummy dataset that streams fake tensors."""

    def __init__(self) -> None:
        super().__init__(task=TaskType.CLASSIFICATION)

    @property
    def samples(self) -> pd.DataFrame:
        return pd.DataFrame({
            "image_path": ["test.png"] * 10 + ["test_anomalous.png"] * 10,
            "label_index": [0] * 10 + [1] * 10,
            "label": ["normal"] * 10 + ["anomalous"] * 10,
            "split": ["train"] * 10 + ["test"] * 10,
        })

    @samples.setter
    def samples(self, samples: pd.DataFrame) -> None:
        """Overwrite the samples with a new dataframe."""
        self._samples = samples

    def __len__(self) -> int:
        """Length of the dataset."""
        return 20

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get item from the dataset."""
        item = self._samples.iloc[index]
        image = torch.randn(3, 256, 256)
        return {"image": image, "label_index": item["label_index"], "label": item["label"]}


class DummyDataModule(AnomalibDataModule):
    """Dummy data module that streams fake tensors."""

    def __init__(self) -> None:
        super().__init__(
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=2,
            val_split_mode=ValSplitMode.FROM_TRAIN,
            val_split_ratio=0.2,
        )

    def _setup(self, _stage: str | None = None) -> None:
        """Setup the data module."""
        self.train_data = DummyDataset()
        self.val_data = DummyDataset()
        self.test_data = DummyDataset()


@pytest.fixture(scope="module")
def model():
    """Fixture for the model."""
    return Padim()


@pytest.fixture(scope="module")
def data():
    """Fixture for the data."""
    return DummyDataModule()


class TestAnomalibEngine:
    """Test the Anomalib engine."""

    def test_train(self, model, data):
        """Test the train method."""
        engine = AnomalyEngine(model, data)
        engine.train()

    # def test_test(self, model, data):
    #     """Test the test method."""
    #     raise AssertionError

    # def test_predict(self, model, data):
    #     """Test the predict method."""
    #     raise AssertionError

    # def test_export(self, model, data):
    #     """Test the export method."""
    #     raise AssertionError

    # def test_optimize(self, model, data):
    #     """Test the optimize method."""
    #     raise AssertionError
