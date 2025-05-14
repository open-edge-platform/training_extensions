"""Anomalib engine."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from anomalib.data import AnomalibDataModule
from anomalib.engine import Engine as _AnomalibEngine
from anomalib.models.components import AnomalyModule
from anomalib.deploy import ExportType
from otx.engine import Engine
from otx.types import ANNOTATIONS, METRICS


class AnomalyEngine(Engine):
    """Anomalib engine."""

    def __init__(self, model: AnomalyModule, data: AnomalibDataModule, **kwargs):
        self.model = model
        self.data = data

    def train(self, **kwargs) -> METRICS:
        """Train the model."""
        engine = self._create_anomalib_engine(**kwargs)
        engine.train(model=self.model, datamodule=self.data)

    def test(self, **kwargs) -> METRICS:
        """Test the model."""
        engine = self._create_anomalib_engine(**kwargs)
        engine.test(model=self.model, datamodule=self.data)

    def predict(self, **kwargs) -> ANNOTATIONS:
        """Predict the model."""
        engine = self._create_anomalib_engine(**kwargs)
        return engine.predict(model=self.model, datamodule=self.data)

    def export(self, **kwargs) -> Path:
        """Export the model."""
        engine = self._create_anomalib_engine(**kwargs)
        engine.export(model=self.model, export_type=ExportType.OPENVINO, datamodule=self.data)

    
    def _create_anomalib_engine(self, **kwargs) -> _AnomalibEngine:
        """Create an Anomalib engine.

        Args:
            **kwargs: Additional keyword arguments for the Anomalib engine.

        Returns:
            AnomalibEngine: The created Anomalib engine.
        """
        return _AnomalibEngine(**kwargs)

    @staticmethod
    def is_supported(model: AnomalyModule, data: AnomalibDataModule) -> bool:
        """Check if the engine is supported for the given model and data."""
        return isinstance(model, AnomalyModule) and isinstance(data, AnomalibDataModule)
