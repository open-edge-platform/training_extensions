"""Anomalib engine."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from anomalib.data import AnomalibDataModule
from anomalib.deploy import ExportType
from anomalib.engine import Engine as _AnomalibEngine
from anomalib.models.components import AnomalyModule

from otx.engine import Engine
from otx.types import ANNOTATIONS, METRICS

if TYPE_CHECKING:
    from pathlib import Path


class AnomalyEngine(Engine):
    """Anomalib engine."""

    def __init__(
        self,
        model: AnomalyModule,
        data: AnomalibDataModule | Path | str,
        work_dir: Path | str = "results",
        **kwargs,
    ):
        self.model = model
        self.data = data
        self.work_dir = work_dir

    def train(self, **kwargs) -> METRICS:
        """Train the model.

        This runs fit followed by test.
        """
        engine = self._create_anomalib_engine(**kwargs)
        engine.train(model=self.model, datamodule=self.data)
        # TODO(ashwinvaidya17): return metrics
        return []

    def test(self, **kwargs) -> METRICS:
        """Test the model."""
        engine = self._create_anomalib_engine(**kwargs)
        return engine.test(model=self.model, datamodule=self.data)

    def predict(self, **kwargs) -> ANNOTATIONS:
        """Predict the model."""
        engine = self._create_anomalib_engine(**kwargs)
        return engine.predict(model=self.model, datamodule=self.data)

    def export(self, **kwargs) -> Path:
        """Export the model."""
        engine = self._create_anomalib_engine(**kwargs)
        return engine.export(model=self.model, export_type=ExportType.OPENVINO, datamodule=self.data)

    def _create_anomalib_engine(self, **kwargs) -> _AnomalibEngine:
        """Create an Anomalib engine.

        Args:
            **kwargs: Additional keyword arguments for the Anomalib engine.

        Returns:
            AnomalibEngine: The created Anomalib engine.
        """
        return _AnomalibEngine(**kwargs, default_root_dir=self.work_dir)

    @staticmethod
    def is_supported(model: AnomalyModule, data: AnomalibDataModule) -> bool:
        """Check if the engine is supported for the given model and data."""
        return isinstance(model, AnomalyModule) and isinstance(data, AnomalibDataModule)
