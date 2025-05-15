"""Anomalib engine."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from anomalib.data import AnomalibDataModule
from anomalib.deploy import ExportType
from anomalib.engine import Engine as _AnomalibEngine
from anomalib.models.components import AnomalyModule

from otx.data import OTXPredItem
from otx.engine import Engine
from otx.types import ANNOTATIONS, METRICS

if TYPE_CHECKING:
    from pathlib import Path


class AnomalyEngine(Engine):
    """Anomalib engine.

    Args:
        model: The model to train.
        data: The data to train on.
        work_dir: The directory to save the results.
        **kwargs: Additional keyword arguments for the Anomalib engine. These can also be arguments to the
             Lightning Trainer.

    Example:
        >>> from anomalib.models import Padim
        >>> from anomalib.data import MVTecAD
        >>> from otx.backend.anomalib.engine import AnomalyEngine
        >>> engine = AnomalyEngine(model=Padim(), data=MVTecAD())
        >>> engine.train()
    """

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

        Returns:
            METRICS: Dictionary of metrics.
        """
        engine = self._create_anomalib_engine(**kwargs)
        engine.train(model=self.model, datamodule=self.data)
        # TODO(ashwinvaidya17): return metrics
        return [{}]

    def test(self, **kwargs) -> METRICS:
        """Test the model.

        Returns:
            METRICS: Dictionary of metrics.
        """
        engine = self._create_anomalib_engine(**kwargs)
        results = engine.test(model=self.model, datamodule=self.data)
        return [{key: value} for key, value in results[0].items()]

    def predict(self, **kwargs) -> ANNOTATIONS:
        """Predict the model.

        Returns:
            ANNOTATIONS: List of OTXPredItem.
        """
        engine = self._create_anomalib_engine(**kwargs)
        results: list[dict] = engine.predict(model=self.model, datamodule=self.data)
        predictions = []
        for result in results:
            for image, pred_scores, anomaly_maps, label in zip(
                result["image"],
                result["pred_scores"].unsqueeze(0),
                result["anomaly_maps"],
                result["label"],
            ):
                predictions.append(OTXPredItem(image=image, scores=pred_scores, saliency_map=anomaly_maps, label=label))
        return predictions

    def export(self, **kwargs) -> Path:
        """Export the model.

        Returns:
            Path: The path to the exported model.
        """
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
