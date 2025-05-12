"""Anomalib engine."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.engine import Engine
from anomalib.models import AnomalibModel
from anomalib.data import AnomalibDataModule
from anomalib.engine import Engine as AnomalibEngine


class AnomalibEngine(Engine):
    """Anomalib engine."""

    def __init__(self, model: AnomalibModel, data: AnomalibDataModule, **kwargs):
        self.model = model
        self.data = data

    def _create_anomalib_engine(self, **kwargs) -> AnomalibEngine:
        """Create an Anomalib engine.

        Args:
            **kwargs: Additional keyword arguments for the Anomalib engine.

        Returns:
            AnomalibEngine: The created Anomalib engine.
        """
        return AnomalibEngine(
            model=self.model,
            data=self.data,
            **kwargs,
        )
