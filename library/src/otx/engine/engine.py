# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from otx.types import PathLike
    from otx.types.types import ANNOTATIONS, DATA, METRICS, MODEL


class Engine(ABC):
    """Engine base class."""

    @abstractmethod
    def train(self, **kwargs) -> METRICS:
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def test(self, **kwargs) -> METRICS:
        """Test the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs) -> ANNOTATIONS:
        """Predict on model."""
        raise NotImplementedError

    @abstractmethod
    def export(self, **kwargs) -> Path:
        """Export the model."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_supported(model: MODEL, data: DATA) -> bool:
        """Check if the engine is supported for the given model and data."""
        raise NotImplementedError

    @property
    @abstractmethod
    def work_dir(self) -> PathLike:
        """Get the working directory for the engine."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> MODEL:
        """Returns the model object associated with the engine.

        Returns:
            MODEL: model object.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def datamodule(self) -> DATA:
        """Returns the datamodule object associated with the engine.

        Returns:
            DATA: datamodule object.
        """
        raise NotImplementedError
