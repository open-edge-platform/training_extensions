"""Engine base class."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from otx.types import ANNOTATIONS, DATA, METRICS, MODEL


class Engine(ABC):
    """Engine base class."""

    def __init__(self, model: MODEL, data: DATA, **kwargs):
        """Initialize the engine.
        
        This ensures that all the engines have the same interface.
        """
        raise NotImplementedError
    
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
