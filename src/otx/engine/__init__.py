"""Engine API."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from .engine import Engine

if TYPE_CHECKING:
    from otx.types import DATA, MODEL


def create_engine(model: MODEL, data: DATA) -> Engine:
    """Create an engine.

    Args:
        model: The model to use
        data: The data/datamodule to use

    Returns:
        An instance of an Engine subclass that supports the model and data

    Raises:
        ValueError: If no compatible engine is found
    """
    # Get all concrete (non-abstract) subclasses of Engine
    engine_classes: list[type[Engine]] = Engine.__subclasses__()

    for engine_cls in engine_classes:
        if engine_cls.is_supported(model, data):
            # Type ignore since mypy can't verify the constructor signature of subclasses
            return engine_cls(model=model, datamodule=data)  # type: ignore[call-arg]

    msg = f"No engine found for model {model} and data {data}"
    raise ValueError(msg)
