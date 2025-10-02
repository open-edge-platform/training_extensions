# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""API for OTX Entry-Point User."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .engine import Engine

if TYPE_CHECKING:
    from otx.types.types import DATA, MODEL


def create_engine(model: MODEL, data: DATA, **kwargs) -> Engine:
    """Create an engine.

    Args:
        model: The model to use
        data: The data/datamodule to use
        kwargs: Additional keyword arguments for engine initialization

    Returns:
        An instance of an Engine subclass that supports the model and data

    Raises:
        ValueError: If no compatible engine is found
    """
    from otx.backend.native.engine import OTXEngine
    from otx.backend.openvino.engine import OVEngine

    supported_engines = [OTXEngine, OVEngine]
    # Dynamically discover all custom subclasses of Engine
    for child_engines in Engine.__subclasses__():
        if child_engines not in supported_engines:
            supported_engines.append(child_engines)

    for engine_cls in supported_engines:
        if not hasattr(engine_cls, "is_supported"):
            msg = f"Engine {engine_cls.__name__} does not implement is_supported method."
            raise ValueError(msg)
        if engine_cls.is_supported(model, data):
            # Type ignore since mypy can't verify the constructor signature of subclasses
            return engine_cls(model=model, data=data, **kwargs)  # type: ignore[call-arg]

    msg = f"No engine found for model {model} and data {data}"
    raise ValueError(msg)


__all__ = ["Engine", "create_engine"]
