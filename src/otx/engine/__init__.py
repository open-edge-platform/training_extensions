"""API for OTX Entry-Point User."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from otx.backend.native.engine import NativeEngine

from .engine import Engine

__all__ = ["Engine"]

if TYPE_CHECKING:
    from otx.types import DATA, MODEL

SUPPORTED_ENGINES = [NativeEngine]


def create_engine(model: "MODEL", data: "DATA") -> Engine:
    """Create an engine."""
    for engine in SUPPORTED_ENGINES:
        if engine.is_supported(model, data):
            return engine(model=model, data=data)
    msg = f"No engine found for model {model} and data {data}"
    raise ValueError(msg)
