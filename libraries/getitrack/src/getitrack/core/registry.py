# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Name-to-class registry for tracker algorithms.

Maps algorithm names to `BaseTracker` subclasses. Subclasses register
via `register_algorithm`; `BaseTracker.from_config` looks them up by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from getitrack.core.base import BaseTracker


ALGORITHM_REGISTRY: dict[str, type[BaseTracker]] = {}


def register_algorithm(name: str) -> Callable[[type[BaseTracker]], type[BaseTracker]]:
    """Return a decorator that registers a `BaseTracker` subclass under ``name``."""

    def _wrap(cls: type[BaseTracker]) -> type[BaseTracker]:
        if name in ALGORITHM_REGISTRY:
            msg = f"algorithm '{name}' is already registered by {ALGORITHM_REGISTRY[name].__name__}"
            raise ValueError(msg)
        ALGORITHM_REGISTRY[name] = cls
        return cls

    return _wrap
