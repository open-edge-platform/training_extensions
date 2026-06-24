# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Name-to-class registry for tracker algorithms.

Maps algorithm names to `BaseTracker` subclasses and their config classes.
Subclasses register via `register_algorithm`; `BaseTracker.from_config` looks
up the tracker by name and `resolve_tracker_config` resolves raw data to the
matching config variant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from getitrack.config import TrackerConfig
    from getitrack.core.base import BaseTracker


ALGORITHM_REGISTRY: dict[str, type[BaseTracker]] = {}

_DEFAULT_ALGORITHM = "bytetrack"


def register_algorithm(
    name: str,
    config: type[TrackerConfig],
) -> Callable[[type[BaseTracker]], type[BaseTracker]]:
    """Return a decorator that registers a `BaseTracker` subclass under ``name``.

    ``config`` is the algorithm's `TrackerConfig` subclass; it is attached
    to the tracker as ``config_cls`` and used to validate raw configuration.
    """

    def _wrap(cls: type[BaseTracker]) -> type[BaseTracker]:
        if name in ALGORITHM_REGISTRY:
            msg = f"algorithm '{name}' is already registered by {ALGORITHM_REGISTRY[name].__name__}"
            raise ValueError(msg)
        declared = config.model_fields["algorithm"].default
        if declared != name:
            msg = f"config {config.__name__} pins algorithm={declared!r}, but is registered as '{name}'"
            raise ValueError(msg)
        cls.config_cls = config
        ALGORITHM_REGISTRY[name] = cls
        return cls

    return _wrap


def resolve_tracker_config(data: object) -> TrackerConfig:
    """Resolve a raw configuration mapping to the registered algorithm's config.

    Dispatches on the ``algorithm`` key (default ``bytetrack``) and validates
    the data against that algorithm's `config_cls`.
    """
    name = data.get("algorithm", _DEFAULT_ALGORITHM) if isinstance(data, dict) else _DEFAULT_ALGORITHM
    if name not in ALGORITHM_REGISTRY:
        known = sorted(ALGORITHM_REGISTRY) or ["<none registered>"]
        msg = f"unknown algorithm '{name}'; registered: {known}"
        raise KeyError(msg)
    return ALGORITHM_REGISTRY[name].config_cls.model_validate(data)
