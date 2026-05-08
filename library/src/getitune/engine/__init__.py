# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""API for getitune Entry-Point User."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jsonargparse import ArgumentParser, Namespace

from .engine import Engine

if TYPE_CHECKING:
    from getitune.types.types import DATA, MODEL


def instantiate_model(model_cfg: dict[str, Any]) -> Any:
    """Instantiate a model from a ``class_path`` + ``init_args`` config dict.

    Determines the correct base class (Lightning or Ultralytics) from the
    ``class_path`` and uses *jsonargparse* for construction — identical to
    how Lightning models are built, extended to Ultralytics models.

    Args:
        model_cfg: Dict with ``class_path`` and ``init_args`` keys.

    Returns:
        Instantiated model (``LightningModel`` or ``UltralyticsModel``).

    Raises:
        ValueError: If ``class_path`` is missing or unresolvable.
    """
    class_path = model_cfg.get("class_path", "")
    if not class_path:
        msg = "model_cfg must contain a 'class_path' key"
        raise ValueError(msg)

    if "ultralytics" in class_path:
        from getitune.backend.ultralytics.models.base import UltralyticsModel

        base_cls: type = UltralyticsModel
    else:
        from getitune.backend.lightning.models.base import LightningModel

        base_cls = LightningModel

    parser = ArgumentParser()
    parser.add_subclass_arguments(base_cls, "model", required=False, fail_untyped=False)
    return parser.instantiate_classes(Namespace(model=model_cfg)).get("model")


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
    from getitune.backend.lightning.engine import LightningEngine
    from getitune.backend.openvino.engine import OVEngine

    supported_engines: list[type[Engine]] = [LightningEngine, OVEngine]

    # Ultralytics backend (optional)
    try:
        from getitune.backend.ultralytics.engine import UltralyticsEngine

        supported_engines.append(UltralyticsEngine)
    except ImportError:
        pass

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


__all__ = ["Engine", "create_engine", "instantiate_model"]
