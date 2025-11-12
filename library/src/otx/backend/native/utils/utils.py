# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

_T = TypeVar("_T")
_V = TypeVar("_V")


def is_ckpt_from_otx_v1(ckpt: dict) -> bool:
    """Check the checkpoint where it comes from.

    Args:
        ckpt (dict): the checkpoint file

    Returns:
        bool: True means the checkpoint comes from otx1
    """
    return "model" in ckpt and "VERSION" in ckpt and ckpt["VERSION"] == 1


def is_ckpt_for_finetuning(ckpt: dict) -> bool:
    """Check the checkpoint will be used to finetune.

    Args:
        ckpt (dict): the checkpoint file

    Returns:
        bool: True means the checkpoint will be used to finetune.
    """
    return "state_dict" in ckpt


def remove_state_dict_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Remove prefix from state_dict keys."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(prefix, "")
        new_state_dict[new_key] = value
    return new_state_dict


def ensure_callable(func: Callable[[_T], _V]) -> Callable[[_T], _V]:
    """If the given input is not callable, raise TypeError."""
    if not callable(func):
        raise TypeError(func)
    return func


@contextmanager
def mock_modules_for_chkpt() -> Iterator[None]:
    """Context manager to mock modules for OTX v2.2-2.4 checkpoint loading and restore sys.modules after."""
    import sys
    import types

    import otx
    from otx.types.label import AnomalyLabelInfo, HLabelInfo, LabelInfo, SegLabelInfo

    # Save original sys.modules
    original_sys_modules = dict(sys.modules)

    try:
        # Fake modules
        OTXTrainType = type("OTXTrainType", (object,), {"__init__": lambda *_: None})  # noqa: N806
        UnlabeledDataConfig = type("UnlabeledDataConfig", (object,), {"__init__": lambda *_: None})  # noqa: N806
        VisualPromptingConfig = type("VisualPromptingConfig", (object,), {"__init__": lambda *_: None})  # noqa: N806

        # Register all missing modules in sys.modules
        setattr(sys.modules["otx.config.data"], "UnlabeledDataConfig", UnlabeledDataConfig)  # noqa: B010
        setattr(sys.modules["otx.config.data"], "VisualPromptingConfig", VisualPromptingConfig)  # noqa: B010
        setattr(sys.modules["otx.types.label"], "LabelInfo", LabelInfo)  # noqa: B010
        setattr(sys.modules["otx.types.label"], "HLabelInfo", HLabelInfo)  # noqa: B010
        setattr(sys.modules["otx.types.label"], "SegLabelInfo", SegLabelInfo)  # noqa: B010
        setattr(sys.modules["otx.types.label"], "AnomalyLabelInfo", AnomalyLabelInfo)  # noqa: B010
        setattr(sys.modules["otx.types.task"], "OTXTrainType", OTXTrainType)  # noqa: B010

        sys.modules["otx.core"] = types.ModuleType("otx.core")
        sys.modules["otx.core.config"] = otx.config
        sys.modules["otx.core.config.data"] = otx.config.data
        sys.modules["otx.core.types"] = otx.types
        sys.modules["otx.core.types.task"] = otx.types.task
        sys.modules["otx.core.types.label"] = otx.types.label
        sys.modules["otx.core.model"] = otx.backend.native.models  # type: ignore[attr-defined]
        sys.modules["otx.core.metrics"] = otx.metrics

        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_sys_modules)
