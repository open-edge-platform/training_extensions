# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for the data module."""

from __future__ import annotations

import importlib
import inspect
import logging
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

import torch

from getitune.utils.device import is_xpu_available

if TYPE_CHECKING:
    from torch.utils.data import Dataset, Sampler

    from getitune.config.data import SamplerConfig


logger = logging.getLogger(__name__)


def instantiate_sampler(sampler_config: SamplerConfig, dataset: Dataset, **kwargs) -> Sampler:
    """Instantiate a sampler object based on the provided configuration.

    Args:
        sampler_config (SamplerConfig): The configuration object for the sampler.
        dataset (Dataset): The dataset object to be sampled.
        **kwargs: Additional keyword arguments to be passed to the sampler's constructor.

    Returns:
        Sampler: The instantiated sampler object.
    """
    class_module, class_name = sampler_config.class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    sampler_class = getattr(module, class_name)
    init_signature = list(inspect.signature(sampler_class.__init__).parameters.keys())
    if "batch_size" not in init_signature:
        kwargs.pop("batch_size", None)
    # Handle None init_args
    init_args = sampler_config.init_args or {}
    sampler_kwargs = {**init_args, **kwargs}
    return sampler_class(dataset, **sampler_kwargs)


def get_adaptive_num_workers(num_dataloader: int = 1) -> int | None:
    """Measure appropriate num_workers value and return it."""
    num_devices = torch.xpu.device_count() if is_xpu_available() else torch.cuda.device_count()
    if num_devices == 0:
        return None
    return min(cpu_count() // (num_dataloader * num_devices), 8)  # max available num_workers is 8


def import_object_from_module(obj_path: str) -> Any:  # noqa: ANN401
    """Get object from import format string."""
    module_name, obj_name = obj_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
