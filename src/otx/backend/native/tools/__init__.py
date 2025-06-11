"""Utilities for Native backend."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .adaptive_bs import adapt_batch_size
from .auto_configurator import AutoConfigurator

__all__ = ["AutoConfigurator", "adapt_batch_size"]
