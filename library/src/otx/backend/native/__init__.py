# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native backend."""

from .lightning import accelerators, strategies

__all__ = [
    "accelerators",
    "strategies",
]
