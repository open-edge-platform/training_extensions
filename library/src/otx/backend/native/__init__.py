# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native backend."""

import multiprocessing

from .lightning import accelerators, strategies

if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("forkserver")

__all__ = [
    "accelerators",
    "strategies",
]
