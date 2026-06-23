"""CLI for Native backend.

Note: This module is a backward-compatibility shim.
The canonical import path is ``getitune.utils``.
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from getitune.utils import get_getitune_root_path, list_models

__all__ = ["get_getitune_root_path", "list_models"]
