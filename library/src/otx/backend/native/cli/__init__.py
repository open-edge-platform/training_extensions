"""CLI for Native backend.

Note: This is temporary as the new CLI should cover all the utilities mentioned here.
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .utils import get_otx_root_path, list_models

__all__ = ["list_models", "get_otx_root_path"]
