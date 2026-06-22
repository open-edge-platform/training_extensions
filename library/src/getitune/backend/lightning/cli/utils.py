# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatibility shim — the canonical location is ``getitune.utils.recipes``."""

from getitune.utils.recipes import RECIPE_PATH, get_getitune_root_path, list_models

__all__ = ["RECIPE_PATH", "get_getitune_root_path", "list_models"]
