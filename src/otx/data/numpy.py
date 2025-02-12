"""Numpy data items."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from .base import DataItem


@dataclass
class NumpyDataItem(DataItem):
    """Numpy dataItem."""
