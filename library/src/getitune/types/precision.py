# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geti Tune precision type definition."""

from enum import Enum


class OTXPrecisionType(str, Enum):
    """Geti Tune precision type definition."""

    FP16 = "FP16"
    FP32 = "FP32"
