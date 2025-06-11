# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone modules for OTX segmentation model."""

from .litehrnet import LiteHRNetBackbone
from .mscan import MSCAN

__all__ = ["LiteHRNetBackbone", "MSCAN"]
