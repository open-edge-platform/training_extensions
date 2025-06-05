# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom backbone implementations for detection task."""

from .csp_darknet import CSPDarknet
from .presnet import PResNet
from .cspnext import CSPNeXt

__all__ = ["CSPDarknet", "PResNet", "CSPNeXt"]
