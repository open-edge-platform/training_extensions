# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom model implementations for detection task."""

from .atss import ATSS
from .d_fine import DFine
from .deim import DEIMDFine
from .rtdetr import RTDETR
from .rtmdet import RTMDet
from .ssd import SSD
from .yolox import YOLOX

__all__ = ["SSD", "YOLOX", "ATSS", "RTDETR", "RTMDet", "DFine", "DEIMDFine"]
