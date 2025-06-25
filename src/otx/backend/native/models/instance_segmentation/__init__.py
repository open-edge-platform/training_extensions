# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX instance segmentation models."""

from .maskrcnn import MaskRCNN
from .maskrcnn_tv import MaskRCNNTV
from .rtmdet_inst import RTMDetInst

__all__ = ["MaskRCNNTV", "MaskRCNN", "RTMDetInst"]
