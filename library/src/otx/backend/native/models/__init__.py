# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX custom models."""

import multiprocessing

if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("forkserver")

from .anomaly import Padim, Stfpm, Uflow
from .classification import (
    EfficientNet,
    MobileNetV3,
    TimmModel,
    TVModel,
    VisionTransformer,
)
from .detection import ATSS, RTDETR, SSD, DFine, RTMDet
from .instance_segmentation import MaskRCNN, MaskRCNNTV, RTMDetInst
from .keypoint_detection import RTMPose
from .segmentation import DinoV2Seg, LiteHRNet, SegNext

__all__ = [
    "Padim",
    "Stfpm",
    "Uflow",
    "EfficientNet",
    "TimmModel",
    "MobileNetV3",
    "TVModel",
    "VisionTransformer",
    "ATSS",
    "DFine",
    "SSD",
    "RTMDet",
    "RTDETR",
    "MaskRCNN",
    "MaskRCNNTV",
    "RTMDetInst",
    "RTMPose",
    "DinoV2Seg",
    "LiteHRNet",
    "SegNext",
]
