# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX custom models."""

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
    "ATSS",
    "RTDETR",
    "SSD",
    "DFine",
    "DinoV2Seg",
    "EfficientNet",
    "LiteHRNet",
    "MaskRCNN",
    "MaskRCNNTV",
    "MobileNetV3",
    "Padim",
    "RTMDet",
    "RTMDetInst",
    "RTMPose",
    "SegNext",
    "Stfpm",
    "TVModel",
    "TimmModel",
    "Uflow",
    "VisionTransformer",
]
