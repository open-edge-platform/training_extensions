# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX custom models."""

from .anomaly import Padim, Stfpm, Uflow
from .classification import (
    EfficientNetHLabelCls,
    EfficientNetMulticlassCls,
    EfficientNetMultilabelCls,
    MobileNetV3HLabelCls,
    MobileNetV3MulticlassCls,
    MobileNetV3MultilabelCls,
    TimmModelHLabelCls,
    TimmModelMulticlassCls,
    TimmModelMultilabelCls,
    TVModelHLabelCls,
    TVModelMulticlassCls,
    TVModelMultilabelCls,
    VisionTransformerHLabelCls,
    VisionTransformerMulticlassCls,
    VisionTransformerMultilabelCls,
)
from .detection import ATSS, RTDETR, SSD, DFine, RTMDet
from .instance_segmentation import MaskRCNN, MaskRCNNTV, RTMDetInst
from .keypoint_detection import RTMPose
from .segmentation import DinoV2Seg, LiteHRNet, SegNext

__all__ = [
    "Padim",
    "Stfpm",
    "Uflow",
    "EfficientNetHLabelCls",
    "EfficientNetMulticlassCls",
    "EfficientNetMultilabelCls",
    "MobileNetV3HLabelCls",
    "MobileNetV3MulticlassCls",
    "MobileNetV3MultilabelCls",
    "TimmModelHLabelCls",
    "TimmModelMulticlassCls",
    "TimmModelMultilabelCls",
    "TVModelHLabelCls",
    "TVModelMulticlassCls",
    "TVModelMultilabelCls",
    "VisionTransformerHLabelCls",
    "VisionTransformerMulticlassCls",
    "VisionTransformerMultilabelCls",
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
