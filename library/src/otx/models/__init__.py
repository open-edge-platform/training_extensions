# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reimport models from differnt backends for user frendly imports."""

from otx.backend.native.models import (
    ATSS,
    RTDETR,
    SSD,
    YOLOX,
    DEIMDFine,
    DFine,
    DinoV2Seg,
    EfficientNet,
    LiteHRNet,
    MaskRCNN,
    MaskRCNNTV,
    MobileNetV3,
    RTMDet,
    RTMDetInst,
    RTMPose,
    SegNext,
    TimmModel,
    TVModel,
    VisionTransformer,
)
from otx.backend.openvino.models import (
    OVDetectionModel,
    OVHlabelClassificationModel,
    OVInstanceSegmentationModel,
    OVKeypointDetectionModel,
    OVModel,
    OVMulticlassClassificationModel,
    OVMultilabelClassificationModel,
    OVSegmentationModel,
)

__all__ = [
    # detection
    "ATSS",
    "RTDETR",
    "SSD",
    "YOLOX",
    "DEIMDFine",
    "DFine",
    # semantic segmentation
    "DinoV2Seg",
    # classification
    "EfficientNet",
    "LiteHRNet",
    # instance segmentation
    "MaskRCNN",
    "MaskRCNNTV",
    "MobileNetV3",
    "OVDetectionModel",
    "OVHlabelClassificationModel",
    "OVInstanceSegmentationModel",
    "OVKeypointDetectionModel",
    # OpenVINO models
    "OVModel",
    "OVMulticlassClassificationModel",
    "OVMultilabelClassificationModel",
    "OVSegmentationModel",
    "RTMDet",
    "RTMDetInst",
    "RTMPose",
    "SegNext",
    "TVModel",
    "TimmModel",
    "VisionTransformer",
]
