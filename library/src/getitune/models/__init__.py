# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reimport models from differnt backends for user frendly imports."""

from getitune.backend.lightning.models import (
    ATSS,
    DEIMV2,
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
    RTMDetInst,
    RTMPose,
    SegNext,
    TimmModel,
    TVModel,
    VisionTransformer,
)
from getitune.backend.openvino.models import (
    OVDetectionModel,
    OVHlabelClassificationModel,
    OVInstanceSegmentationModel,
    OVKeypointDetectionModel,
    OVModel,
    OVMulticlassClassificationModel,
    OVMultilabelClassificationModel,
    OVSegmentationModel,
)

try:
    from getitune.backend.ultralytics.models import (
        UltralyticsDetectionModel,
        UltralyticsInstSegModel,
    )
except ImportError:
    UltralyticsDetectionModel = None  # type: ignore[assignment]
    UltralyticsInstSegModel = None  # type: ignore[assignment]

__all__ = [
    # detection
    "ATSS",
    "DEIMV2",
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
    "RTMDetInst",
    "RTMPose",
    "SegNext",
    "TVModel",
    "TimmModel",
    "VisionTransformer",
]

if UltralyticsDetectionModel is not None:
    __all__.extend(
        [
            "UltralyticsDetectionModel",
            "UltralyticsInstSegModel",
        ]
    )
