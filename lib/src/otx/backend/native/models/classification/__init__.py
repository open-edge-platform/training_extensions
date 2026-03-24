# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX classification models."""

from .factory import (
    EfficientNet,
    MobileNetV3,
    ShuffleNetV2,
    TimmModel,
    TVModel,
    VisionTransformer,
)

__all__ = [
    "EfficientNet",
    "TimmModel",
    "MobileNetV3",
    "ShuffleNetV2",
    "TVModel",
    "VisionTransformer",
]
