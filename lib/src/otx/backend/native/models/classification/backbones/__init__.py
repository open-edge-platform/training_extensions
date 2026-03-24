# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backbone modules for OTX custom model."""

from .efficientnet import EfficientNetBackbone
from .mobilenet_v3 import MobileNetV3Backbone
from .shufflenet_v2 import ShuffleNetV2Backbone
from .timm import TimmBackbone
from .torchvision import TorchvisionBackbone
from .vision_transformer import VisionTransformerBackbone

__all__ = [
    "EfficientNetBackbone",
    "TimmBackbone",
    "MobileNetV3Backbone",
    "ShuffleNetV2Backbone",
    "VisionTransformerBackbone",
    "TorchvisionBackbone",
]
