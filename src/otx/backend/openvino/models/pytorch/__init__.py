# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pytorch models with OpenVINO optimizations for OTX."""

from .base_wrapper import BaseWrapper
from .classification_wrapper import ClassificationWrapper
from .regression_wrapper import RegressionWrapper
from .segmentation_wrapper import SegmentationWrapper
from .detection_wrapper import DetectionWrapper
from .compiler import compile_model


__all__ = [
    "BaseWrapper",
    "ClassificationWrapper",
    "RegressionWrapper",
    "SegmentationWrapper",
    "DetectionWrapper",
    "compile_model",
]