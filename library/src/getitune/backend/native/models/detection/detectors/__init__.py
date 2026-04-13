# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base models classes implementations for detection task."""

from .detection_transformer import DETR
from .rfdetr import RFDETRDetector
from .single_stage_detector import SingleStageDetector

__all__ = ["DETR", "RFDETRDetector", "SingleStageDetector"]
