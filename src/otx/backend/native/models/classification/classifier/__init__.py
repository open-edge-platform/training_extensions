# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Head modules for OTX custom model."""

from .base_classifier import ImageClassifier
from .h_label_classifier import HLabelClassifier

__all__ = ["ImageClassifier", "HLabelClassifier"]
