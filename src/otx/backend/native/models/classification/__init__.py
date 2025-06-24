# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX classification models."""

from .hlabel_models import (
    EfficientNetHLabelCls,
    MobileNetV3HLabelCls,
    TimmModelHLabelCls,
    TVModelHLabelCls,
    VisionTransformerHLabelCls,
)
from .multiclass_models import (
    EfficientNetMulticlassCls,
    MobileNetV3MulticlassCls,
    TimmModelMulticlassCls,
    TVModelMulticlassCls,
    VisionTransformerMulticlassCls,
)
from .multilabel_models import (
    EfficientNetMultilabelCls,
    MobileNetV3MultilabelCls,
    TimmModelMultilabelCls,
    TVModelMultilabelCls,
    VisionTransformerMultilabelCls,
)

__all__ = [
    "EfficientNetMulticlassCls",
    "TimmModelMulticlassCls",
    "MobileNetV3MulticlassCls",
    "TVModelMulticlassCls",
    "VisionTransformerMulticlassCls",
    "EfficientNetHLabelCls",
    "TimmModelHLabelCls",
    "MobileNetV3HLabelCls",
    "TVModelHLabelCls",
    "VisionTransformerHLabelCls",
    "EfficientNetMultilabelCls",
    "TimmModelMultilabelCls",
    "MobileNetV3MultilabelCls",
    "TVModelMultilabelCls",
    "VisionTransformerMultilabelCls",
]
