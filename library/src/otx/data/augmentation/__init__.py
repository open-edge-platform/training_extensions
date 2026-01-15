# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX Augmentation module.

This module provides a two-stage augmentation pipeline:
- CPU stage: Size-dependent augmentations in Dataset workers (torchvision.transforms.v2)
- GPU stage: Batch-level augmentations via Lightning Callback (Kornia)
"""

from otx.data.augmentation.pipeline import (
    CPUAugmentationPipeline,
    GPUAugmentationPipeline,
)

__all__ = [
    "CPUAugmentationPipeline",
    "GPUAugmentationPipeline",
]
