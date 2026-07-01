# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pretrained weight URLs for classification backbones.

Central registry of remote checkpoint URLs used to initialize classification
backbones with pretrained weights. Each mapping is keyed by ``model_name`` (the
identifier passed to a model/backbone) and resolves to a downloadable checkpoint
hosted on Intel's weight storage.

These constants are consumed by the classification loader mixins (see
:mod:`getitune.backend.lightning.models.classification.utils.loaders`), which
resolve the URL for a given ``model_name``, download the file into
``PRETRAINED_WEIGHTS_CACHE_DIR``, and load it into the backbone.

Attributes:
    VIT_PRETRAINED_URLS: ViT / DINOv2 ImageNet-21k checkpoints, shared across the
        multiclass, multilabel, and hierarchical-label tasks. ViT entries are
        JAX ``.npz`` weights; DINOv2 entries are PyTorch ``.pth`` weights.
    MOBILENETV3_PRETRAINED_URLS: MobileNetV3 ImageNet ``.pth`` checkpoints.

Note:
    URLs point to ``https://storage.geti.intel.com/weights/`` and require network
    access at load time unless the checkpoint is already cached.
"""

# ViT / DINOv2 ImageNet-21k checkpoints (shared by multiclass, multilabel, hlabel).
from __future__ import annotations

VIT_PRETRAINED_URLS: dict[str, str] = {
    "vit-tiny": (
        "https://storage.geti.intel.com/weights/"
        "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit-small": (
        "https://storage.geti.intel.com/weights/"
        "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit-base": (
        "https://storage.geti.intel.com/weights/"
        "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit-large": (
        "https://storage.geti.intel.com/weights/"
        "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "dinov2-small": "https://storage.geti.intel.com/weights/dinov2_vits14_reg4_pretrain.pth",
    "dinov2-base": "https://storage.geti.intel.com/weights/dinov2_vitb14_reg4_pretrain.pth",
    "dinov2-large": "https://storage.geti.intel.com/weights/dinov2_vitl14_reg4_pretrain.pth",
    "dinov2-giant": "https://storage.geti.intel.com/weights/dinov2_vitg14_reg4_pretrain.pth",
}

# MobileNetV3 ImageNet checkpoints.
MOBILENETV3_PRETRAINED_URLS: dict[str, str] = {
    "mobilenetv3_small": "https://storage.geti.intel.com/weights/mobilenetv3-small-55df8e1f.pth",
    "mobilenetv3_large": "https://storage.geti.intel.com/weights/mobilenetv3-large-1cd25616.pth",
    "mobilenetv3_large_075": "https://storage.geti.intel.com/weights/mobilenetv3-large-0.75-9632d2a8.pth",
}
