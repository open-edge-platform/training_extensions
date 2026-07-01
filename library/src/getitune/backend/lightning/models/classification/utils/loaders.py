# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pretrained-weight loader mixins for classification models.

Each mixin implements ``load_pretrained`` for one download backend and operates on ``self.model.backbone``.
Mix into a model *before* the task base class so the mixin's ``load_pretrained`` overrides the base no-op.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

from torch.hub import download_url_to_file

from getitune.backend.lightning.models.utils.utils import (
    load_checkpoint,
)

if TYPE_CHECKING:
    from torch import nn

    from getitune.backend.lightning.models.classification.backbones.vision_transformer import (
        VisionTransformerBackbone,
    )
    from getitune.types import PathLike

logger = logging.getLogger(__name__)


class _ClassifierModel(Protocol):
    """A classifier exposing a ``backbone`` submodule."""

    backbone: nn.Module


class _HasBackboneModel(Protocol):
    model: _ClassifierModel
    model_name: str


class _HasBackboneModelWithUrls(Protocol):
    pretrained_urls: dict[str, str]
    model: _ClassifierModel
    model_name: str


class PytorchcvLoaderMixin:
    """Load backbone weights via pytorchcv's model store (EfficientNet)."""

    def load_pretrained(self: _HasBackboneModel, weights: PathLike | None = None) -> None:
        """Download EfficientNet backbone weights into the cache dir."""
        from pytorchcv.models.common.model_store import download_model

        cache_dir = str(Path(weights).parent) if weights is not None else os.environ["PRETRAINED_WEIGHTS_CACHE_DIR"]
        download_model(
            net=self.model.backbone,
            model_name=self.model_name,
            local_model_store_dir_path=cache_dir,
        )
        logger.info("Loaded backbone weights from %s", cache_dir)


class CheckpointLoaderMixin:
    """Load backbone weights from an HTTP checkpoint URL (MobileNetV3)."""

    def load_pretrained(self: _HasBackboneModelWithUrls, weights: PathLike | None = None) -> None:
        """Download an HTTP checkpoint and load it into the backbone."""
        if weights is None:
            weights = self.pretrained_urls[self.model_name]
        load_checkpoint(self.model.backbone, str(weights))
        logger.info("Loaded backbone weights from %s", weights)


class TorchvisionLoaderMixin:
    """Load backbone weights from Torchvision (EfficientNet)."""

    def load_pretrained(self: _HasBackboneModel, weights: PathLike | None = None) -> None:
        """Load weights: a local checkpoint if given, else torchvision's official set."""
        if weights is not None and Path(weights).exists():
            load_checkpoint(self.model.backbone, str(weights))
            return

        from torchvision.models import get_model, get_model_weights

        ref = get_model(name=self.model_name, weights=get_model_weights(self.model_name))
        self.model.backbone.features.load_state_dict(ref.features.state_dict())  # pyrefly: ignore[missing-attribute]


class TimmLoaderMixin:
    """Load backbone weights via ``timm.models.load_pretrained``."""

    def load_pretrained(self: _HasBackboneModel, weights: PathLike | None = None) -> None:
        """Load weights: a local checkpoint if given, else timm's pretrained source."""
        if weights is not None and Path(weights).exists():
            load_checkpoint(self.model.backbone, str(weights))
            return

        from timm.models import load_pretrained

        timm_model = self.model.backbone.model  # the nn.Module created by timm.create_model
        load_pretrained(
            timm_model,  # pyrefly: ignore[bad-argument-type]
            pretrained_cfg=timm_model.pretrained_cfg,  # pyrefly: ignore[missing-attribute]
        )
        logger.info("Loaded timm pretrained weights for %s", self.model_name)


class _ViTClassifierModel(Protocol):
    backbone: VisionTransformerBackbone


class _HasViTBackboneWithUrls(Protocol):
    pretrained_urls: dict[str, str]
    model: _ViTClassifierModel
    model_name: str


class VisionTransformerLoaderMixin:
    """Load backbone weights for ViT architecture."""

    def load_pretrained(self: _HasViTBackboneWithUrls, weights: PathLike | None = None) -> None:
        """Load weights: a local checkpoint if given, else torchvision's official set."""
        if weights is not None and Path(weights).exists():
            self.model.backbone.load_checkpoint(checkpoint_path=Path(weights))  # pyrefly: ignore[not-callable]
        elif self.model_name in self.pretrained_urls:
            pretrained_url = self.pretrained_urls[self.model_name]
            logger.info("init weight - %s", pretrained_url)
            parts = urlparse(pretrained_url)
            filename = Path(parts.path).name

            cache_dir = Path(os.environ["PRETRAINED_WEIGHTS_CACHE_DIR"])
            cache_file = cache_dir / filename
            if not Path.exists(cache_file):
                download_url_to_file(pretrained_url, str(cache_file), "", progress=True)
            self.model.backbone.load_checkpoint(checkpoint_path=cache_file)  # pyrefly: ignore[not-callable]
        else:
            warnings.warn(
                "No pretrained weights found for the specified model. Initializing model with random weights.",
                stacklevel=1,
            )
