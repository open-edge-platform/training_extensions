# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MobileNetV3 model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.classification.backbones import MobileNetV3Backbone
from getitune.backend.lightning.models.classification.classifier import ImageClassifier
from getitune.backend.lightning.models.classification.multiclass_models.base import LightningMulticlassClsModel
from getitune.backend.lightning.models.classification.necks.gap import GlobalAveragePooling
from getitune.backend.lightning.models.modules.base_module import BaseModule
from getitune.backend.lightning.models.utils.utils import load_checkpoint_to_model, load_from_http
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
from getitune.metrics.accuracy import MultiClassClsMetricCallable
from getitune.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.metrics import MetricCallable


# Pretrained URLs (torchvision ImageNet-1K weights hosted on Intel storage)
_PRETRAINED_URLS: dict[str, str] = {
    "mobilenetv3_large": "https://storage.geti.intel.com/weights/mobilenetv3-large-1cd25616.pth",
    "mobilenetv3_small": "https://storage.geti.intel.com/weights/mobilenetv3-small-55df8e1f.pth",
}


class MobileNetV3ClsHead(BaseModule):
    """MobileNetV3 classification head matching torchvision's architecture.

    Torchvision's MobileNetV3 uses a 2-layer classifier:
        Linear(in_channels, hidden_channels) -> Hardswish -> Dropout -> Linear(hidden_channels, num_classes)

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input feature channels (960 for large, 576 for small).
        hidden_channels: Hidden layer size (1280 for large, 1024 for small).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 960,
        hidden_channels: int = 1280,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, feats: tuple[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if isinstance(feats, tuple):
            feats = feats[-1]
        return self.classifier(feats)

    def predict(self, feats: tuple[torch.Tensor] | torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
        """Inference with softmax."""
        cls_score = self(feats)
        return self._get_predictions(cls_score)

    def _get_predictions(self, cls_score: torch.Tensor) -> torch.Tensor:
        """Get softmax predictions."""
        return functional.softmax(cls_score, dim=1)


class MobileNetV3MulticlassCls(LightningMulticlassClsModel):
    """MobileNetV3MulticlassCls is a class that represents a MobileNetV3 model for multiclass classification.

    Args:
        label_info (LabelInfoTypes): The label information.
        data_input_params (DataInputParams | dict | None, optional): The data input parameters
            such as input size and normalization. If None is given,
            default parameters for the specific model will be used.
        model_name (Literal["mobilenetv3_large", "mobilenetv3_small"], optional): The model name.
            Defaults to "mobilenetv3_large".
        optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal["mobilenetv3_large", "mobilenetv3_small"] = "mobilenetv3_large",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        num_classes = num_classes if num_classes is not None else self.num_classes
        # Create backbone WITHOUT pretrained loading (we load the full checkpoint below)
        backbone = MobileNetV3Backbone(
            mode=self.model_name,
            input_size=self.data_input_params.input_size,
            pretrained=False,
        )
        cfg = MobileNetV3Backbone.MV3_CFG[self.model_name]
        backbone_out_channels = cfg["out_channels"]
        hidden_channels = cfg["hid_channels"]
        neck = GlobalAveragePooling(dim=2)

        head = MobileNetV3ClsHead(
            num_classes=num_classes,
            in_channels=backbone_out_channels,
            hidden_channels=hidden_channels,
        )

        model = ImageClassifier(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=nn.CrossEntropyLoss(),
        )

        # Load full pretrained checkpoint (features + conv + classifier) into the model.
        # Checkpoint keys: features.* -> backbone.features.*, conv.* -> backbone.conv.*,
        # classifier.* -> head.classifier.*
        # Note: "features." must be first in key_mapping so nested .conv. keys inside
        # features (e.g., features.9.conv.8.weight) match "features." first.
        if self.model_name in _PRETRAINED_URLS:
            checkpoint = load_from_http(_PRETRAINED_URLS[self.model_name], map_location="cpu")
            load_checkpoint_to_model(
                model,
                checkpoint,
                key_mapping={
                    "features.": "backbone.features.",
                    "conv.": "backbone.conv.",
                    "classifier.": "head.classifier.",
                },
            )

        return model

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
