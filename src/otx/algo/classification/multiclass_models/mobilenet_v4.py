"""
MobileNetV4 Classification Model for OTX Framework

This module provides a specialized implementation of MobileNetV4 for multi-class image classification
using the OpenVINO Training Extensions (OTX) framework and TIMM backbone.
"""
from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Union

from otx.algo.classification.backbones.timm import TimmBackbone
from otx.algo.classification.classifier import ImageClassifier
from otx.algo.classification.heads import LinearClsHead
from otx.algo.classification.necks.gap import GlobalAveragePooling
from otx.core.model.multiclass_classification import OTXMulticlassClsModel
from otx.core.types.label import LabelInfoTypes
from otx.core.metrics.accuracy import MultiClassClsMetricCallable
from otx.core.model.base import (
    DataInputParams, 
    DefaultOptimizerCallable, 
    DefaultSchedulerCallable
)
class MobileNetV4MulticlassCls(OTXMulticlassClsModel):
    """
    MobileNetV4 Multi-class Classification Model
    Args:
        label_info (LabelInfoTypes): Label information for classification task
        data_input_params (DataInputParams): Input data parameters
        model_name (str, optional): Specific MobileNetV4 variant from TIMM
        variant (str, optional): Model variant like 'conv_small', 'conv_medium', 'conv_large'
        optimizer (OptimizerCallable, optional): Custom optimizer configuration
        scheduler (LRSchedulerCallable, optional): Custom learning rate scheduler
        metric (MetricCallable, optional): Custom evaluation metric
        torch_compile (bool, optional): Enable TorchScript compilation
    """
    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: str = 'mobilenetv4_conv_medium',
        variant: Optional[str] = None,
        optimizer: Optional[OptimizerCallable] = DefaultOptimizerCallable,
        scheduler: Optional[Union[LRSchedulerCallable, LRSchedulerListCallable]] = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        # Use variant if provided, otherwise extract from model_name
        self.variant = variant or model_name.split('_')[-1]
        
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self, num_classes: Optional[int] = None) -> nn.Module:
        """
        Creating the MobileNetV4 model architecture
        Constructs the model using:
        - TimmBackbone for feature extraction
        - Global Average Pooling as neck
        - Linear Classification Head
        Args:
            num_classes (Optional[int]): Number of output classes
        Returns:
            nn.Module: Configured classification model
        """
        num_classes = num_classes or self.num_classes
        # Creating backbone with dynamic variant selection
        backbone = TimmBackbone(
            model_name=f'mobilenetv4_{self.variant}',
            pretrained=True
        )
        return ImageClassifier(
            backbone=backbone,
            neck=GlobalAveragePooling(dim=2),
            head=LinearClsHead(
                num_classes=num_classes,
                in_channels=backbone.num_features,
                dropout_rate=0.2
            ),
            loss=nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing from recipe
        )
    def forward_for_tracing(self, image: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward method for model tracing during export
        Supports both regular inference and explanation modes
        Args:
            image (torch.Tensor): Input image tensor
        Returns:
            Inference result or explanation tensor
        """
        return self.model(images=image, mode="explain" if self.explain_mode else "tensor")
    @classmethod
    def variants(cls) -> list[str]:
        """
        List available MobileNetV4 variants
        Returns:
            list[str]: Supported model variants
        """
        return ['conv_small', 'conv_medium', 'conv_large', 'hybrid_medium', 'hybrid_large']
# Optional: Alias for easier instantiation
MobileNetV4Model = MobileNetV4MulticlassCls