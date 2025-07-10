# mypy: disable_error_code=misc

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX classification factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from otx.backend.native.models.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.metrics.accuracy import MultiClassClsMetricCallable

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

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.backend.native.models.base import DataInputParams
    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class MobileNetV3:
    """Factory class for MobileNetV3 models.

    Args:
        label_info (LabelInfoTypes): The label information.
        data_input_params (DataInputParams): The data input parameters such as input size and normalization.
        freeze_backbone (bool, optional): Whether to freeze the backbone during training. Defaults to False.
            Note: only multiclass classification supports this argument.
        model_name (str, optional): The model name. Defaults to "mobilenetv3_large".
        task (Literal["multi_class", "multi_label", "h_label"], optional): The task type.
            Can be "multi_class", "multi_label", or "h_label". Defaults to "multi_class".
        optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
    """

    @overload
    def __new__(
        cls,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        task: Literal["multi_class", "multi_label", "h_label"] = "multi_class",
        freeze_backbone: bool = False,
        model_name: str = "mobilenetv3_large",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> MobileNetV3MulticlassCls | MobileNetV3MultilabelCls | MobileNetV3HLabelCls:
        ...

    def __new__(
        cls,
        task: Literal["multi_class", "multi_label", "h_label"] = "multi_class",
        **kwargs,
    ) -> MobileNetV3MulticlassCls | MobileNetV3MultilabelCls | MobileNetV3HLabelCls:
        """Factory method to create MobileNetV3 models based on the task type."""
        if task == "multi_class":
            return MobileNetV3MulticlassCls(**kwargs)
        if task == "multi_label":
            return MobileNetV3MultilabelCls(**kwargs)
        if task == "h_label":
            return MobileNetV3HLabelCls(**kwargs)
        msg = f"Unsupported task type: {task}"
        raise ValueError(msg)


class EfficientNet:
    """Factory class for EfficientNet models."""

    @overload
    def __new__(
        cls,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        task: Literal["multi_class", "multi_label", "h_label"] = "multi_class",
        model_name: str = "efficientnet_b0",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> EfficientNetMulticlassCls | EfficientNetMultilabelCls | EfficientNetHLabelCls:
        ...

    def __new__(
        cls,
        task: Literal["multi_class", "multi_label", "h_label"] = "multi_class",
        **kwargs,
    ) -> EfficientNetMulticlassCls | EfficientNetMultilabelCls | EfficientNetHLabelCls:
        """Factory method to create EfficientNet models based on the task type.

        Args:
            label_info (LabelInfoTypes): The label information.
            data_input_params (DataInputParams): The data input parameters such as input size and normalization.
            freeze_backbone (bool, optional): Whether to freeze the backbone during training. Defaults to False.
                Note: only multiclass classification supports this argument.
            model_name (str, optional): The model name. Defaults to "efficientnet_b0".
            task (Literal["multi_class", "multi_label", "h_label"], optional): The task type.
                Can be "multi_class", "multi_label", or "h_label". Defaults to "multi_class".
            optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
            torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        """
        if task == "multi_class":
            return EfficientNetMulticlassCls(**kwargs)
        if task == "multi_label":
            return EfficientNetMultilabelCls(**kwargs)
        if task == "h_label":
            return EfficientNetHLabelCls(**kwargs)
        msg = f"Unsupported task type: {task}"
        raise ValueError(msg)


class TimmModel:
    """Factory class for TimmModel models."""

    @overload
    def __new__(
        cls,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        task: Literal["multi_class", "multi_label", "h_label"] = "multi_class",
        model_name: str = "tf_efficientnetv2_s.in21k",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> TimmModelMulticlassCls | TimmModelMultilabelCls | TimmModelHLabelCls:
        ...

    def __new__(
        cls,
        task: Literal["multi_class", "multi_label", "h_label"] = "multi_class",
        **kwargs,
    ) -> TimmModelMulticlassCls | TimmModelMultilabelCls | TimmModelHLabelCls:
        """Factory method to create Timm models based on the task type.

        Args:
            label_info (LabelInfoTypes): The label information.
            data_input_params (DataInputParams): The data input parameters such as input size and normalization.
            freeze_backbone (bool, optional): Whether to freeze the backbone during training.
                Note: only multiclass classification supports this argument. Defaults to False.
            model_name (str, optional): The model name. Defaults to "tf_efficientnetv2_s.in21k".
            task (Literal["multi_class", "multi_label", "h_label"], optional): The task type.
                Can be "multi_class", "multi_label", or "h_label". Defaults to "multi_class".
            optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
            torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        """
        if task == "multi_class":
            return TimmModelMulticlassCls(**kwargs)
        if task == "multi_label":
            return TimmModelMultilabelCls(**kwargs)
        if task == "h_label":
            return TimmModelHLabelCls(**kwargs)
        msg = f"Unsupported task type: {task}"
        raise ValueError(msg)


class TVModel:
    """Factory class for Torch Vision models."""

    @overload
    def __new__(
        cls,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        task: Literal["multi_class", "multi_label", "h_label"] = "multi_class",
        model_name: str = "efficientnet_v2_s",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> TVModelMulticlassCls | TVModelMultilabelCls | TVModelHLabelCls:
        ...

    def __new__(
        cls,
        task: Literal["multi_class", "multi_label", "h_label"] = "multi_class",
        **kwargs,
    ) -> TVModelMulticlassCls | TVModelMultilabelCls | TVModelHLabelCls:
        """Factory to create TV models based on the task type.

        Args:
            label_info (LabelInfoTypes): The label information.
            data_input_params (DataInputParams): The data input parameters such as input size and normalization.
            freeze_backbone (bool, optional): Whether to freeze the backbone during training.
                Note: only multiclass classification supports this argument. Defaults to False.
            model_name (str, optional): The model name. Defaults to "efficientnet_v2_s".
            task (Literal["multi_class", "multi_label", "h_label"], optional): The task type.
                Can be "multi_class", "multi_label", or "h_label". Defaults to "multi_class".
            optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
            torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        """
        if task == "multi_class":
            return TVModelMulticlassCls(**kwargs)
        if task == "multi_label":
            return TVModelMultilabelCls(**kwargs)
        if task == "h_label":
            return TVModelHLabelCls(**kwargs)
        msg = f"Unsupported task type: {task}"
        raise ValueError(msg)


class VisionTransformer:
    """Factory class for VisionTransformer models."""

    @overload
    def __new__(
        cls,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        task: Literal["multi_class", "multi_label", "h_label"] = "multi_class",
        model_name: str = "vit-tiny",
        freeze_backbone: bool = False,
        lora: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> VisionTransformerMulticlassCls | VisionTransformerMultilabelCls | VisionTransformerHLabelCls:
        ...

    def __new__(
        cls,
        task: Literal["multi_class", "multi_label", "h_label"] = "multi_class",
        **kwargs,
    ) -> VisionTransformerMulticlassCls | VisionTransformerMultilabelCls | VisionTransformerHLabelCls:
        """Factory to create VisionTransformer models based on the task type.

        Args:
            label_info (LabelInfoTypes): The label information.
            data_input_params (DataInputParams): The data input parameters such as input size and normalization.
            freeze_backbone (bool, optional): Whether to freeze the backbone during training.
                Note: only multiclass classification supports this argument. Defaults to False.
            model_name (str, optional): The model name. Defaults to "vit-tiny".
            task (Literal["multi_class", "multi_label", "h_label"], optional): The task type.
                Can be "multi_class", "multi_label", or "h_label". Defaults to "multi_class".
            optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
            torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        """
        if task == "multi_class":
            return VisionTransformerMulticlassCls(**kwargs)
        if task == "multi_label":
            return VisionTransformerMultilabelCls(**kwargs)
        if task == "h_label":
            return VisionTransformerHLabelCls(**kwargs)
        msg = f"Unsupported task type: {task}"
        raise ValueError(msg)
