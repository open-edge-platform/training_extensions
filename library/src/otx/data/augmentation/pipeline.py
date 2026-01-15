# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU and GPU augmentation pipelines for OTX.

This module provides the core augmentation pipeline classes:
- CPUAugmentationPipeline: Runs in Dataset workers using torchvision.transforms.v2
- GPUAugmentationPipeline: Runs on GPU using Kornia (to be implemented)
"""

from __future__ import annotations

import ast
import operator
import typing
from copy import copy
from inspect import isclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import torchvision.transforms.v2 as tvt_v2
import typeguard
from lightning.pytorch.cli import instantiate_class
from omegaconf import DictConfig
from torch import nn

from otx.data.entity.sample import OTXSample
from otx.data.utils import import_object_from_module

if TYPE_CHECKING:
    from otx.config.data import SubsetConfig


class CPUAugmentationPipeline(nn.Module):
    """CPU-stage augmentation pipeline using torchvision.transforms.v2.

    This pipeline runs in Dataset workers (before collate) and handles:
    - Intensity mapping (uint16 → float32 for medical images)
    - Size-dependent geometric augmentations (Resize, RandomResizedCrop)
    - Augmentations applied to image, bboxes, masks, keypoints, etc.

    All outputs are fixed-size tensors suitable for batch stacking.

    The pipeline supports two types of transforms:
    1. OTX-style transforms: Have a `forward(*OTXDataItem)` signature and handle
       all data types internally (image, bboxes, masks, etc.)
    2. Native torchvision.v2 transforms: Applied to all tv_tensors extracted
       from the OTXDataItem (image, bboxes, masks, keypoints)

    Args:
        augmentations: List of torchvision.transforms.v2 transforms or OTX transforms.

    Example:
        >>> pipeline = CPUAugmentationPipeline([
        ...     v2.RandomResizedCrop(size=(224, 224)),
        ...     v2.RandomHorizontalFlip(p=0.5),
        ...     v2.ToDtype(torch.float32, scale=True),
        ... ])
        >>> item = pipeline(item)  # OTXDataItem with fixed-size image
    """

    def __init__(self, augmentations: list[nn.Module] | None = None) -> None:
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations or [])

    @classmethod
    def list_available_transforms(cls) -> list[type[tvt_v2.Transform]]:
        """List available TorchVision transform (only V2) classes."""
        return [
            obj
            for name in dir(tvt_v2)
            if (obj := getattr(tvt_v2, name)) and isclass(obj) and issubclass(obj, tvt_v2.Transform)
        ]

    @classmethod
    def from_config(cls, config: SubsetConfig) -> CPUAugmentationPipeline:
        """Build CPU augmentation pipeline from SubsetConfig.

        This function handles:
        - New `augmentations_cpu` field (preferred)
        - Input size placeholder replacement $(input_size)
        - Torch dtype string resolution

        Args:
            config: SubsetConfig with augmentations_cpu.

        Returns:
            CPUAugmentationPipeline ready for use in Dataset.
        """
        input_size = getattr(config, "input_size", None)
        aug_configs = config.augmentations_cpu

        if not aug_configs:
            return cls([])

        augmentations = []
        for aug_config in aug_configs:
            cfg = copy(aug_config)
            if isinstance(cfg, (dict, DictConfig)):
                # Skip disabled transforms
                if not cfg.get("enable", True):
                    continue

                # Handle input_size placeholder
                cfg = cls._configure_input_size(dict(cfg), input_size)

                # Instantiate the transform
                transform = cls._dispatch_transform(cfg)
            elif isinstance(cfg, nn.Module):
                transform = cfg
            else:
                msg = f"Unsupported augmentation config type: {type(cfg)}"
                raise TypeError(msg)

            augmentations.append(transform)

        return cls(augmentations)

    @classmethod
    def _dispatch_transform(cls, cfg_transform: DictConfig | dict | tvt_v2.Transform) -> tvt_v2.Transform:
        """Dispatch and instantiate a transform from config or return as-is.

        Args:
            cfg_transform: Transform config dict or already instantiated transform.

        Returns:
            Instantiated transform.
        """
        if isinstance(cfg_transform, (DictConfig, dict)):
            return instantiate_class(args=(), init=cfg_transform)

        if isinstance(cfg_transform, tvt_v2.Transform):
            return cfg_transform

        msg = (
            "CPUAugmentationPipeline accepts only three types "
            "for config.transforms: DictConfig | dict | tvt_v2.Transform. "
            f"However, its type is {type(cfg_transform)}."
        )
        raise TypeError(msg)

    @classmethod
    def _resolve_torch_dtypes(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        """Resolve torch dtype strings to actual torch.dtype objects.

        Handles strings like 'torch.float32', 'torch.uint8', etc. in init_args.

        Args:
            cfg: Augmentation config dict with class_path and init_args.

        Returns:
            Config with dtype strings resolved to torch.dtype objects.
        """
        init_args = cfg.get("init_args", {})
        if not init_args:
            return cfg

        dtype_map = {
            "torch.float32": torch.float32,
            "torch.float16": torch.float16,
            "torch.float64": torch.float64,
            "torch.bfloat16": torch.bfloat16,
            "torch.int8": torch.int8,
            "torch.int16": torch.int16,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "torch.uint8": torch.uint8,
            "torch.bool": torch.bool,
        }

        for key, val in init_args.items():
            if isinstance(val, str) and val in dtype_map:
                init_args[key] = dtype_map[val]

        return cfg

    @classmethod
    def _configure_input_size(
        cls,
        cfg: dict[str, Any],
        input_size: int | tuple[int, int] | None,
    ) -> dict[str, Any]:
        """Evaluate the input_size and replace the placeholder in the init_args.

        Input size should be specified as $(input_size). (e.g. $(input_size) * 0.5)
        Only simple multiplication or division evaluation is supported. For example,
        $(input_size) * -0.5    => supported
        $(input_size) * 2.1 / 3 => supported
        $(input_size) + 1       => not supported
        The function decides to pass tuple type or int type based on the type hint of the argument.
        float point values are rounded to int.

        Args:
            cfg: Augmentation config dict with class_path and init_args.
            input_size: Target input size (H, W) or single int.

        Returns:
            Config with placeholders replaced by actual values.
        """
        init_args = cfg.get("init_args", {})
        if not init_args:
            return cfg

        _input_size: tuple[int, int] | None = None
        if input_size is not None:
            _input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)  # type: ignore[assignment]

        def check_type(value: Any, expected_type: Any) -> bool:  # noqa: ANN401
            try:
                typeguard.check_type(value, expected_type)
            except typeguard.TypeCheckError:
                return False
            return True

        model_cls = None
        for key, val in init_args.items():
            if not (isinstance(val, str) and "$(input_size)" in val):
                continue

            if input_size is None:
                msg = (
                    f"{cfg['class_path'].split('.')[-1]} initial argument has `$(input_size)`, "
                    "but input_size is set to None."
                )
                raise RuntimeError(msg)

            if model_cls is None:
                model_cls = import_object_from_module(cfg["class_path"])

            available_types = typing.get_type_hints(model_cls.__init__).get(key)

            if available_types is None or check_type(_input_size, available_types):
                # Pass tuple[int, int]
                init_args[key] = cls._eval_input_size_str(val.replace("$(input_size)", str(_input_size)))
            elif check_type(_input_size[0], available_types):  # type: ignore[index]
                # Pass int
                init_args[key] = cls._eval_input_size_str(val.replace("$(input_size)", str(_input_size[0])))  # type: ignore[index]
            else:
                msg = f"{key} argument should be able to get int or tuple[int, int], but it can get {available_types}"
                raise RuntimeError(msg)

        return cfg

    @classmethod
    def _eval_input_size_str(cls, str_to_eval: str) -> tuple[int, ...] | int:
        """Safe eval function for _configure_input_size.

        The function is implemented for `_configure_input_size`, so implementation is aligned to it as below
        - Only multiplication or division evaluation are supported.
        - Only constant and tuple can be operand.
        - tuple is changed to numpy array before evaluation.
        - result value is rounded to int.

        Args:
            str_to_eval: String expression to evaluate.

        Returns:
            Evaluated result as int or tuple of ints.
        """
        bin_ops: dict[type, Callable[[Any, Any], Any]] = {
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }
        un_ops: dict[type, Callable[[Any], Any]] = {
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        available_ops = tuple(bin_ops) + tuple(un_ops) + (ast.BinOp, ast.UnaryOp)

        tree = ast.parse(str_to_eval, mode="eval")

        def _eval(node: Any) -> Any:  # noqa: ANN401
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Tuple):
                return np.array([_eval(val) for val in node.elts])
            if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
                left = _eval(node.left)
                right = _eval(node.right)
                return bin_ops[type(node.op)](left, right)
            if isinstance(node, ast.UnaryOp) and type(node.op) in un_ops:
                operand = _eval(node.operand) if isinstance(node.operand, available_ops) else node.operand.value
                return un_ops[type(node.op)](operand)
            msg = f"Bad syntax, {type(node)}. Available operations for calculating input size are {available_ops}"
            raise SyntaxError(msg)

        ret = _eval(tree)
        if isinstance(ret, np.ndarray):
            return tuple(ret.round().astype(np.int32).tolist())
        return round(ret)

    def _is_native_torchvision_transform(self, transform: tvt_v2.Transform) -> bool:
        """Check if the transform is a native torchvision transform."""
        module = type(transform).__module__
        return module.startswith("torchvision.")

    def _apply_native_transform(self, transform: tvt_v2.Transform, inputs: OTXSample) -> OTXSample:
        """Apply native torchvision transform only to image-related fields.

        TorchVision v2 expects standard field names like `boxes`/`labels`; we
        map to those before calling the transform and map back afterward.
        We also keep `img_info` in sync when the image size changes.
        """
        # Build a dict of transformable fields with torchvision-friendly keys
        transformable: dict[str, Any] = {}
        if (image := getattr(inputs, "image", None)) is not None:
            transformable["image"] = image
        if (masks := getattr(inputs, "masks", None)) is not None:
            transformable["masks"] = masks
        if (bboxes := getattr(inputs, "bboxes", None)) is not None:
            transformable["boxes"] = bboxes
        if (label := getattr(inputs, "label", None)) is not None:
            transformable["labels"] = label
        if (img_info := getattr(inputs, "img_info", None)) is not None:
            transformable["img_info"] = img_info

        if not transformable:
            return inputs

        # Apply transform to transformable fields
        # If there's only an image, pass it directly; otherwise pass as dict
        if len(transformable) == 1 and "image" in transformable:
            result = transform(transformable["image"])
            inputs.image = result
        else:
            result = transform(transformable)
            if isinstance(result, dict):
                for key, value in result.items():
                    if key == "boxes":
                        inputs.bboxes = value
                    elif key == "labels":
                        inputs.label = value
                    else:
                        setattr(inputs, key, value)
            else:
                # Single result, assume it's the image
                inputs.image = result
        return inputs

    def forward(self, *inputs: OTXSample) -> OTXSample | None:
        """Forward with skipping None."""
        needs_unpacking = len(inputs) > 1
        for transform in self.augmentations:
            if self._is_native_torchvision_transform(transform):
                # Apply native transforms only to image-related fields
                outputs = self._apply_native_transform(transform, inputs[0])
            else:
                outputs = transform(*inputs)
            if outputs is None:
                return outputs
            inputs = outputs if needs_unpacking else (outputs,)  # type: ignore[assignment]
        return outputs

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        aug_strs = [f"  {aug}" for aug in self.augmentations]
        return "CPUAugmentationPipeline(\n" + "\n".join(aug_strs) + "\n)"


class GPUAugmentationPipeline(nn.Module):
    """GPU-stage augmentation pipeline using Kornia.

    This pipeline runs on GPU after batch transfer via Lightning Callback.
    It handles augmentations that benefit from GPU acceleration:
    - Color augmentations (ColorJiggle, RandomGrayscale)
    - Geometric augmentations (RandomHorizontalFlip, RandomRotation)
    - Normalization (applied last)

    The pipeline expects batched tensors in BCHW format with values in [0, 1].

    Args:
        augmentations: List of Kornia augmentation modules.
        mean: Normalization mean (extracted from Normalize if present).
        std: Normalization std (extracted from Normalize if present).

    Example:
        >>> import kornia.augmentation as K
        >>> pipeline = GPUAugmentationPipeline([
        ...     K.RandomHorizontalFlip(p=0.5),
        ...     K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
        ...     K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ... ])
        >>> batch = pipeline(batch)  # BCHW tensor
    """

    def __init__(
        self,
        augmentations: list[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations or [])
        self._mean, self._std = self._extract_normalization_params(self.augmentations)

    @staticmethod
    def _extract_normalization_params(
        augmentations: list[nn.Module],
    ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        """Extract mean and std from Normalize augmentation in the list.

        Args:
            augmentations: List of augmentation modules.

        Returns:
            Tuple of (mean, std) extracted from first Normalize found, or (None, None).
        """
        mean: tuple[float, float, float] | None = None
        std: tuple[float, float, float] | None = None

        for aug in augmentations:
            # Check if this is a Normalize augmentation
            if hasattr(aug, "mean") and hasattr(aug, "std"):
                # Extract mean from the Normalize module
                if aug.mean is not None:
                    mean_val = aug.mean
                    if hasattr(mean_val, "tolist"):
                        mean = tuple(mean_val.tolist())
                    elif hasattr(mean_val, "__iter__"):
                        mean = tuple(mean_val)
                    else:
                        mean = (float(mean_val),) * 3

                # Extract std from the Normalize module
                if aug.std is not None:
                    std_val = aug.std
                    if hasattr(std_val, "tolist"):
                        std = tuple(std_val.tolist())
                    elif hasattr(std_val, "__iter__"):
                        std = tuple(std_val)
                    else:
                        std = (float(std_val),) * 3

                # Stop after finding the first Normalize
                if mean is not None and std is not None:
                    break

        return mean, std

    @property
    def mean(self) -> tuple[float, float, float] | None:
        """Get normalization mean."""
        return self._mean

    @property
    def std(self) -> tuple[float, float, float] | None:
        """Get normalization std."""
        return self._std

    @classmethod
    def list_available_transforms(cls) -> list[type]:
        """List available Kornia augmentation classes."""
        try:
            import kornia.augmentation as K

            return [
                obj
                for name in dir(K)
                if (obj := getattr(K, name))
                and isclass(obj)
                and issubclass(obj, (K.AugmentationBase2D, K.IntensityAugmentationBase2D))
            ]
        except ImportError:
            return []

    @classmethod
    def from_config(cls, config: SubsetConfig) -> GPUAugmentationPipeline:
        """Build GPU augmentation pipeline from SubsetConfig.

        This function handles:
        - `augmentations_gpu` field with Kornia augmentations
        - Extraction of normalization parameters for model update
        - Input size placeholder replacement $(input_size)

        Args:
            config: SubsetConfig with augmentations_gpu.

        Returns:
            GPUAugmentationPipeline ready for use in Callback.
        """
        input_size = getattr(config, "input_size", None)
        aug_configs = config.augmentations_gpu

        if not aug_configs:
            return cls([])

        augmentations = []

        for aug_config in aug_configs:
            cfg = copy(aug_config)
            if isinstance(cfg, (dict, DictConfig)):
                # Skip disabled transforms
                if not cfg.get("enable", True):
                    continue

                # Handle input_size placeholder
                cfg = CPUAugmentationPipeline._configure_input_size(dict(cfg), input_size)

                # Extract normalization parameters before instantiation
                class_path = cfg.get("class_path", "")
                if "Normalize" in class_path:
                    init_args = cfg.get("init_args", {})
                    raw_mean = init_args.get("mean")
                    raw_std = init_args.get("std")
                    if raw_mean is not None:
                        tuple(raw_mean) if hasattr(raw_mean, "__iter__") else (raw_mean,) * 3  # type: ignore[assignment]
                    if raw_std is not None:
                        tuple(raw_std) if hasattr(raw_std, "__iter__") else (raw_std,) * 3  # type: ignore[assignment]

                # Instantiate the transform
                transform = cls._dispatch_transform(cfg)
            elif isinstance(cfg, nn.Module):
                transform = cfg
                # Try to extract mean/std from already instantiated Normalize
                if hasattr(transform, "mean") and hasattr(transform, "std"):
                    tuple(transform.mean.tolist()) if hasattr(transform.mean, "tolist") else tuple(transform.mean)  # type: ignore[assignment]
                    tuple(transform.std.tolist()) if hasattr(transform.std, "tolist") else tuple(transform.std)  # type: ignore[assignment]
            else:
                msg = f"Unsupported augmentation config type: {type(cfg)}"
                raise TypeError(msg)

            augmentations.append(transform)

        return cls(augmentations)

    @classmethod
    def _dispatch_transform(cls, cfg_transform: DictConfig | dict | nn.Module) -> nn.Module:
        """Dispatch and instantiate a transform from config or return as-is.

        Args:
            cfg_transform: Transform config dict or already instantiated transform.

        Returns:
            Instantiated transform.
        """
        if isinstance(cfg_transform, (DictConfig, dict)):
            return instantiate_class(args=(), init=cfg_transform)

        if isinstance(cfg_transform, nn.Module):
            return cfg_transform

        msg = (
            "GPUAugmentationPipeline accepts only three types: "
            f"DictConfig | dict | nn.Module. However, its type is {type(cfg_transform)}."
        )
        raise TypeError(msg)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply GPU augmentations to batched images.

        Args:
            images: Batched images tensor in BCHW format, values in [0, 1].

        Returns:
            Augmented images tensor.
        """
        for transform in self.augmentations:
            images = transform(images)
        return images

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        aug_strs = [f"  {aug}" for aug in self.augmentations]
        info = f"  mean={self._mean}, std={self._std}" if self._mean or self._std else ""
        return "GPUAugmentationPipeline(\n" + "\n".join(aug_strs) + (f"\n{info}" if info else "") + "\n)"
