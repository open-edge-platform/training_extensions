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
                return torch.tensor([_eval(val) for val in node.elts])
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
        if isinstance(ret, torch.Tensor):
            return tuple(ret.round().int().tolist())
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
    """GPU-stage augmentation pipeline using Kornia AugmentationSequential.

    This pipeline runs on GPU after batch transfer via Lightning Callback.
    It uses Kornia's AugmentationSequential for efficient batch-level processing
    with support for multiple data types (images, bboxes, masks, keypoints).

    Key features:
    - Uses Kornia AugmentationSequential for optimized batch processing
    - Supports data_keys for transforming bboxes, masks, keypoints along with images
    - Extracts normalization parameters for model export

    The pipeline expects batched tensors in BCHW format with values in [0, 1].

    Args:
        augmentations: List of Kornia augmentation modules.
        data_keys: List of data keys to transform (e.g., ["input", "bbox", "mask"]).
            Defaults to ["input"] for image-only augmentation.

    Example:
        >>> import kornia.augmentation as K
        >>> pipeline = GPUAugmentationPipeline(
        ...     augmentations=[
        ...         K.RandomHorizontalFlip(p=0.5),
        ...         K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
        ...         K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ...     ],
        ...     data_keys=["input"],
        ... )
        >>> augmented_images = pipeline(batch_images)
    """

    def __init__(
        self,
        augmentations: list[nn.Module] | None = None,
        data_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._augmentations_list = augmentations or []
        self._data_keys = data_keys or ["input"]
        self._mean, self._std = self._extract_normalization_params(self._augmentations_list)

        # Build Kornia AugmentationSequential for efficient batch processing
        if self._augmentations_list:
            import kornia.augmentation as K

            self.aug_sequential = K.AugmentationSequential(
                *self._augmentations_list,
                data_keys=self._data_keys,
                same_on_batch=False,
            )
        else:
            self.aug_sequential = None

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
            # Check if this is a Normalize augmentation (Kornia stores in flags dict)
            flags = getattr(aug, "flags", {})
            aug_mean = flags.get("mean") if isinstance(flags, dict) else None
            aug_std = flags.get("std") if isinstance(flags, dict) else None

            # Fallback to direct attributes if flags not present
            if aug_mean is None:
                aug_mean = getattr(aug, "mean", None)
            if aug_std is None:
                aug_std = getattr(aug, "std", None)

            if aug_mean is not None and aug_std is not None:
                # Extract mean value
                if hasattr(aug_mean, "tolist"):
                    mean = tuple(aug_mean.tolist())
                elif hasattr(aug_mean, "__iter__"):
                    mean = tuple(aug_mean)
                else:
                    mean = (float(aug_mean),) * 3

                # Extract std value
                if hasattr(aug_std, "tolist"):
                    std = tuple(aug_std.tolist())
                elif hasattr(aug_std, "__iter__"):
                    std = tuple(aug_std)
                else:
                    std = (float(aug_std),) * 3

                # Stop after finding the first Normalize
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

    @property
    def data_keys(self) -> list[str]:
        """Get data keys used by the pipeline."""
        return self._data_keys

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
    def from_config(
        cls,
        config: SubsetConfig,
        data_keys: list[str] | None = None,
    ) -> GPUAugmentationPipeline:
        """Build GPU augmentation pipeline from SubsetConfig.

        This function handles:
        - `augmentations_gpu` field with Kornia augmentations
        - Extraction of normalization parameters for model update
        - Input size placeholder replacement $(input_size)
        - data_keys for Kornia AugmentationSequential

        Args:
            config: SubsetConfig with augmentations_gpu.
            data_keys: List of data keys for AugmentationSequential.
                Defaults to ["input"] for image-only augmentation.

        Returns:
            GPUAugmentationPipeline ready for use in Callback.
        """
        input_size = getattr(config, "input_size", None)
        aug_configs = config.augmentations_gpu

        if not aug_configs:
            return cls([], data_keys=data_keys)

        augmentations = []

        for aug_config in aug_configs:
            cfg = copy(aug_config)
            if isinstance(cfg, (dict, DictConfig)):
                # Skip disabled transforms
                if not cfg.get("enable", True):
                    continue

                # Handle input_size placeholder
                cfg = CPUAugmentationPipeline._configure_input_size(dict(cfg), input_size)

                # Instantiate the transform
                transform = cls._dispatch_transform(cfg)
            elif isinstance(cfg, nn.Module):
                transform = cfg
            else:
                msg = f"Unsupported augmentation config type: {type(cfg)}"
                raise TypeError(msg)

            augmentations.append(transform)

        return cls(augmentations, data_keys=data_keys)

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

    def forward(
        self,
        images: torch.Tensor,
        labels: list[torch.Tensor] | None = None,
        bboxes: list[torch.Tensor] | None = None,
        masks: list[torch.Tensor] | None = None,
        keypoints: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        """Apply GPU augmentations to batched data using Kornia AugmentationSequential.


        Args:
            images: Batched images tensor in BCHW format, values in [0, 1].
            labels: List of labels per image (optional).
            bboxes: List of bounding boxes per image (optional).
            masks: List of masks per image (optional).
            keypoints: List of keypoints per image (optional).

        Returns:
            Dict with augmented data: {"images": tensor, "labels": list, "bboxes": list, "masks": list, "keypoints": list}
        """
        if self.aug_sequential is None:
            return {"images": images, "labels": labels, "bboxes": bboxes, "masks": masks, "keypoints": keypoints}

        # Build input list based on what data is provided and matches data_keys
        inputs = [images]
        provided_keys = ["input"]

        if labels is not None and "label" in self._data_keys:
            inputs.append(labels)
            provided_keys.append("label")
        if bboxes is not None and "bbox_xyxy" in self._data_keys:
            inputs.append(bboxes)
            provided_keys.append("bbox_xyxy")
        if masks is not None and "mask" in self._data_keys:
            inputs.append(masks)
            provided_keys.append("mask")
        if keypoints is not None and "keypoints" in self._data_keys:
            inputs.append(keypoints)
            provided_keys.append("keypoints")

        # Apply augmentation to all inputs
        results = self.aug_sequential(*inputs)

        # Parse results back
        output = {"images": None, "labels": labels, "bboxes": bboxes, "masks": masks, "keypoints": keypoints}
        if isinstance(results, (tuple, list)):
            for i, key in enumerate(provided_keys):
                if key == "input":
                    output["images"] = results[i]
                elif key == "label":
                    output["labels"] = results[i]
                elif key == "bbox_xyxy":
                    output["bboxes"] = results[i]
                elif key == "mask":
                    output["masks"] = results[i]
                elif key == "keypoints":
                    output["keypoints"] = results[i]
        else:
            # Single output (images only)
            output["images"] = results

        return output

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        if self.aug_sequential is not None:
            aug_str = str(self.aug_sequential)
        else:
            aug_str = "  (empty)"
        info = f"  mean={self._mean}, std={self._std}" if self._mean or self._std else ""
        return f"GPUAugmentationPipeline(\n{aug_str}\n  data_keys={self._data_keys}{info}\n)"

