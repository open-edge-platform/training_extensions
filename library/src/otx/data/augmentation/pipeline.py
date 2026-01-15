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
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import typeguard
from lightning.pytorch.cli import instantiate_class
from omegaconf import DictConfig
from torch import nn
from torchvision import tv_tensors

from otx.data.entity.torch import OTXDataItem
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

    def forward(self, item: OTXDataItem) -> OTXDataItem:
        """Apply CPU augmentations to a single data item.

        Handles both OTX-style transforms (that expect OTXDataItem) and native
        torchvision.v2 transforms (that work on tv_tensors).

        For native torchvision transforms, we extract all applicable tv_tensors
        from the OTXDataItem (image, bboxes, masks, keypoints) and pass them
        together so geometric transforms are applied consistently.

        Args:
            item: OTXDataItem with image tensor and optional annotations.

        Returns:
            Augmented OTXDataItem with transformed image and annotations.
        """
        if not self.augmentations:
            return item

        for aug in self.augmentations:
            # Check if this is an OTX-style transform that handles OTXDataItem directly
            # OTX transforms have a forward() that expects *_inputs: OTXDataItem
            if self._is_otx_transform(aug):
                result = aug(item)
                if result is None:
                    # Some OTX transforms can return None (e.g., MinIoURandomCrop)
                    # In this case, skip remaining transforms
                    return item  # type: ignore[return-value]
                item = result
            else:
                # Native torchvision.v2 transform - apply to all tv_tensors
                item = self._apply_torchvision_transform(aug, item)

        return item

    def _is_otx_transform(self, transform: nn.Module) -> bool:
        """Check if transform is an OTX-style transform expecting OTXDataItem.

        OTX transforms are identified by:
        - Having 'NumpytoTVTensorMixin' in MRO (OTX custom transforms)
        - Being from otx.data.transform_libs module
        """
        # Check module path
        module = type(transform).__module__
        if module.startswith("otx.data.transform_libs"):
            return True

        # Check for NumpytoTVTensorMixin
        return any(base.__name__ == "NumpytoTVTensorMixin" for base in type(transform).__mro__)

    def _apply_torchvision_transform(self, aug: nn.Module, item: OTXDataItem) -> OTXDataItem:
        """Apply a native torchvision.v2 transform to OTXDataItem.

        Extracts tv_tensors from OTXDataItem, applies the transform to all of them
        together (so geometric transforms are consistent), then updates the item.

        Args:
            aug: A torchvision.transforms.v2 transform.
            item: OTXDataItem to transform.

        Returns:
            Updated OTXDataItem with transformed tensors.
        """
        # Convert image to tv_tensors.Image if not already
        image = item.image
        if not isinstance(image, tv_tensors.Image):
            image = tv_tensors.Image(image)

        # Collect all tv_tensors that should be transformed together
        inputs: list[Any] = [image]
        has_bboxes = hasattr(item, "bboxes") and item.bboxes is not None and len(item.bboxes) > 0
        has_masks = hasattr(item, "masks") and item.masks is not None and len(item.masks) > 0
        hasattr(item, "keypoints") and item.keypoints is not None

        if has_bboxes:
            bboxes = item.bboxes
            if not isinstance(bboxes, tv_tensors.BoundingBoxes):
                # Convert to tv_tensors.BoundingBoxes
                canvas_size = tuple(image.shape[-2:])
                bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=canvas_size)
            inputs.append(bboxes)

        if has_masks:
            masks = item.masks
            if not isinstance(masks, tv_tensors.Mask):
                masks = tv_tensors.Mask(masks)
            inputs.append(masks)

        # Apply transform
        # torchvision.v2 transforms can handle multiple inputs
        outputs = aug(*inputs)

        # Unpack outputs
        output_list = list(outputs) if isinstance(outputs, tuple) else [outputs]

        # Update item with transformed tensors
        idx = 0
        item.image = output_list[idx]
        idx += 1

        if has_bboxes and idx < len(output_list):
            item.bboxes = output_list[idx]
            idx += 1

        if has_masks and idx < len(output_list):
            item.masks = output_list[idx]
            idx += 1

        # Update image info if shape changed
        if hasattr(item, "img_info") and item.img_info is not None:
            # Ensure we produce a concrete tuple[int, int] for static type checkers
            h = int(item.image.shape[-2])
            w = int(item.image.shape[-1])
            new_shape: tuple[int, int] = (h, w)
            item.img_info.img_shape = new_shape

        return item

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        aug_strs = [f"  {aug}" for aug in self.augmentations]
        return "CPUAugmentationPipeline(\n" + "\n".join(aug_strs) + "\n)"


def build_cpu_augmentation_pipeline(
    config: SubsetConfig,
) -> CPUAugmentationPipeline:
    """Build CPU augmentation pipeline from SubsetConfig.

    This function handles:
    - Legacy `transforms` field (backward compatibility)
    - New `augmentations_cpu` field
    - Input size placeholder replacement $(input_size)

    Args:
        config: SubsetConfig with augmentations_cpu or transforms.

    Returns:
        CPUAugmentationPipeline ready for use in Dataset.
    """
    input_size = getattr(config, "input_size", None)

    # Determine which config field to use
    # Priority: augmentations_cpu > transforms (legacy)
    aug_configs = config.augmentations_cpu if config.augmentations_cpu else config.transforms

    if not aug_configs:
        return CPUAugmentationPipeline([])

    augmentations = []
    for aug_config in aug_configs:
        cfg = copy(aug_config)
        if isinstance(cfg, (dict, DictConfig)):
            # Skip disabled transforms
            if not cfg.get("enable", True):
                continue

            # Handle input_size placeholder
            cfg = _configure_input_size(dict(cfg), input_size)

            # Handle torch dtype strings (e.g., 'torch.float32' -> torch.float32)
            cfg = _resolve_torch_dtypes(cfg)

            # Instantiate the transform
            transform = instantiate_class(args=(), init=cfg)
        elif isinstance(cfg, nn.Module):
            transform = cfg
        else:
            msg = f"Unsupported augmentation config type: {type(cfg)}"
            raise TypeError(msg)

        augmentations.append(transform)

    return CPUAugmentationPipeline(augmentations)


def _resolve_torch_dtypes(cfg: dict[str, Any]) -> dict[str, Any]:
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


def _configure_input_size(
    cfg: dict[str, Any],
    input_size: int | tuple[int, int] | None,
) -> dict[str, Any]:
    """Replace $(input_size) placeholders in augmentation config.

    Args:
        cfg: Augmentation config dict with class_path and init_args.
        input_size: Target input size (H, W) or single int.

    Returns:
        Config with placeholders replaced by actual values.
    """
    if input_size is None:
        return cfg

    _input_size: tuple[int, int] = (
        (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)  # type: ignore[assignment]
    )

    def check_type(value: Any, expected_type: Any) -> bool:  # noqa: ANN401
        try:
            typeguard.check_type(value, expected_type)
        except typeguard.TypeCheckError:
            return False
        return True

    model_cls = None
    init_args = cfg.get("init_args", {})

    for key, val in init_args.items():
        if not (isinstance(val, str) and "$(input_size)" in val):
            continue

        if model_cls is None:
            model_cls = import_object_from_module(cfg["class_path"])

        available_types = typing.get_type_hints(model_cls.__init__).get(key)

        if available_types is None or check_type(_input_size, available_types):
            # Pass tuple[int, int]
            init_args[key] = _eval_input_size_str(val.replace("$(input_size)", str(_input_size)))
        elif check_type(_input_size[0], available_types):
            # Pass int
            init_args[key] = _eval_input_size_str(val.replace("$(input_size)", str(_input_size[0])))
        else:
            msg = f"{key} should accept int or tuple[int, int], but accepts {available_types}"
            raise RuntimeError(msg)

    return cfg


def _eval_input_size_str(str_to_eval: str) -> tuple[int, ...] | int:
    """Safely evaluate input_size expressions like '$(input_size) * 0.5'.

    Only multiplication and division are supported for safety.

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
    # Explicit tuple of AST node/types that we allow to be recursively evaluated
    tuple(list(bin_ops.keys()) + list(un_ops.keys()) + [ast.BinOp, ast.UnaryOp, ast.Tuple, ast.Constant])

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
            # Support nested unary operations and constants
            if isinstance(node.operand, (ast.BinOp, ast.UnaryOp, ast.Tuple, ast.Constant)):
                operand = _eval(node.operand)
            else:
                msg = f"Unsupported unary operand type: {type(node.operand)}"
                raise SyntaxError(msg)
            return un_ops[type(node.op)](operand)
        msg = f"Unsupported syntax: {type(node)}. Only *, / operations are allowed."
        raise SyntaxError(msg)

    ret = _eval(tree)
    if isinstance(ret, np.ndarray):
        return tuple(ret.round().astype(np.int32).tolist())
    return round(ret)
