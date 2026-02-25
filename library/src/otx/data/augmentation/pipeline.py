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
import os
import typing
from copy import copy, deepcopy
from inspect import isclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import kornia.augmentation as K  # noqa: N812
import torch
import torchvision.transforms.v2 as tvt_v2
import torchvision.utils as tv_utils
import typeguard
from kornia.augmentation.container import ops
from lightning.pytorch.cli import instantiate_class
from omegaconf import DictConfig
from torch import nn

from otx.data.augmentation.intensity import build_intensity_transform
from otx.data.entity.sample import OTXSample
from otx.data.utils import import_object_from_module

if TYPE_CHECKING:
    from otx.config.data import SubsetConfig


# Monkey-patch to fix transform_matrix slicing for list masks
_original_transform_list = ops.MaskSequentialOps.transform_list


@classmethod
def _fixed_transform_list(cls, input, module, param, extra_args=None):  # noqa: ANN001, ANN202, A002
    """Fixed version that slices transform_matrix for each list element."""
    if extra_args is None:
        extra_args = {}
    if isinstance(module, (K.GeometricAugmentationBase2D,)):
        tfm_input = []
        params = cls.get_instance_module_param(param)
        params_i = deepcopy(params)
        for i, inp in enumerate(input):
            params_i["batch_prob"] = params["batch_prob"][i : i + 1]  # Keep tensor shape
            # FIX: Slice transform_matrix for index i
            transform_i = module.transform_matrix[i : i + 1] if module.transform_matrix is not None else None
            tfm_inp = module.transform_masks(
                inp, params=params_i, flags=module.flags, transform=transform_i, **extra_args
            )
            tfm_input.append(tfm_inp)
        return tfm_input
    # Use original for non-geometric
    return _original_transform_list.__func__(cls, input, module, param, extra_args)  # type: ignore[attr-defined]


ops.MaskSequentialOps.transform_list = classmethod(_fixed_transform_list)  # type: ignore[assignment]


class _SampleImageAdapter(nn.Module):
    """Wrap a tensor→tensor transform to operate on the ``.image`` field of a sample.

    This allows raw tensor transforms (e.g. intensity mapping) to be used
    inside ``CPUAugmentationPipeline`` whose ``forward()`` passes full
    :class:`OTXSample` objects through non-native transforms.
    """

    def __init__(self, transform: nn.Module) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, sample: OTXSample) -> OTXSample:
        """Apply wrapped transform to ``sample.image`` in-place and return sample."""
        sample.image = self.transform(sample.image)
        return sample


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
        self._mean, self._std = self._extract_normalization_params(list(self.augmentations))

    @staticmethod
    def _extract_normalization_params(
        augmentations: list[nn.Module],
    ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        """Extract mean and std from the first torchvision Normalize transform found.

        Args:
            augmentations: List of augmentation modules.

        Returns:
            Tuple of (mean, std) extracted from first Normalize found, or (None, None).
        """
        for transform in augmentations:
            if isinstance(transform, tvt_v2.Normalize):
                mean: tuple[float, float, float] = tuple(float(v) for v in transform.mean)  # type: ignore[assignment]
                std: tuple[float, float, float] = tuple(float(v) for v in transform.std)  # type: ignore[assignment]
                return mean, std
        return None, None

    @property
    def mean(self) -> tuple[float, float, float] | None:
        """Get normalization mean."""
        return self._mean

    @property
    def std(self) -> tuple[float, float, float] | None:
        """Get normalization std."""
        return self._std

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
        - Intensity mapping (prepended automatically from ``IntensityConfig``)
        - New ``augmentations_cpu`` field (preferred)
        - Input size placeholder replacement ``$(input_size)``

        The intensity transform is always the **first** operation in the
        pipeline.  For uint8 with ``mode="scale_to_unit"`` this is equivalent
        to the old ``to_dtype(float32, scale=True)``; for uint16 / thermal /
        medical inputs it applies the correct domain-specific mapping.

        Args:
            config: SubsetConfig with augmentations_cpu and intensity.

        Returns:
            CPUAugmentationPipeline ready for use in Dataset.
        """
        input_size = getattr(config, "input_size", None)
        aug_configs = config.augmentations_cpu
        intensity_config = getattr(config, "intensity", None)

        augmentations: list[nn.Module] = []

        # --- 1. Prepend intensity mapping transform ---------------------------
        if intensity_config is not None:
            intensity_transform = build_intensity_transform(intensity_config)
            augmentations.append(_SampleImageAdapter(intensity_transform))

        # --- 2. User-configured augmentations ---------------------------------
        if aug_configs:
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
    def _dispatch_transform(cls, cfg_transform: DictConfig | dict | nn.Module) -> nn.Module:
        """Dispatch and instantiate a transform from config or return as-is.

        Args:
            cfg_transform: Transform config dict or already instantiated transform.

        Returns:
            Instantiated transform.
        """
        if isinstance(cfg_transform, (DictConfig, dict)):
            return instantiate_class(args=(), init=dict(cfg_transform))
        elif isinstance(cfg_transform, nn.Module):
            # Already instantiated transform, return as-is
            return cfg_transform

        msg = f"cfg_transform should be DictConfig | dict | nn.Module. However, its type is {type(cfg_transform)}."
        raise TypeError(msg)

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

    def _is_native_torchvision_transform(self, transform: nn.Module) -> bool:
        """Check if the transform is a native torchvision transform."""
        module = type(transform).__module__
        return module.startswith("torchvision.")

    def _apply_native_transform(self, transform: nn.Module, inputs: OTXSample) -> OTXSample:  # type: ignore[return-value]
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
        outputs: OTXSample | None = inputs[0]  # type: ignore[assignment]
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

        # DEBUG, DELETE
        debug_dir = os.getenv("OTX_GPU_AUG_DEBUG_DIR")
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._debug_max = int(os.getenv("OTX_GPU_AUG_DEBUG_MAX", "50"))
        self._debug_counter = 0
        self._bbox_min_size = float(os.getenv("OTX_GPU_AUG_BBOX_MIN_SIZE", "1.0"))
        self._bbox_min_area = float(os.getenv("OTX_GPU_AUG_BBOX_MIN_AREA", "1.0"))

        # Build Kornia AugmentationSequential for efficient batch processing
        self.aug_sequential: K.AugmentationSequential | None = None
        if self._augmentations_list:
            self.aug_sequential = K.AugmentationSequential(
                *self._augmentations_list,
                data_keys=self._data_keys,
                same_on_batch=False,
            )

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
            if isinstance(aug, K.Normalize):
                flags = aug.flags
                mean = tuple(aug.flags["mean"].tolist())
                std = tuple(aug.flags["std"].tolist())
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
        return [
            obj
            for name in dir(K)
            if (obj := getattr(K, name))
            and isclass(obj)
            and issubclass(obj, (K.AugmentationBase2D, K.IntensityAugmentationBase2D))
        ]

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
                cfg = CPUAugmentationPipeline._configure_input_size(dict(cfg), input_size)  # noqa: SLF001

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
            return instantiate_class(args=(), init=dict(cfg_transform))
        elif isinstance(cfg_transform, nn.Module):
            # Already instantiated transform, return as-is
            return cfg_transform
        msg = f"cfg_transform should be DictConfig | dict | nn.Module. However, its type is {type(cfg_transform)}."
        raise TypeError(msg)

    def forward(
        self,
        images: torch.Tensor,
        labels: list[torch.Tensor] | None = None,
        bboxes: list[torch.Tensor] | None = None,
        masks: list[torch.Tensor] | None = None,
        keypoints: list[torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Apply GPU augmentations to batched data using Kornia AugmentationSequential.

        Args:
            images: Batched images tensor in BCHW format, values in [0, 1].
            labels: List of labels per image (optional).
            bboxes: List of bounding boxes per image (optional).
            masks: List of masks per image (optional). Can be:
                - Semantic segmentation: (C, H, W) format
                - Instance segmentation: (N_instances, H, W) format
            keypoints: List of keypoints per image (optional).

        Returns:
            Dict with augmented data:
            {"images": tensor, "labels": list, "bboxes": list, "masks": list, "keypoints": list}
        """
        if self.aug_sequential is None:
            return {"images": images, "labels": labels, "bboxes": bboxes, "masks": masks, "keypoints": keypoints}

        # Handle instance segmentation masks: Kornia expects (N, C, H, W) but instance
        # masks are (N_instances, H, W). We add a channel dim before and squeeze after.
        # This allows Kornia to properly transform instance masks along with images.
        original_masks = masks
        original_bboxes = bboxes
        original_keypoints = keypoints
        if self._debug_dir is not None:
            self._debug_visualize("before", images, original_bboxes, original_masks, original_keypoints)
        if masks is not None and "mask" in self._data_keys:
            # Instance seg masks: (N_instances, H, W) - 3D per sample
            # Semantic seg masks: (C, H, W) where C is often 1 or num_classes
            # We add channel dim to all masks for consistency with Kornia
            masks = [m.unsqueeze(0) for m in masks]  # (N, H, W) -> (N, 1, H, W)
        # Map data key names to actual data
        data_map = {
            "input": images,
            "label": labels,
            "bbox_xyxy": bboxes,
            "mask": masks,
            "keypoints": keypoints,
        }

        # Build input list in the SAME ORDER as self._data_keys
        # This is critical because Kornia uses the order to match data to keys
        inputs = []
        provided_keys = []
        for key in self._data_keys:
            data = data_map.get(key)
            if data is not None:
                inputs.append(data)
                provided_keys.append(key)

        # Apply augmentation to all inputs
        results = self.aug_sequential(*inputs)

        # Parse results back
        output = {"images": None, "labels": labels, "bboxes": bboxes, "masks": masks, "keypoints": keypoints}
        for i, key in enumerate(provided_keys):
            if key == "input":
                output["images"] = results[i]
            elif key == "label":
                output["labels"] = results[i]
            elif key == "bbox_xyxy":
                output["bboxes"] = results[i]
            elif key == "mask":
                # Remove channel
                mask_results = results[i]
                mask_results = [m.squeeze(0) for m in mask_results]  # (1, N, H, W) -> (N, H, W)
                output["masks"] = mask_results
            elif key == "keypoints":
                output["keypoints"] = results[i]

        # Sanitize geometric annotations after Kornia transforms.
        if output["images"] is not None:
            s_bboxes, s_labels, s_masks, s_keypoints = self._sanitize_annotations(
                typing.cast("torch.Tensor", output["images"]),
                typing.cast("list[torch.Tensor] | None", output["bboxes"]),
                typing.cast("list[torch.Tensor] | None", output["labels"]),
                typing.cast("list[torch.Tensor] | None", output["masks"]),
                typing.cast("list[torch.Tensor] | None", output["keypoints"]),
                min_size=self._bbox_min_size,
                min_area=self._bbox_min_area,
            )
            output["bboxes"] = s_bboxes
            output["labels"] = s_labels
            output["masks"] = s_masks
            output["keypoints"] = s_keypoints

        if self._debug_dir is not None:
            self._debug_visualize(
                "after_gpu",
                typing.cast("torch.Tensor", output["images"]),
                typing.cast("list[torch.Tensor] | None", output["bboxes"]),
                typing.cast("list[torch.Tensor] | None", output["masks"]),
                typing.cast("list[torch.Tensor] | None", output["keypoints"]),
            )

        return output

    def _sanitize_annotations(
        self,
        images: torch.Tensor,
        bboxes: list[torch.Tensor] | None,
        labels: list[torch.Tensor] | None,
        masks: list[torch.Tensor] | None,
        keypoints: list[torch.Tensor] | None,
        min_size: float = 4.0,
        min_area: float = 16.0,
    ) -> tuple[
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        """Sanitize transformed annotations.

        - Clip bboxes to image bounds
        - Remove invalid bboxes (non-finite, x2<=x1, y2<=y1, too small)
        - Filter aligned labels/masks/keypoints using the same valid indices
        """
        if bboxes is None:
            return bboxes, labels, masks, keypoints

        batch_size, _, h, w = images.shape
        if len(bboxes) != batch_size:
            msg = f"GPU sanitize: bboxes batch mismatch, got {len(bboxes)} vs {batch_size}"
            raise RuntimeError(msg)

        if labels is not None:
            if len(labels) != batch_size:
                msg = f"GPU sanitize: labels batch mismatch, got {len(labels)} vs {batch_size}"
                raise RuntimeError(msg)
        if masks is not None:
            if len(masks) != batch_size:
                msg = f"GPU sanitize: masks batch mismatch, got {len(masks)} vs {batch_size}"
                raise RuntimeError(msg)
        if keypoints is not None:
            if len(keypoints) != batch_size:
                msg = f"GPU sanitize: keypoints batch mismatch, got {len(keypoints)} vs {batch_size}"
                raise RuntimeError(msg)

        out_bboxes: list[torch.Tensor] = []
        out_labels: list[torch.Tensor] | None = [] if labels is not None else None
        out_masks: list[torch.Tensor] | None = [] if masks is not None else None
        out_keypoints: list[torch.Tensor] | None = [] if keypoints is not None else None

        for i in range(batch_size):
            boxes = bboxes[i]
            if not (boxes.ndim == 2 and boxes.shape[-1] == 4):
                msg = f"GPU sanitize: bboxes[{i}] must be [N,4], got {tuple(boxes.shape)}"
                raise RuntimeError(msg)

            if boxes.numel() == 0:
                clipped = boxes
                valid = torch.zeros((0,), dtype=torch.bool, device=boxes.device)
            else:
                clipped = boxes
                clipped[:, 0::2].clamp_(0, w)
                clipped[:, 1::2].clamp_(0, h)

                x1, y1, x2, y2 = clipped[:, 0], clipped[:, 1], clipped[:, 2], clipped[:, 3]
                widths = x2 - x1
                heights = y2 - y1
                areas = widths * heights
                valid = (
                    torch.isfinite(clipped).all(dim=1)
                    & (widths > min_size)
                    & (heights > min_size)
                    & (areas >= min_area)
                )

            out_bboxes.append(clipped[valid])

            if out_labels is not None and labels is not None:
                lbl = labels[i]
                if not (lbl.ndim >= 1 and lbl.shape[0] == valid.shape[0]):
                    msg = f"GPU sanitize: labels[{i}] size mismatch with bboxes ({lbl.shape[0]} vs {valid.shape[0]})"
                    raise RuntimeError(msg)
                out_labels.append(lbl[valid])

            if out_masks is not None and masks is not None:
                m = masks[i]
                # For instance masks, first dimension corresponds to object index.
                if m.ndim >= 3 and m.shape[0] == valid.shape[0]:
                    out_masks.append(m[valid])
                elif m.ndim == 2 and valid.shape[0] == 1:
                    out_masks.append(m.unsqueeze(0)[valid])
                else:
                    out_masks.append(m)

            if out_keypoints is not None and keypoints is not None:
                kp = keypoints[i]
                if kp.numel() > 0:
                    kp[..., 0].clamp_(0, w)
                    kp[..., 1].clamp_(0, h)
                if kp.ndim >= 2 and kp.shape[0] == valid.shape[0]:
                    out_keypoints.append(kp[valid])
                else:
                    out_keypoints.append(kp)

        return out_bboxes, out_labels, out_masks, out_keypoints

    def _debug_visualize(
        self,
        tag: str,
        images: torch.Tensor,
        bboxes: list[torch.Tensor] | None,
        masks: list[torch.Tensor] | None,
        keypoints: list[torch.Tensor] | None,
    ) -> None:
        if self._debug_dir is None:
            return
        if self._debug_counter >= self._debug_max:
            return

        assert torch.isfinite(images).all(), "GPU pipeline debug: images contain NaN/Inf"
        img_min = float(images.min().item())
        img_max = float(images.max().item())
        eps = 1e-4
        assert img_min >= -eps and img_max <= 1.0 + eps, (
            f"GPU pipeline debug: image range is out of [0,1], min={img_min:.6f}, max={img_max:.6f}"
        )

        if bboxes is not None:
            assert len(bboxes) == images.shape[0], (
                f"GPU pipeline debug: bboxes batch size mismatch, got {len(bboxes)} vs {images.shape[0]}"
            )
            dup_eps = 1e-3
            for idx, boxes in enumerate(bboxes):
                assert boxes.ndim == 2 and boxes.shape[-1] == 4, (
                    f"GPU pipeline debug: bboxes[{idx}] must be [N,4], got shape={tuple(boxes.shape)}"
                )
                if boxes.numel() == 0:
                    continue

                assert torch.isfinite(boxes).all(), f"GPU pipeline debug: bboxes[{idx}] contain NaN/Inf"

                h, w = images.shape[-2:]
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                assert (x2 > x1).all() and (y2 > y1).all(), (
                    f"GPU pipeline debug: invalid bbox extents at batch index {idx}"
                )
                assert (x1 >= -eps).all() and (y1 >= -eps).all() and (x2 <= w + eps).all() and (y2 <= h + eps).all(), (
                    f"GPU pipeline debug: bboxes[{idx}] are outside image bounds {h}x{w}"
                )

                # Duplicate-box check (within epsilon): fail fast with counts.
                boxes_fp = boxes.to(torch.float32)
                diffs = (boxes_fp[:, None, :] - boxes_fp[None, :, :]).abs().amax(dim=-1)
                duplicate_matrix = diffs <= dup_eps
                duplicate_pairs = torch.triu(duplicate_matrix, diagonal=1)
                dup_pair_count = int(duplicate_pairs.sum().item())
                if dup_pair_count > 0:
                    dup_box_count = int((duplicate_pairs.any(dim=0) | duplicate_pairs.any(dim=1)).sum().item())
                    raise AssertionError(
                        "GPU pipeline debug: duplicate bboxes detected "
                        f"at batch index {idx} (eps={dup_eps}), "
                        f"duplicate_pairs={dup_pair_count}, duplicate_boxes={dup_box_count}, "
                        f"total_boxes={boxes.shape[0]}"
                    )

        if masks is not None:
            assert len(masks) == images.shape[0], (
                f"GPU pipeline debug: masks batch size mismatch, got {len(masks)} vs {images.shape[0]}"
            )
            for idx, mask in enumerate(masks):
                assert torch.isfinite(mask).all(), f"GPU pipeline debug: masks[{idx}] contain NaN/Inf"
                assert mask.ndim in (2, 3), (
                    f"GPU pipeline debug: masks[{idx}] must be 2D or 3D, got shape={tuple(mask.shape)}"
                )
                mh, mw = (mask.shape[-2], mask.shape[-1])
                h, w = images.shape[-2:]
                assert int(mh) == int(h) and int(mw) == int(w), (
                    f"GPU pipeline debug: masks[{idx}] shape ({mh},{mw}) does not match image ({h},{w})"
                )
                if bboxes is not None and mask.ndim == 3:
                    assert mask.shape[0] == bboxes[idx].shape[0], (
                        f"GPU pipeline debug: masks[{idx}] count {mask.shape[0]} != bboxes[{idx}] count {bboxes[idx].shape[0]}"
                    )

        self._debug_dir.mkdir(parents=True, exist_ok=True)
        images = images.detach().cpu().clamp(0, 1)
        images_uint8 = (images * 255).to(torch.uint8)
        batch_box_count = (
            int(sum(int(boxes.shape[0]) for boxes in bboxes))
            if bboxes is not None
            else 0
        )

        batch_size = images_uint8.shape[0]
        for idx in range(batch_size):
            img = images_uint8[idx]

            if masks is not None and idx < len(masks):
                mask = masks[idx]
                mask = mask.detach().cpu()
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                if mask.ndim == 3:
                    mask_bool = mask > 0
                    num_masks = mask_bool.shape[0]
                    colors_list: list[str | tuple[int, int, int]] = [
                        (int(c[0]), int(c[1]), int(c[2])) for c in torch.randint(0, 255, (num_masks, 3))
                    ]
                    img = tv_utils.draw_segmentation_masks(img, mask_bool, colors=colors_list, alpha=0.35)

            if bboxes is not None and idx < len(bboxes):
                boxes = bboxes[idx]
                if boxes.numel() > 0 and boxes.shape[-1] == 4:
                    img = tv_utils.draw_bounding_boxes(img, boxes, colors="yellow", width=2)

            if keypoints is not None and idx < len(keypoints):
                kps = keypoints[idx]
                if kps.numel() > 0:
                    if kps.ndim == 2 and kps.shape[-1] == 2:
                        kps = kps.unsqueeze(0)
                    if kps.ndim == 3 and kps.shape[-1] == 2:
                        img = tv_utils.draw_keypoints(img, kps, colors="red", radius=2)

            # Overlay bbox counters for quick visual sanity checks.
            from PIL import ImageDraw
            from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image

            img_box_count = int(bboxes[idx].shape[0]) if bboxes is not None and idx < len(bboxes) else 0
            dup_box_count = 0
            if bboxes is not None and idx < len(bboxes) and bboxes[idx].numel() > 0:
                boxes_fp = bboxes[idx].to(torch.float32)
                diffs = (boxes_fp[:, None, :] - boxes_fp[None, :, :]).abs().amax(dim=-1)
                duplicate_matrix = diffs <= 1e-3
                duplicate_pairs = torch.triu(duplicate_matrix, diagonal=1)
                dup_box_count = int((duplicate_pairs.any(dim=0) | duplicate_pairs.any(dim=1)).sum().item())

            overlay_text = f"batch_boxes={batch_box_count} img_boxes={img_box_count} dup_boxes={dup_box_count}"

            img_pil = to_pil_image(img)
            draw = ImageDraw.Draw(img_pil)
            text_w = max(120, 7 * len(overlay_text) + 8)
            draw.rectangle((4, 4, text_w, 22), fill=(0, 0, 0))
            draw.text((8, 7), overlay_text, fill=(255, 255, 255))
            img = pil_to_tensor(img_pil)

            save_path = self._debug_dir / f"{self._debug_counter:06d}_{tag}_b{idx}.png"
            tv_utils.save_image(img.float() / 255.0, save_path)

        self._debug_counter += 1

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        aug_str = str(self.aug_sequential) if self.aug_sequential is not None else "  (empty)"
        info = f"  mean={self._mean}, std={self._std}" if self._mean or self._std else ""
        return f"GPUAugmentationPipeline(\n{aug_str}\n  data_keys={self._data_keys}{info}\n)"
