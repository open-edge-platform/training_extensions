# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Config data type objects for data."""
# NOTE: omegaconf would fail to parse dataclass with `from __future__ import annotations` in Python 3.8, 3.9
# ruff: noqa: FA100

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from otx.types.transformer_libs import TransformLibType


@dataclass
class IntensityConfig:
    """Configuration for high-bit-depth intensity mapping.

    Used for medical imaging where inputs may be uint16 (0-65535).

    Attributes:
        storage_dtype: Input storage dtype ("uint8", "uint16", "int16", "float32").
        max_value: Maximum input value for scaling. None = auto (255 for uint8, 65535 for uint16).
        mode: Intensity mapping mode ("scale_to_unit", "window", "percentile").
        window_center: Center of intensity window (for mode="window").
        window_width: Width of intensity window (for mode="window").
        percentile_low: Low percentile for clipping (for mode="percentile").
        percentile_high: High percentile for clipping (for mode="percentile").
    """

    storage_dtype: str = "uint8"
    max_value: float | None = None
    mode: str = "scale_to_unit"
    window_center: float | None = None
    window_width: float | None = None
    percentile_low: float = 1.0
    percentile_high: float = 99.0


@dataclass
class SamplerConfig:
    """Configuration class for defining the sampler used in the data loading process.

    This is passed in the form of a dataclass, which is instantiated when the dataloader is created.

    [TODO]: Need to replace this with a proper Sampler class.
    Currently, SamplerConfig, which belongs to the sampler of SubsetConfig,
    belongs to the nested dataclass of dataclass, which is not easy to instantiate from the CLI.
    So currently replace sampler with a corresponding dataclass that resembles the configuration of another object,
    providing limited functionality.
    """

    class_path: str = "torch.utils.data.RandomSampler"
    init_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubsetConfig:
    """DTO for dataset subset configuration.

    Attributes:
        batch_size (int): Batch size produced by the dataloader.
        subset_name (str): Datumaro Dataset's subset name for this subset config.
        transforms (list[dict[str, Any]] | Compose): [DEPRECATED] Legacy transforms field.
            Use augmentations_cpu instead for new implementations.
        augmentations_cpu (list[dict[str, Any]]): CPU-stage augmentations using torchvision.transforms.v2.
            These run in Dataset workers before collate. Must output fixed-size tensors for batching.
            Examples: Resize, RandomResizedCrop, intensity mapping transforms.
        augmentations_gpu (list[dict[str, Any]]): GPU-stage augmentations using Kornia.
            These run after batch transfer to GPU via Lightning Callback.
            Examples: RandomHorizontalFlip, ColorJiggle, Normalize.
        intensity (IntensityConfig): High-bit-depth intensity mapping configuration.
        transform_lib_type (TransformLibType): [DEPRECATED] Specifies the transform library type used.
        num_workers (int): Number of worker processes for the dataloader.
        sampler (SamplerConfig): Sampler configuration for the dataloader.
        to_tv_image (bool): [DEPRECATED] Whether to convert images to torch tensors.
        input_size (tuple[int, int] | None): Input size expected by the model.
            If `$(input_size)` is present in augmentations, it will be replaced with this value.

    Example:
        ```python
        train_subset_config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            augmentations_cpu=[
                {"class_path": "torchvision.transforms.v2.RandomResizedCrop", "init_args": {"size": (224, 224)}},
            ],
            augmentations_gpu=[
                {"class_path": "kornia.augmentation.RandomHorizontalFlip", "init_args": {"p": 0.5}},
                {"class_path": "kornia.augmentation.Normalize", "init_args": {"mean": [0.485, 0.456, 0.406],
                                                                              "std": [0.229, 0.224, 0.225]}},
            ],
            num_workers=2,
        )
        ```
    """

    batch_size: int = 6
    subset_name: str = "train"
    augmentations_cpu: list[dict[str, Any]] = field(default_factory=list)
    augmentations_gpu: list[dict[str, Any]] = field(default_factory=list)
    intensity: IntensityConfig = field(default_factory=IntensityConfig)
    # DEPRECATED: Legacy fields for backward compatibility during transition
    transforms: list[dict[str, Any]] = field(default_factory=list)
    transform_lib_type: TransformLibType = TransformLibType.TORCHVISION
    num_workers: int = 2
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    to_tv_image: bool = True
    input_size: tuple[int, int] | None = None


@dataclass
class TileConfig:
    """DTO for tiler configuration."""

    enable_tiler: bool = False
    enable_adaptive_tiling: bool = True
    tile_size: tuple[int, int] = (400, 400)
    overlap: float = 0.2
    iou_threshold: float = 0.45
    max_num_instances: int = 1500
    object_tile_ratio: float = 0.03
    sampling_ratio: float = 1.0
    with_full_img: bool = False

    def clone(self) -> TileConfig:
        """Return a deep copied one of this instance."""
        return deepcopy(self)
