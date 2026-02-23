# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX base data entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from torch import Tensor
from torch.utils._pytree import tree_flatten
from torchvision import tv_tensors

from otx.types.image import ImageType

if TYPE_CHECKING:
    from collections.abc import Mapping


class ImageInfo(tv_tensors.TVTensor):
    """Meta info for image.

    Attributes:
        img_id: Image id
        img_shape: Image shape (heigth, width) after preprocessing
        ori_shape: Image shape (heigth, width) right after loading it
        padding: Number of pixels to pad all borders (left, top, right, bottom)
        scale_factor: Scale factor (height, width) if the image is resized during preprocessing.
            Default value is `(1.0, 1.0)` when there is no resizing. However, if the image is cropped,
            it will lose the scaling information and be `None`.
        ignored_labels: Label that should be ignored in this image. Default to None.
        keep_ratio: If true, the image is resized while keeping the aspect ratio. Default to False.
    """

    img_idx: int
    img_shape: tuple[int, int]
    ori_shape: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    scale_factor: tuple[float, float] | None = (1.0, 1.0)
    ignored_labels: list[int]
    keep_ratio: bool = False

    @classmethod
    def _wrap(
        cls,
        dummy_tensor: Tensor,
        *,
        img_idx: int,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
        scale_factor: tuple[float, float] | None = (1.0, 1.0),
        ignored_labels: list[int] | None = None,
        keep_ratio: bool = False,
    ) -> ImageInfo:
        image_info = dummy_tensor.as_subclass(cls)
        image_info.img_idx = img_idx
        image_info.img_shape = img_shape
        image_info.ori_shape = ori_shape
        image_info.padding = padding
        image_info.scale_factor = scale_factor
        image_info.ignored_labels = ignored_labels if ignored_labels else []
        image_info.keep_ratio = keep_ratio
        return image_info

    def __new__(  # noqa: D102
        cls,
        img_idx: int,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
        scale_factor: tuple[float, float] | None = (1.0, 1.0),
        ignored_labels: list[int] | None = None,
        keep_ratio: bool = False,
    ) -> ImageInfo:
        return cls._wrap(
            dummy_tensor=Tensor(),
            img_idx=img_idx,
            img_shape=img_shape,
            ori_shape=ori_shape,
            padding=padding,
            scale_factor=scale_factor,
            ignored_labels=ignored_labels,
            keep_ratio=keep_ratio,
        )

    @classmethod
    def _wrap_output(
        cls,
        output: Tensor,
        args: tuple[()] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> ImageType:
        """Wrap an output (`torch.Tensor`) obtained from PyTorch function.

        For example, this function will be called when

        >>> img_info = ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10))
        >>> `_wrap_output()` will be called after the PyTorch function `to()` is called
        >>> img_info = img_info.to(device=torch.cuda)
        """
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))

        if isinstance(output, Tensor) and not isinstance(output, ImageInfo):
            image_info = next(x for x in flat_params if isinstance(x, ImageInfo))
            output = ImageInfo._wrap(
                dummy_tensor=output,
                img_idx=image_info.img_idx,
                img_shape=image_info.img_shape,
                ori_shape=image_info.ori_shape,
                padding=image_info.padding,
                scale_factor=image_info.scale_factor,
                ignored_labels=image_info.ignored_labels,
                keep_ratio=image_info.keep_ratio,
            )
        elif isinstance(output, (tuple, list)):
            image_infos = [x for x in flat_params if isinstance(x, ImageInfo)]
            output = type(output)(
                ImageInfo._wrap(
                    dummy_tensor=dummy_tensor,
                    img_idx=image_info.img_idx,
                    img_shape=image_info.img_shape,
                    ori_shape=image_info.ori_shape,
                    padding=image_info.padding,
                    scale_factor=image_info.scale_factor,
                    ignored_labels=image_info.ignored_labels,
                    keep_ratio=image_info.keep_ratio,
                )
                for dummy_tensor, image_info in zip(output, image_infos)
            )
        return output

    def __repr__(self) -> str:
        return (
            "ImageInfo("
            f"img_idx={self.img_idx}, "
            f"img_shape={self.img_shape}, "
            f"ori_shape={self.ori_shape}, "
            f"padding={self.padding}, "
            f"scale_factor={self.scale_factor}, "
            f"ignored_labels={self.ignored_labels}, "
            f"keep_ratio={self.keep_ratio})"
        )


class OTXBatchLossEntity(Dict[str, Tensor]):
    """Data entity to represent model output losses."""
