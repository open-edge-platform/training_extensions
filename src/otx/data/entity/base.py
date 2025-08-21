# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX base data entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import torch
import torchvision.transforms.v2.functional as F  # noqa: N812
from torch import Tensor
from torch.utils._pytree import tree_flatten
from torchvision import tv_tensors
from torchvision.utils import _log_api_usage_once

from otx.types.image import ImageColorChannel, ImageType

if TYPE_CHECKING:
    from collections.abc import Mapping


def custom_wrap(wrappee: Tensor, *, like: tv_tensors.TVTensor, **kwargs) -> tv_tensors.TVTensor:
    """Add `Points` in tv_tensors.wrap.

    If `like` is
        - tv_tensors.BoundingBoxes : the `format` and `canvas_size` of `like` are assigned to `wrappee`
        - Points : the `canvas_size` of `like` is assigned to `wrappee`
    Unless, they are passed as `kwargs`.

    Args:
        wrappee (Tensor): The tensor to convert.
        like (tv_tensors.TVTensor): The reference. `wrappee` will be converted into the same subclass as `like`.
        kwargs: Can contain "format" and "canvas_size" if `like` is a tv_tensor.BoundingBoxes,
            or "canvas_size" if `like` is a `Points`. Ignored otherwise.
    """
    if isinstance(like, tv_tensors.BoundingBoxes):
        return tv_tensors.BoundingBoxes._wrap(  # noqa: SLF001
            wrappee,
            format=kwargs.get("format", like.format),
            canvas_size=kwargs.get("canvas_size", like.canvas_size),
        )
    elif isinstance(like, Points):  # noqa: RET505
        return Points._wrap(wrappee, canvas_size=kwargs.get("canvas_size", like.canvas_size))  # noqa: SLF001

    # # TODO(Vlad): remove this after torch upgrade. This workaround prevents a failure when like is also a Tensor
    # if type(like) == type(wrappee):
    #     return wrappee

    return wrappee.as_subclass(type(like))


tv_tensors.wrap = custom_wrap


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
        normalized: If true, this image is normalized with `norm_mean` and `norm_std`
        norm_mean: Mean vector used to normalize this image
        norm_std: Standard deviation vector used to normalize this image
        image_color_channel: Color channel type of this image, RGB or BGR.
        ignored_labels: Label that should be ignored in this image. Default to None.
        keep_ratio: If true, the image is resized while keeping the aspect ratio. Default to False.
    """

    img_idx: int
    img_shape: tuple[int, int]
    ori_shape: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    scale_factor: tuple[float, float] | None = (1.0, 1.0)
    normalized: bool = False
    norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    image_color_channel: ImageColorChannel = ImageColorChannel.RGB
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
        normalized: bool = False,
        norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        ignored_labels: list[int] | None = None,
        keep_ratio: bool = False,
    ) -> ImageInfo:
        image_info = dummy_tensor.as_subclass(cls)
        image_info.img_idx = img_idx
        image_info.img_shape = img_shape
        image_info.ori_shape = ori_shape
        image_info.padding = padding
        image_info.scale_factor = scale_factor
        image_info.normalized = normalized
        image_info.norm_mean = norm_mean
        image_info.norm_std = norm_std
        image_info.image_color_channel = image_color_channel
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
        normalized: bool = False,
        norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
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
            normalized=normalized,
            norm_mean=norm_mean,
            norm_std=norm_std,
            image_color_channel=image_color_channel,
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
                normalized=image_info.normalized,
                norm_mean=image_info.norm_mean,
                norm_std=image_info.norm_std,
                image_color_channel=image_info.image_color_channel,
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
                    normalized=image_info.normalized,
                    norm_mean=image_info.norm_mean,
                    norm_std=image_info.norm_std,
                    image_color_channel=image_info.image_color_channel,
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
            f"normalized={self.normalized}, "
            f"norm_mean={self.norm_mean}, "
            f"norm_std={self.norm_std}, "
            f"image_color_channel={self.image_color_channel}, "
            f"ignored_labels={self.ignored_labels}, "
            f"keep_ratio={self.keep_ratio})"
        )


@F.register_kernel(functional=F.resize, tv_tensor_cls=ImageInfo)
def _resize_image_info(image_info: ImageInfo, size: list[int], **kwargs) -> ImageInfo:  # noqa: ARG001
    """Register ImageInfo to TorchVision v2 resize kernel."""
    if len(size) == 2:
        image_info.img_shape = (size[0], size[1])
    elif len(size) == 1:
        image_info.img_shape = (size[0], size[0])
    else:
        raise ValueError(size)

    ori_h, ori_w = image_info.ori_shape
    new_h, new_w = image_info.img_shape
    image_info.scale_factor = (new_h / ori_h, new_w / ori_w)
    return image_info


@F.register_kernel(functional=F.crop, tv_tensor_cls=ImageInfo)
def _crop_image_info(
    image_info: ImageInfo,
    height: int,
    width: int,
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    image_info.img_shape = (height, width)
    image_info.scale_factor = None
    return image_info


@F.register_kernel(functional=F.resized_crop, tv_tensor_cls=ImageInfo)
def _resized_crop_image_info(
    image_info: ImageInfo,
    size: list[int],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    if len(size) == 2:
        image_info.img_shape = (size[0], size[1])
    elif len(size) == 1:
        image_info.img_shape = (size[0], size[0])
    else:
        raise ValueError(size)

    image_info.scale_factor = None
    return image_info


@F.register_kernel(functional=F.center_crop, tv_tensor_cls=ImageInfo)
def _center_crop_image_info(
    image_info: ImageInfo,
    output_size: list[int],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    img_shape = F._geometry._center_crop_parse_output_size(output_size)  # noqa: SLF001
    image_info.img_shape = (img_shape[0], img_shape[1])

    image_info.scale_factor = None
    return image_info


@F.register_kernel(functional=F.pad, tv_tensor_cls=ImageInfo)
def _pad_image_info(
    image_info: ImageInfo,
    padding: int | list[int],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    left, right, top, bottom = F._geometry._parse_pad_padding(padding)  # noqa: SLF001
    height, width = image_info.img_shape
    image_info.padding = (left, top, right, bottom)
    image_info.img_shape = (height + top + bottom, width + left + right)
    return image_info


@F.register_kernel(functional=F.normalize, tv_tensor_cls=ImageInfo)
def _normalize_image_info(
    image_info: ImageInfo,
    mean: list[float],
    std: list[float],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    image_info.normalized = True
    image_info.norm_mean = (mean[0], mean[1], mean[2])
    image_info.norm_std = (std[0], std[1], std[2])
    return image_info


class Points(tv_tensors.TVTensor):
    """`torch.Tensor` subclass for points.

    Attributes:
        data: Any data that can be turned into a tensor with `torch.as_tensor`.
        canvas_size (two-tuple of ints): Height and width of the corresponding image or video.
        dtype (torch.dtype, optional): Desired data type of the point. If omitted, will be inferred from `data`.
        device (torch.device, optional): Desired device of the point. If omitted and `data` is a
            `torch.Tensor`, the device is taken from it. Otherwise, the point is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the point. If omitted and
            `data` is a `torch.Tensor`, the value is taken from it. Otherwise, defaults to `False`.
    """

    canvas_size: tuple[int, int]

    @classmethod
    def _wrap(cls, tensor: Tensor, *, canvas_size: tuple[int, int]) -> Points:
        points = tensor.as_subclass(cls)
        points.canvas_size = canvas_size
        return points

    def __new__(  # noqa: D102
        cls,
        data: Any,  # noqa: ANN401
        *,
        canvas_size: tuple[int, int],
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Points:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor, canvas_size=canvas_size)

    @classmethod
    def _wrap_output(
        cls,
        output: Tensor,
        args: tuple[()] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Points:
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))
        first_point_from_args = next(x for x in flat_params if isinstance(x, Points))
        canvas_size = first_point_from_args.canvas_size

        if isinstance(output, Tensor) and not isinstance(output, Points):
            output = Points._wrap(output, canvas_size=canvas_size)
        elif isinstance(output, (tuple, list)):
            output = type(output)(Points._wrap(part, canvas_size=canvas_size) for part in output)
        return output

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # noqa: ANN401
        return self._make_repr(canvas_size=self.canvas_size)


def resize_points(
    points: torch.Tensor,
    canvas_size: tuple[int, int],
    size: tuple[int, int] | list[int],
    max_size: int | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize points."""
    old_height, old_width = canvas_size
    new_height, new_width = F._geometry._compute_resized_output_size(  # noqa: SLF001
        canvas_size,
        size=size,
        max_size=max_size,
    )

    if (new_height, new_width) == (old_height, old_width):
        return points, canvas_size

    w_ratio = new_width / old_width
    h_ratio = new_height / old_height
    ratios = torch.tensor([w_ratio, h_ratio], device=points.device)
    return (
        points.mul(ratios).to(points.dtype),
        (new_height, new_width),
    )


@F.register_kernel(functional=F.resize, tv_tensor_cls=Points)
def _resize_points_dispatch(
    inpt: Points,
    size: tuple[int, int] | list[int],
    max_size: int | None = None,
    **kwargs,  # noqa: ARG001
) -> Points:
    output, canvas_size = resize_points(
        inpt.as_subclass(torch.Tensor),
        inpt.canvas_size,
        size,
        max_size=max_size,
    )
    return tv_tensors.wrap(output, like=inpt, canvas_size=canvas_size)


def pad_points(
    points: torch.Tensor,
    canvas_size: tuple[int, int],
    padding: tuple[int, ...] | list[int],
    padding_mode: str = "constant",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad points."""
    if padding_mode not in ["constant"]:
        # TODO(sungchul): add support of other padding modes
        raise ValueError(f"Padding mode '{padding_mode}' is not supported with bounding boxes")  # noqa: EM102, TRY003

    left, right, top, bottom = F._geometry._parse_pad_padding(padding)  # noqa: SLF001

    pad = [left, top]
    points = points + torch.tensor(pad, dtype=points.dtype, device=points.device)

    height, width = canvas_size
    height += top + bottom
    width += left + right
    canvas_size = (height, width)

    return clamp_points(points, canvas_size=canvas_size), canvas_size


@F.register_kernel(functional=F.pad, tv_tensor_cls=Points)
def _pad_points_dispatch(
    inpt: Points,
    padding: tuple[int, ...] | list[int],
    padding_mode: str = "constant",
    **kwargs,  # noqa: ARG001
) -> Points:
    output, canvas_size = pad_points(
        inpt.as_subclass(torch.Tensor),
        canvas_size=inpt.canvas_size,
        padding=padding,
        padding_mode=padding_mode,
    )
    return tv_tensors.wrap(output, like=inpt, canvas_size=canvas_size)


@F.register_kernel(functional=F.get_size, tv_tensor_cls=Points)
def get_size_points(point: Points) -> list[int]:
    """Get size of points."""
    return list(point.canvas_size)


def _clamp_points(points: Tensor, canvas_size: tuple[int, int]) -> Tensor:
    in_dtype = points.dtype
    points = points.clone() if points.is_floating_point() else points.float()
    points[..., 0].clamp_(min=0, max=canvas_size[1])
    points[..., 1].clamp_(min=0, max=canvas_size[0])
    return points.to(in_dtype)


def clamp_points(inpt: Tensor, canvas_size: tuple[int, int] | None = None) -> Tensor:
    """Clamp point range."""
    if not torch.jit.is_scripting():
        _log_api_usage_once(clamp_points)

    if torch.jit.is_scripting() or F._utils.is_pure_tensor(inpt):  # noqa: SLF001
        if canvas_size is None:
            raise ValueError("For pure tensor inputs, `canvas_size` has to be passed.")  # noqa: EM101, TRY003
        return _clamp_points(inpt, canvas_size=canvas_size)
    elif isinstance(inpt, Points):  # noqa: RET505
        if canvas_size is not None:
            raise ValueError("For point tv_tensor inputs, `canvas_size` must not be passed.")  # noqa: EM101, TRY003
        output = _clamp_points(inpt.as_subclass(Tensor), canvas_size=inpt.canvas_size)
        return tv_tensors.wrap(output, like=inpt)
    else:
        raise TypeError(  # noqa: TRY003
            f"Input can either be a plain tensor or a point tv_tensor, but got {type(inpt)} instead.",  # noqa: EM102
        )


class OTXBatchLossEntity(Dict[str, Tensor]):
    """Data entity to represent model output losses."""
