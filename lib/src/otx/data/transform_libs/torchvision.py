# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helper to support TorchVision data transform functions."""

from __future__ import annotations

import ast
import copy
import itertools
import math
import operator
import typing
from inspect import isclass
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Sequence

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.v2 as tvt_v2
import typeguard
from lightning.pytorch.cli import instantiate_class
from numpy import random
from omegaconf import DictConfig
from scipy.stats import truncnorm
from torchvision import tv_tensors
from torchvision._utils import sequence_to_str
from torchvision.transforms.v2 import GaussianBlur, GaussianNoise
from torchvision.transforms.v2 import functional as F  # noqa: N812

from otx.data.entity.base import (
    Points,
    _crop_image_info,
    _pad_image_info,
    _resize_image_info,
    _resized_crop_image_info,
)
from otx.data.entity.torch import OTXDataItem
from otx.data.transform_libs.utils import (
    CV2_INTERP_CODES,
    cache_randomness,
    centers_bboxes,
    clip_bboxes,
    crop_masks,
    crop_polygons,
    flip_bboxes,
    flip_image,
    flip_masks,
    flip_polygons,
    get_bboxes_from_masks,
    get_bboxes_from_polygons,
    get_image_shape,
    is_inside_bboxes,
    overlap_bboxes,
    project_bboxes,
    project_polygons,
    rescale_bboxes,
    rescale_keypoints,
    rescale_masks,
    rescale_polygons,
    scale_size,
    to_np_image,
    translate_bboxes,
    translate_masks,
    translate_polygons,
)
from otx.data.utils import import_object_from_module

if TYPE_CHECKING:
    from otx.config.data import SubsetConfig


# mypy: disable-error-code="attr-defined"


def custom_query_size(flat_inputs: list[Any]) -> tuple[int, int]:  # noqa: D103
    sizes = {
        tuple(F.get_size(inpt))
        for inpt in flat_inputs
        if tvt_v2._utils.check_type(  # noqa: SLF001
            inpt,
            (
                F.is_pure_tensor,
                tv_tensors.Image,
                PIL.Image.Image,
                tv_tensors.Video,
                tv_tensors.Mask,
                tv_tensors.BoundingBoxes,
                Points,
            ),
        )
    }
    if not sizes:
        raise TypeError("No image, video, mask, bounding box, or point was found in the sample")  # noqa: EM101, TRY003
    elif len(sizes) > 1:  # noqa: RET506
        msg = f"Found multiple HxW dimensions in the sample: {sequence_to_str(sorted(sizes))}"
        raise ValueError(msg)
    h, w = sizes.pop()
    return h, w


tvt_v2._utils.query_size = custom_query_size  # noqa: SLF001


class NumpytoTVTensorMixin:
    """Convert numpy to tv tensors."""

    is_numpy_to_tvtensor: bool

    def convert(self, inputs: OTXDataItem | None) -> OTXDataItem | None:
        """Convert numpy to tv tensors."""
        if self.is_numpy_to_tvtensor and inputs is not None:
            if (image := getattr(inputs, "image", None)) is not None and isinstance(image, np.ndarray):
                inputs.image = F.to_image(image.copy())
            if (bboxes := getattr(inputs, "bboxes", None)) is not None and isinstance(bboxes, np.ndarray):
                inputs.bboxes = tv_tensors.BoundingBoxes(bboxes, format="xyxy", canvas_size=inputs.img_info.img_shape)  # type: ignore[attr-defined, union-attr]
            if (masks := getattr(inputs, "masks", None)) is not None and isinstance(masks, np.ndarray):
                inputs.masks = tv_tensors.Mask(masks)  # type: ignore[union-attr]
        return inputs


class MinIoURandomCrop(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmdet.datasets.transforms.MinIoURandomCrop with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1338-L1490

    Args:
        min_ious (Sequence[float]): minimum IoU threshold for all intersections with bounding boxes.
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w, where a >= min_crop_size).
        bbox_clip_border (bool, optional): Whether clip the objects outside the border of the image. Defaults to True.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
        probability (float): probability of applying this transformation. Defaults to 1.
    """

    def __init__(
        self,
        min_ious: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size: float = 0.3,
        bbox_clip_border: bool = True,
        is_numpy_to_tvtensor: bool = True,
        probability: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor
        self.prob = probability

    @cache_randomness
    def _random_mode(self) -> int | float:
        return random.choice(self.sample_mode)

    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Forward for MinIoURandomCrop."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        if torch.rand(1) >= self.prob:
            return self.convert(inputs)

        img: np.ndarray = to_np_image(inputs.image)
        boxes = inputs.bboxes  # type: ignore[union-attr]
        h, w, c = img.shape
        while True:
            mode = self._random_mode()
            self.mode = mode
            if mode == 1:
                return self.convert(inputs)

            min_iou = self.mode
            for _ in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = (
                    overlap_bboxes(torch.as_tensor(patch.reshape(-1, 4).astype(np.float32)), boxes).numpy().reshape(-1)
                )
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes: torch.Tensor, patch: np.ndarray) -> np.ndarray:
                        centers = centers_bboxes(boxes).numpy()
                        return (
                            (centers[:, 0] > patch[0])
                            * (centers[:, 1] > patch[1])
                            * (centers[:, 0] < patch[2])
                            * (centers[:, 1] < patch[3])
                        )

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    if (bboxes := getattr(inputs, "bboxes", None)) is not None:
                        mask = is_center_of_bboxes_in_patch(bboxes, patch)
                        bboxes = bboxes[mask]
                        bboxes = translate_bboxes(bboxes, (-patch[0], -patch[1]))
                        if self.bbox_clip_border:
                            bboxes = clip_bboxes(bboxes, (patch[3] - patch[1], patch[2] - patch[0]))
                        inputs.bboxes = tv_tensors.BoundingBoxes(  # type: ignore[union-attr]
                            bboxes,
                            format="XYXY",
                            canvas_size=(patch[3] - patch[1], patch[2] - patch[0]),
                        )

                        # labels
                        if inputs.label is not None:
                            inputs.label = inputs.label[mask]  # type: ignore[union-attr]

                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1] : patch[3], patch[0] : patch[2]]
                inputs.image = img
                inputs.img_info = _crop_image_info(inputs.img_info, *img.shape[:2])
                return self.convert(inputs)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(min_ious={self.min_ious}, "
        repr_str += f"min_crop_size={self.min_crop_size}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str


class Resize(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmdet.datasets.transforms.Resize with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L135-L246

    TODO : optimize logic to torcivision pipeline

    Args:
        scale (int or tuple): Images scales for resizing with (height, width). Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing with (height, width).
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        interpolation (str): Interpolation method. Defaults to 'bilinear'.
        interpolation_mask (str): Interpolation method for mask. Defaults to 'nearest'.
        transform_bbox (bool): Whether to transform bounding boxes. Defaults to False.
        transform_keypoints (bool): Whether to transform keypoints. Defaults to False.
        transform_mask (bool): Whether to transform masks. Defaults to False.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
    """

    def __init__(
        self,
        scale: int | tuple[int, int] | None = None,  # (H, W)
        scale_factor: float | tuple[float, float] | None = None,  # (H, W)
        keep_ratio: bool = False,
        clip_object_border: bool = True,
        interpolation: str = "bilinear",
        interpolation_mask: str = "nearest",
        transform_bbox: bool = False,
        transform_keypoints: bool = False,
        transform_mask: bool = False,
        is_numpy_to_tvtensor: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        assert scale is not None or scale_factor is not None, "`scale` and`scale_factor` can not both be `None`"  # noqa: S101

        if scale is None:
            self.scale = None
        elif isinstance(scale, int):
            self.scale = (scale, scale)
        else:
            self.scale = tuple(scale)  # type: ignore[assignment]

        self.transform_bbox = transform_bbox
        self.transform_keypoints = transform_keypoints
        self.transform_mask = transform_mask
        self.interpolation = interpolation
        self.interpolation_mask = interpolation_mask
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple) and len(scale_factor) == 2:
            self.scale_factor = scale_factor
        else:
            msg = f"expect scale_factor is float or Tuple(float), butget {type(scale_factor)}"
            raise TypeError(msg)

        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor

    def _resize_img(self, inputs: OTXDataItem) -> tuple[OTXDataItem, tuple[float, float] | None]:
        """Resize images with inputs.img_info.img_shape."""
        scale_factor: tuple[float, float] | None = getattr(inputs.img_info, "scale_factor", None)  # (H, W)
        if (img := getattr(inputs, "image", None)) is not None:
            img = to_np_image(img)
            img_shape = get_image_shape(img)
            scale: tuple[int, int] = self.scale or scale_size(
                img_shape,
                self.scale_factor,  # type: ignore[arg-type]
            )  # (H, W)

            if self.keep_ratio:
                h, w = img.shape[:2]
                new_scale = min(scale[0] / h, scale[1] / w)
                img = cv2.resize(
                    img,
                    None,
                    fx=new_scale,
                    fy=new_scale,
                    interpolation=CV2_INTERP_CODES[self.interpolation],
                )
                scale = img.shape[:2]

            else:
                img = cv2.resize(img, scale[::-1], interpolation=CV2_INTERP_CODES[self.interpolation])

            inputs.image = img
            inputs.img_info = _resize_image_info(inputs.img_info, img.shape[:2])
            inputs.img_info.keep_ratio = self.keep_ratio  # type: ignore[union-attr]
            scale_factor = (scale[0] / img_shape[0], scale[1] / img_shape[1])
        return inputs, scale_factor

    def _resize_bboxes(self, inputs: OTXDataItem, scale_factor: tuple[float, float]) -> OTXDataItem:
        """Resize bounding boxes with scale_factor only for `Resize`."""
        if (bboxes := getattr(inputs, "bboxes", None)) is not None:
            bboxes = rescale_bboxes(bboxes, scale_factor)
            if self.clip_object_border:
                bboxes = clip_bboxes(bboxes, inputs.img_info.img_shape)  # type: ignore[union-attr]
            inputs.bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=inputs.img_info.img_shape)  # type: ignore[union-attr]
        return inputs

    def _resize_keypoints(self, inputs: OTXDataItem, scale_factor: tuple[float, float]) -> OTXDataItem:
        """Resize keypoints with scale_factor only for `Resize`."""
        if inputs.keypoints is not None:  # type: ignore[union-attr]
            inputs.keypoints[:, :2] = rescale_keypoints(inputs.keypoints[:, :2], scale_factor)  # type: ignore[union-attr]
        return inputs

    def _resize_masks(self, inputs: OTXDataItem, scale_factor: tuple[float, float]) -> OTXDataItem:
        """Resize masks with scale_factor only for `Resize`."""
        masks = getattr(inputs, "masks", None)
        if masks is not None and len(masks) > 0:
            # bit mask
            masks = masks.numpy() if not isinstance(masks, np.ndarray) else masks
            masks = rescale_masks(masks, scale_factor, interpolation=self.interpolation_mask)
            inputs.masks = masks  # type: ignore[union-attr]

        if (polygons := getattr(inputs, "polygons", None)) is not None and len(polygons) > 0:
            # polygon mask
            polygons = rescale_polygons(polygons, scale_factor)
            inputs.polygons = polygons  # type: ignore[union-attr]
        return inputs

    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Transform function to resize images, bounding boxes, and masks."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]
        inputs, scale_factor = self._resize_img(inputs)
        if self.transform_bbox:
            inputs = self._resize_bboxes(inputs, scale_factor)  # type: ignore[arg-type, assignment]

        if self.transform_keypoints:
            inputs = self._resize_keypoints(inputs, scale_factor)  # type: ignore[arg-type, assignment]

        if self.transform_mask:
            inputs = self._resize_masks(inputs, scale_factor)  # type: ignore[arg-type, assignment]

        return self.convert(inputs)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"scale_factor={self.scale_factor}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        repr_str += f"clip_object_border={self.clip_object_border}, "
        repr_str += f"interpolation={self.interpolation}, "
        repr_str += f"interpolation_mask={self.interpolation_mask}, "
        repr_str += f"transform_bbox={self.transform_bbox}, "
        repr_str += f"transform_keypoint={self.transform_keypoints}, "
        repr_str += f"transform_mask={self.transform_mask}, "
        repr_str += f"is_numpy_to_tvtensor={self.is_numpy_to_tvtensor})"
        return repr_str


class RandomResizedCrop(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Crop the given image to random scale and aspect ratio.

    This class implements mmpretrain.datasets.transforms.RandomResizedCrop reimplemented as torchvision.transform.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Args:
        scale (Sequence[int] | int): Desired output scale of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        crop_ratio_range (tuple): Range of the random size of the cropped
            image compared to the original image. Defaults to (0.08, 1.0).
        aspect_ratio_range (tuple): Range of the random aspect ratio of the
            cropped image compared to the original image.
            Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        transform_mask (bool): Whether to transform masks. Defaults to False.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
    """

    def __init__(
        self,
        scale: Sequence[int] | int,
        crop_ratio_range: tuple[float, float] = (0.08, 1.0),
        aspect_ratio_range: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        max_attempts: int = 10,
        interpolation: str = "bilinear",
        transform_mask: bool = False,
        is_numpy_to_tvtensor: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(scale, Sequence):
            assert len(scale) == 2  # noqa: S101
            assert scale[0] > 0  # noqa: S101
            assert scale[1] > 0  # noqa: S101
            self.scale = scale
        else:
            assert scale > 0  # noqa: S101
            self.scale = (scale, scale)
        if (crop_ratio_range[0] > crop_ratio_range[1]) or (aspect_ratio_range[0] > aspect_ratio_range[1]):
            msg = (
                "range should be of kind (min, max). "
                f"But received crop_ratio_range {crop_ratio_range} "
                f"and aspect_ratio_range {aspect_ratio_range}."
            )
            raise ValueError(msg)
        assert isinstance(max_attempts, int)  # noqa: S101
        assert max_attempts >= 0, "max_attempts mush be int and no less than 0."  # noqa: S101
        assert interpolation in (  # noqa: S101
            "nearest",
            "bilinear",
            "bicubic",
            "area",
            "lanczos",
        )

        self.crop_ratio_range = crop_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.transform_mask = transform_mask
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w

        for _ in range(self.max_attempts):
            target_area = np.random.uniform(*self.crop_ratio_range) * area
            log_ratio = (math.log(self.aspect_ratio_range[0]), math.log(self.aspect_ratio_range[1]))
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))
            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_w <= w and 0 < target_h <= h:
                offset_h = np.random.randint(0, h - target_h + 1)
                offset_w = np.random.randint(0, w - target_w + 1)

                return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.aspect_ratio_range):
            target_w = w
            target_h = int(round(target_w / min(self.aspect_ratio_range)))
        elif in_ratio > max(self.aspect_ratio_range):
            target_h = h
            target_w = int(round(target_h * max(self.aspect_ratio_range)))
        else:  # whole image
            target_w = w
            target_h = h
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return offset_h, offset_w, target_h, target_w

    def _bbox_clip(self, bboxes: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
        """Clip bboxes to fit the image shape.

        Copy from mmcv.image.geometric.bbox_clip

        Args:
            bboxes (ndarray): Shape (..., 4*k)
            img_shape (tuple[int]): (height, width) of the image.

        Returns:
            ndarray: Clipped bboxes.
        """
        cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
        cmin[0::2] = img_shape[1] - 1
        cmin[1::2] = img_shape[0] - 1
        return np.maximum(np.minimum(bboxes, cmin), 0)

    def _bbox_scaling(self, bboxes: np.ndarray, scale: float, clip_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Scaling bboxes w.r.t the box center.

        Copy from mmcv.image.geometric.bbox_scaling

        Args:
            bboxes (ndarray): Shape(..., 4).
            scale (float): Scaling factor.
            clip_shape (tuple[int], optional): If specified, bboxes that exceed the
                boundary will be clipped according to the given shape (h, w).

        Returns:
            ndarray: Scaled bboxes.
        """
        if float(scale) == 1.0:
            scaled_bboxes = bboxes.copy()
        else:
            w = bboxes[..., 2] - bboxes[..., 0] + 1
            h = bboxes[..., 3] - bboxes[..., 1] + 1
            dw = (w * (scale - 1)) * 0.5
            dh = (h * (scale - 1)) * 0.5
            scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
        if clip_shape is not None:
            return self._bbox_clip(scaled_bboxes, clip_shape)
        return scaled_bboxes

    def _crop_img(
        self,
        img: np.ndarray,
        bboxes: np.ndarray,
        scale: float = 1.0,
        pad_fill: float | list | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """Crop image patches.

        Copy from mmcv.image.geometric.imcrop
        3 steps: scale the bboxes -> clip bboxes -> crop and pad.

        Args:
            img (ndarray): Image to be cropped.
            bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
            scale (float, optional): Scale ratio of bboxes, the default value
                1.0 means no scaling.
            pad_fill (Number | list[Number]): Value to be filled for padding.
                Default: None, which means no padding.

        Returns:
            list[ndarray] | ndarray: The cropped image patches.
        """
        chn = 1 if img.ndim == 2 else img.shape[2]
        if pad_fill is not None and isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]

        _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
        scaled_bboxes = self._bbox_scaling(_bboxes, scale).astype(np.int32)
        clipped_bbox = self._bbox_clip(scaled_bboxes, img.shape)

        patches = []
        for i in range(clipped_bbox.shape[0]):
            x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
            if pad_fill is None:
                patch = img[y1 : y2 + 1, x1 : x2 + 1, ...]
            else:
                _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
                patch_h = _y2 - _y1 + 1
                patch_w = _x2 - _x1 + 1
                patch_shape = (patch_h, patch_w) if chn == 1 else (patch_h, patch_w, chn)
                patch = np.array(pad_fill, dtype=img.dtype) * np.ones(patch_shape, dtype=img.dtype)
                x_start = 0 if _x1 >= 0 else -_x1
                y_start = 0 if _y1 >= 0 else -_y1
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                patch[y_start : y_start + h, x_start : x_start + w, ...] = img[y1 : y1 + h, x1 : x1 + w, ...]
            patches.append(patch)

        if bboxes.ndim == 1:
            return patches[0]
        return patches

    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Transform function to randomly resized crop images and masks."""
        inputs = _inputs[0]
        if (img := getattr(inputs, "image", None)) is not None:
            img = to_np_image(img)
            offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
            bboxes = np.array(
                [
                    offset_w,
                    offset_h,
                    offset_w + target_w - 1,
                    offset_h + target_h - 1,
                ],
            )
            img = self._crop_img(img, bboxes=bboxes)
            inputs.img_info = _crop_image_info(inputs.img_info, *img.shape[:2])
            img = cv2.resize(
                img,
                tuple(self.scale[::-1]),
                dst=None,
                interpolation=CV2_INTERP_CODES[self.interpolation],
            )
            inputs.image = img
            inputs.img_info = _resize_image_info(inputs.img_info, img.shape[:2])

            if self.transform_mask and (masks := getattr(inputs, "masks", None)) is not None:
                masks = to_np_image(masks)
                masks = self._crop_img(masks, bboxes=bboxes)
                masks = cv2.resize(
                    masks,
                    tuple(self.scale[::-1]),
                    dst=None,
                    interpolation=CV2_INTERP_CODES["nearest"],
                )
                if masks.ndim == 2:
                    masks = masks[None]
                inputs.masks = tv_tensors.Mask(masks)  # type: ignore[union-attr]

        return self.convert(inputs)

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f"(scale={self.scale}"
        repr_str += ", crop_ratio_range="
        repr_str += f"{tuple(round(s, 4) for s in self.crop_ratio_range)}"
        repr_str += ", aspect_ratio_range="
        repr_str += f"{tuple(round(r, 4) for r in self.aspect_ratio_range)}"
        repr_str += f", max_attempts={self.max_attempts}"
        repr_str += f", interpolation={self.interpolation}"
        repr_str += f", transform_mask={self.transform_mask}"
        repr_str += f", is_numpy_to_tvtensor={self.is_numpy_to_tvtensor})"
        return repr_str


class EfficientNetRandomCrop(RandomResizedCrop):
    """EfficientNet style RandomResizedCrop.

    This class implements mmpretrain.datasets.transforms.EfficientNetRandomCrop reimplemented as torchvision.transform.

    Args:
        scale (int): Desired output scale of the crop. Only int size is
            accepted, a square crop (size, size) is made.
        min_covered (Number): Minimum ratio of the cropped area to the original
             area. Defaults to 0.1.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Defaults to 32.
        crop_ratio_range (tuple): Range of the random size of the cropped
            image compared to the original image. Defaults to (0.08, 1.0).
        aspect_ratio_range (tuple): Range of the random aspect ratio of the
            cropped image compared to the original image.
            Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bicubic'.
        backend (str): The image resize backend type, accepted values are
            'cv2' and 'pillow'. Defaults to 'cv2'.
    """

    def __init__(
        self,
        scale: int,
        min_covered: float = 0.1,
        crop_padding: int = 32,
        interpolation: str = "bicubic",
        **kwarg,
    ):
        assert isinstance(scale, int)  # noqa: S101
        super().__init__(scale, interpolation=interpolation, **kwarg)
        assert min_covered >= 0, "min_covered should be no less than 0."  # noqa: S101
        assert crop_padding >= 0, "crop_padding should be no less than 0."  # noqa: S101

        self.min_covered = min_covered
        self.crop_padding = crop_padding

    # https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/data.py
    @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w
        min_target_area = self.crop_ratio_range[0] * area
        max_target_area = self.crop_ratio_range[1] * area

        for _ in range(self.max_attempts):
            aspect_ratio = np.random.uniform(*self.aspect_ratio_range)
            min_target_h = int(round(math.sqrt(min_target_area / aspect_ratio)))
            max_target_h = int(round(math.sqrt(max_target_area / aspect_ratio)))

            if max_target_h * aspect_ratio > w:
                max_target_h = int((w + 0.5 - 1e-7) / aspect_ratio)
                if max_target_h * aspect_ratio > w:
                    max_target_h -= 1

            max_target_h = min(max_target_h, h)
            min_target_h = min(max_target_h, min_target_h)

            # slightly differs from tf implementation
            target_h = int(round(np.random.uniform(min_target_h, max_target_h)))
            target_w = int(round(target_h * aspect_ratio))
            target_area = target_h * target_w

            # slight differs from tf. In tf, if target_area > max_target_area,
            # area will be recalculated
            if (
                target_area < min_target_area
                or target_area > max_target_area
                or target_w > w
                or target_h > h
                or target_area < self.min_covered * area
            ):
                continue

            offset_h = np.random.randint(0, h - target_h + 1)
            offset_w = np.random.randint(0, w - target_w + 1)

            return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        img_short = min(h, w)
        crop_size = self.scale[0] / (self.scale[0] + self.crop_padding) * img_short

        offset_h = max(0, int(round((h - crop_size) / 2.0)))
        offset_w = max(0, int(round((w - crop_size) / 2.0)))
        return offset_h, offset_w, crop_size, crop_size

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = super().__repr__()[:-1]
        repr_str += f", min_covered={self.min_covered}"
        repr_str += f", crop_padding={self.crop_padding})"
        return repr_str


class RandomFlip(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmdet.datasets.transforms.RandomFlip with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L496-L596

    TODO : optimize logic to torcivision pipeline

     - ``prob`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``prob`` .
        E.g., ``prob=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
     - ``prob`` is float, ``direction`` is list of string: the image will
        be ``direction[i]``ly flipped with probability of
        ``prob/len(direction)``.
        E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
     - ``prob`` is list of float, ``direction`` is list of string:
        given ``len(prob) == len(direction)``, the image will
        be ``direction[i]``ly flipped with probability of ``prob[i]``.
        E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with
        probability of 0.3, vertically with probability of 0.5.

    Args:
        probability (float | list[float], optional): The flipping probability.
            Defaults to None.
        direction(str | list[str]): The flipping direction. Options
            If input is a list, the length must equal ``prob``. Each
            element in ``prob`` indicates the flip probability of
            corresponding direction. Defaults to 'horizontal'.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
    """

    def __init__(
        self,
        probability: float | Iterable[float] | None = None,
        direction: str | Sequence[str | None] = "horizontal",
        is_numpy_to_tvtensor: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(probability, list):
            assert all(isinstance(p, float) for p in probability)  # noqa: S101
            assert 0 <= sum(probability) <= 1  # noqa: S101
        elif isinstance(probability, float):
            assert 0 <= probability <= 1  # noqa: S101
        else:
            msg = f"probability must be float or list of float, but got `{type(probability)}`."
            raise TypeError(msg)
        self.prob = probability

        valid_directions = ["horizontal", "vertical", "diagonal"]
        if isinstance(direction, str):
            assert direction in valid_directions  # noqa: S101
        elif isinstance(direction, list):
            assert all(isinstance(d, str) for d in direction)  # noqa: S101
            assert set(direction).issubset(set(valid_directions))  # noqa: S101
        else:
            msg = f"direction must be either str or list of str, but got `{type(direction)}`."
            raise TypeError(msg)
        self.direction = direction

        if isinstance(probability, list):
            assert len(probability) == len(self.direction)  # noqa: S101

        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`."""
        if isinstance(self.direction, Sequence) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = [*list(self.direction), None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = [*self.prob, non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1.0 - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        return np.random.choice(direction_list, p=prob_list)

    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        if (cur_dir := self._choose_direction()) is not None:
            # flip image
            img = to_np_image(inputs.image)
            img = flip_image(img, direction=cur_dir)
            # copy is required as flip_image might return a view which is non-contiguous, and thus cannot be converted
            # to tensor directly
            inputs.image = img
            img_shape = get_image_shape(img)

            # flip bboxes
            if (bboxes := getattr(inputs, "bboxes", None)) is not None:
                bboxes = flip_bboxes(bboxes, inputs.img_info.img_shape, direction=cur_dir)  # type: ignore[union-attr]
                inputs.bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=img_shape)  # type: ignore[union-attr]

            # flip masks
            if (masks := getattr(inputs, "masks", None)) is not None and len(masks) > 0:
                masks = masks.numpy() if not isinstance(masks, np.ndarray) else masks
                inputs.masks = np.stack([flip_image(mask, direction=cur_dir) for mask in masks])  # type: ignore[union-attr]

            # flip polygons
            if (polygons := getattr(inputs, "polygons", None)) is not None and len(polygons) > 0:
                height, width = inputs.img_info.img_shape  # type: ignore[union-attr]
                inputs.polygons = flip_polygons(polygons, height, width, cur_dir)  # type: ignore[union-attr]

        return self.convert(inputs)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob}, "
        repr_str += f"direction={self.direction}, "
        repr_str += f"is_numpy_to_tvtensor={self.is_numpy_to_tvtensor})"
        return repr_str


class RandomGaussianBlur(GaussianBlur):
    """Modified version of the torchvision GaussianBlur."""

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        sigma: int | tuple[float, float] = (0.1, 2.0),
        probability: float = 0.5,
    ) -> None:
        super().__init__(kernel_size=kernel_size, sigma=sigma)
        self.prob = probability

    def transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """Main transform function."""
        if self.prob >= np.random.rand():
            return super().transform(inpt, params)
        return inpt


class RandomGaussianNoise(GaussianNoise):
    """Modified version of the torchvision GaussianNoise.

    This augmentation allows to add gaussian noise to unscaled image.
    Only float32 images are supported for this augmentation.
    """

    def __init__(self, mean: float = 0.0, sigma: float = 0.1, clip: bool = True, probability: float = 0.5) -> None:
        super().__init__(mean=mean, sigma=sigma, clip=clip)
        self.prob = probability

    def _is_scaled(self, tensor: torch.Tensor) -> bool:
        return torch.max(tensor) <= 1 + 1e-5

    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem:
        """Main transform function."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]
        if (img := getattr(inputs, "image", None)) is not None and self.prob >= np.random.rand():
            scaled = self._is_scaled(img)
            sigma = self.sigma * 255 if not scaled else self.sigma
            mean = self.mean * 255 if not scaled else self.mean
            clip = False if not scaled else self.clip

            img = self._call_kernel(F.gaussian_noise, img, mean=mean, sigma=sigma, clip=clip)
            if not scaled:
                img = torch.clamp(img, 0, 255)

            inputs.image = img

        return inputs


class PhotoMetricDistortion(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmdet.datasets.transforms.PhotoMetricDistortion with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1084-L1210

    TODO : optimize logic to torcivision pipeline

    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (sequence): range of contrast.
        saturation_range (sequence): range of saturation.
        hue_delta (int): delta of hue.
        probability (float): the probability of applying each transformation.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
    """

    def __init__(
        self,
        brightness_delta: int = 32,
        contrast: Sequence[int | float] = (0.5, 1.5),
        saturation: Sequence[int | float] = (0.5, 1.5),
        hue_delta: int = 18,
        probability: float = 0.5,
        is_numpy_to_tvtensor: bool = True,
    ) -> None:
        super().__init__()

        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast
        self.saturation_lower, self.saturation_upper = saturation
        self.hue_delta = hue_delta
        self.prob = probability
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor

    @cache_randomness
    def _random_flags(self) -> Sequence[int | float]:
        mode = random.rand() > self.prob
        brightness_flag = random.rand() > self.prob
        contrast_flag = random.rand() > self.prob
        saturation_flag = random.rand() > self.prob
        hue_flag = random.rand() > self.prob
        swap_flag = random.rand() > self.prob
        delta_value = random.uniform(-self.brightness_delta, self.brightness_delta)
        alpha_value = random.uniform(self.contrast_lower, self.contrast_upper)
        saturation_value = random.uniform(self.saturation_lower, self.saturation_upper)
        hue_value = random.uniform(-self.hue_delta, self.hue_delta)
        swap_value = random.permutation(3)

        return (
            mode,
            brightness_flag,
            contrast_flag,
            saturation_flag,
            hue_flag,
            swap_flag,
            delta_value,
            alpha_value,
            saturation_value,
            hue_value,
            swap_value,
        )

    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Transform function to perform photometric distortion on images."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        if (img := getattr(inputs, "image", None)) is not None:
            img = to_np_image(img)
            img = img.astype(np.float32)

            (
                mode,
                brightness_flag,
                contrast_flag,
                saturation_flag,
                hue_flag,
                swap_flag,
                delta_value,
                alpha_value,
                saturation_value,
                hue_value,
                swap_value,
            ) = self._random_flags()

            # random brightness
            if brightness_flag:
                img += delta_value

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            if mode == 1 and contrast_flag:
                img *= alpha_value

            # TODO (sungchul): OTX consumes RGB images but mmx assumes they are BGR.
            # convert color from BGR to HSV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # f32 -> f32

            # random saturation
            if saturation_flag:
                img[..., 1] *= saturation_value
                # For image(type=float32), after convert bgr to hsv by opencv,
                # valid saturation value range is [0, 1]
                if saturation_value > 1:
                    img[..., 1] = img[..., 1].clip(0, 1)

            # random hue
            if hue_flag:
                img[..., 0] += hue_value
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)  # f32 -> f32

            # random contrast
            if mode == 0 and contrast_flag:
                img *= alpha_value

            # randomly swap channels
            if swap_flag:
                img = img[..., swap_value]

            inputs.image = img
        return self.convert(inputs)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(brightness_delta={self.brightness_delta}, "
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)}, "
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)}, "
        repr_str += f"hue_delta={self.hue_delta}, "
        repr_str += f"is_numpy_to_tvtensor={self.is_numpy_to_tvtensor})"
        return repr_str


class RandomAffine(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmdet.datasets.transforms.RandomAffine with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L2736-L2901

    RandomAffine supports images, bounding boxes, masks, and polygons.

    TODO : optimize logic to torcivision pipeline

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float, float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int, int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int, int, int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        transform_mask (bool): Whether to transform the mask. Defaults to True.
        mask_fill_value (int): Fill value for mask. Defaults to 0.
        transform_polygon (bool): Whether to transform polygons. Defaults to True.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
    """

    def __init__(
        self,
        max_rotate_degree: float = 10.0,
        max_translate_ratio: float = 0.1,
        scaling_ratio_range: tuple[float, float] = (0.5, 1.5),
        max_shear_degree: float = 2.0,
        border: tuple[int, int] = (0, 0),  # (H, W)
        border_val: tuple[int, int, int] = (114, 114, 114),
        bbox_clip_border: bool = True,
        transform_mask: bool = True,
        transform_polygon: bool = True,
        recompute_bbox: bool = True,
        mask_fill_value: int = 0,
        is_numpy_to_tvtensor: bool = True,
    ) -> None:
        super().__init__()
        self._validate_parameters(max_translate_ratio, scaling_ratio_range)
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border  # (H, W)
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border
        self.transform_mask = transform_mask
        self.transform_polygon = transform_polygon
        self.recompute_bbox = recompute_bbox
        self.mask_fill_value = mask_fill_value
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor

    @staticmethod
    def _validate_parameters(max_translate_ratio: float, scaling_ratio_range: tuple[float, float]) -> None:
        """Validate input parameters."""
        if not 0 <= max_translate_ratio <= 1:
            msg = f"max_translate_ratio must be between 0 and 1, got {max_translate_ratio}"
            raise ValueError(msg)
        if scaling_ratio_range[0] > scaling_ratio_range[1]:
            msg = f"scaling_ratio_range[0] must be <= scaling_ratio_range[1], got {scaling_ratio_range}"
            raise ValueError(msg)
        if scaling_ratio_range[0] <= 0:
            msg = f"scaling_ratio_range[0] must be > 0, got {scaling_ratio_range[0]}"
            raise ValueError(msg)

    @cache_randomness
    def _get_random_homography_matrix(self, height: int, width: int) -> np.ndarray:
        """Generate random homography matrix for affine transformation.

        Args:
            height (int): Image height including border.
            width (int): Image width including border.

        Returns:
            np.ndarray: 3x3 homography matrix.
        """
        # Generate transformation parameters
        rotation_degree = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        scaling_ratio = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])
        x_shear_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_shear_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        trans_x = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height

        # Create transformation matrices
        rotation_matrix = self._get_rotation_matrix(rotation_degree)
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)
        shear_matrix = self._get_shear_matrix(x_shear_degree, y_shear_degree)
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        # Combine transformations: T * Sh * R * S
        return translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix

    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem:
        """Forward pass of RandomAffine transform.

        Args:
            inputs: Input data containing image and annotations.

        Returns:
            Transformed data item or original input if no valid annotations remain.

        Raises:
            ValueError: If inputs format is invalid.
        """
        if len(_inputs) != 1:
            msg = f"RandomAffine can only transform single input, got {len(_inputs)}"
            raise ValueError(msg)

        inputs = _inputs[0]
        img = to_np_image(inputs.image)

        # Get random homography matrix for affine transformation
        height, width = img.shape[:2]  # type: ignore[union-attr]
        homography_matrix = self._get_random_homography_matrix(height, width)
        output_shape = (height + self.border[0] * 2, width + self.border[1] * 2)

        transformed_img = self._warp_image(img, homography_matrix, output_shape)
        inputs.image = transformed_img
        inputs.img_info = _resize_image_info(inputs.img_info, transformed_img.shape[:2])
        valid_index = None
        valid_bboxes = hasattr(inputs, "bboxes") and inputs.bboxes is not None and len(inputs.bboxes) > 0

        if valid_bboxes:
            # Test transform bboxes to see if any remain valid
            valid_index = self._transform_bboxes(inputs, homography_matrix, output_shape)
            # If no valid annotations will remain after transformation, skip entirely
            if not valid_index.any():
                inputs.image = img
                return self.convert(inputs)  # type: ignore[return-value]

        if hasattr(inputs, "masks") and inputs.masks is not None and len(inputs.masks) > 0:
            self._transform_masks(inputs, homography_matrix, output_shape, valid_index)

        if hasattr(inputs, "polygons") and inputs.polygons is not None and len(inputs.polygons) > 0:
            self._transform_polygons(inputs, homography_matrix, output_shape, valid_index)

        if valid_bboxes and self.recompute_bbox:
            self._recompute_bboxes(inputs, output_shape)

        return self.convert(inputs)  # type: ignore[return-value]

    def _warp_image(
        self,
        image: np.ndarray,
        homography_matrix: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Warp image using the homography matrix.

        Args:
            image: Input image.
            homography_matrix: Homography matrix.
            output_shape: Output shape (height, width).

        Returns:
            np.ndarray: Warped image.
        """
        height, width = output_shape
        return cv2.warpPerspective(image, homography_matrix, dsize=(width, height), borderValue=self.border_val)

    def _transform_bboxes(
        self,
        inputs: OTXDataItem,
        warp_matrix: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Transform bounding boxes and return valid indices.

        Args:
            inputs: Input data item.
            warp_matrix: Transformation matrix.
            output_shape: Output image shape (height, width).

        Returns:
            np.ndarray: Boolean array indicating valid bboxes.
        """
        bboxes = project_bboxes(inputs.bboxes, warp_matrix)

        if self.bbox_clip_border:
            bboxes = clip_bboxes(bboxes, output_shape)

        # Get valid indices and filter
        valid_index = is_inside_bboxes(bboxes, output_shape)

        if valid_index.any():
            inputs.bboxes = tv_tensors.BoundingBoxes(
                bboxes[valid_index],
                format="XYXY",
                canvas_size=output_shape,
            )
            inputs.label = inputs.label[valid_index]  # type: ignore[index]

        return valid_index

    def _transform_masks(
        self,
        inputs: OTXDataItem,
        warp_matrix: np.ndarray,
        output_size: tuple[int, int],
        valid_index: np.ndarray | None = None,
    ) -> None:
        """Transform masks using the warp matrix.

        Args:
            inputs: Input data item.
            warp_matrix: Transformation matrix.
            output_size: Output size (width, height).
            valid_index: Boolean array indicating valid objects.
        """
        if not self.transform_mask or not hasattr(inputs, "masks") or inputs.masks is None or len(inputs.masks) == 0:
            return

        # Convert valid_index to numpy boolean array if it's a tensor
        if valid_index is not None and hasattr(valid_index, "numpy"):
            valid_index = valid_index.numpy()

        # Filter masks using valid_index first
        masks = inputs.masks[valid_index] if valid_index is not None else inputs.masks
        masks = masks.numpy() if not isinstance(masks, np.ndarray) else masks

        if masks.ndim == 3:
            masks = list(masks)

        transformed_masks = []
        for mask in masks:
            transformed_mask = self._warp_single_mask(mask, warp_matrix, output_size)
            transformed_masks.append(transformed_mask)

        if transformed_masks:
            masks_array = np.stack(transformed_masks).astype(np.uint8)
            inputs.masks = tv_tensors.Mask(torch.from_numpy(masks_array > 0).to(torch.bool))

    def _warp_single_mask(self, mask: np.ndarray, warp_matrix: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        """Warp a single mask using appropriate interpolation.

        Args:
            mask: Input mask.
            warp_matrix: Transformation matrix.
            output_size: Output size (width, height).

        Returns:
            np.ndarray: Warped mask.
        """
        unique_values = np.unique(mask)
        height, width = output_size

        # Binary mask: use 255/127 threshold for cleaner results
        if len(unique_values) <= 2 and np.max(unique_values) <= 1:
            warped_mask = cv2.warpPerspective(
                mask.astype(np.uint8) * 255,
                warp_matrix,
                dsize=(width, height),
                borderValue=0,
            )
            return warped_mask > 127

        return cv2.warpPerspective(
            mask.astype(np.uint8),
            warp_matrix,
            dsize=(width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )

    def _transform_polygons(
        self,
        inputs: OTXDataItem,
        warp_matrix: np.ndarray,
        output_shape: tuple[int, int],
        valid_index: np.ndarray | None = None,
    ) -> None:
        """Transform polygons using the warp matrix.

        Args:
            inputs: Input data item.
            warp_matrix: Transformation matrix.
            output_shape: Output shape (height, width).
            valid_index: Boolean array indicating valid objects.
        """
        if (
            not self.transform_polygon
            or not hasattr(inputs, "polygons")
            or inputs.polygons is None
            or len(inputs.polygons) == 0
        ):
            return

        # Convert valid_index to numpy boolean array if it's a tensor
        if valid_index is not None and hasattr(valid_index, "numpy"):
            valid_index = valid_index.numpy()

        # Filter polygons using valid_index if available
        filtered_polygons = (
            [p for p, keep in zip(inputs.polygons, valid_index) if keep] if valid_index is not None else inputs.polygons
        )

        if filtered_polygons:
            inputs.polygons = project_polygons(filtered_polygons, warp_matrix, output_shape)

    def _recompute_bboxes(self, inputs: OTXDataItem, output_shape: tuple[int, int]) -> None:
        """Recomputes the bounding boxes after tranforming from the mask or polygons if available.

        Args:
            inputs: Input data item.
            output_shape: Output shape (height, width).
        """
        has_polygons = hasattr(inputs, "polygons") and inputs.polygons is not None and len(inputs.polygons) > 0
        has_masks = hasattr(inputs, "masks") and inputs.masks is not None and len(inputs.masks) > 0

        if not has_polygons and not has_masks:
            return

        # bboxes here are XYXY format
        bboxes = inputs.bboxes
        bboxes = bboxes.numpy() if not isinstance(bboxes, np.ndarray) else bboxes  # type: ignore[union-attr]

        if has_masks:
            masks = inputs.masks
            masks = masks.numpy() if not isinstance(masks, np.ndarray) else masks  # type: ignore[union-attr]
            for i, mask in enumerate(masks):
                points = cv2.findNonZero(mask.astype(np.uint8))
                if points is not None:
                    x, y, w, h = cv2.boundingRect(points)
                    bboxes[i] = np.array([x, y, x + w, y + h])

        elif has_polygons:
            polygons = inputs.polygons
            for i, polygon in enumerate(polygons):  # type: ignore[arg-type]
                points_1d = np.array(polygon.points, dtype=np.float32)
                if len(points_1d) % 2 != 0:
                    continue

                points = points_1d.reshape(-1, 2)
                x, y, w, h = cv2.boundingRect(points)
                bboxes[i] = np.array([x, y, x + w, y + h])

        inputs.bboxes = tv_tensors.BoundingBoxes(
            bboxes,
            format="XYXY",
            canvas_size=output_shape,
        )

    def __repr__(self) -> str:
        """Return string representation of the transform."""
        params = [
            f"max_rotate_degree={self.max_rotate_degree}",
            f"max_translate_ratio={self.max_translate_ratio}",
            f"scaling_ratio_range={self.scaling_ratio_range}",
            f"max_shear_degree={self.max_shear_degree}",
            f"border={self.border}",
            f"border_val={self.border_val}",
            f"bbox_clip_border={self.bbox_clip_border}",
            f"transform_mask={self.transform_mask}",
            f"transform_polygon={self.transform_polygon}",
            f"mask_fill_value={self.mask_fill_value}",
            f"is_numpy_to_tvtensor={self.is_numpy_to_tvtensor}",
        ]
        return f"{self.__class__.__name__}({', '.join(params)})"

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        """Create rotation transformation matrix.

        Args:
            rotate_degrees: Rotation angle in degrees.

        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        radian = math.radians(rotate_degrees)
        cos_val, sin_val = np.cos(radian), np.sin(radian)
        return np.array(
            [
                [cos_val, -sin_val, 0.0],
                [sin_val, cos_val, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        """Create scaling transformation matrix.

        Args:
            scale_ratio: Scaling factor.

        Returns:
            np.ndarray: 3x3 scaling matrix.
        """
        return np.array(
            [
                [scale_ratio, 0.0, 0.0],
                [0.0, scale_ratio, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float, y_shear_degrees: float) -> np.ndarray:
        """Create shear transformation matrix.

        Args:
            x_shear_degrees: Shear angle in x direction (degrees).
            y_shear_degrees: Shear angle in y direction (degrees).

        Returns:
            np.ndarray: 3x3 shear matrix.
        """
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        return np.array(
            [
                [1, np.tan(x_radian), 0.0],
                [np.tan(y_radian), 1, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        """Create translation transformation matrix.

        Args:
            x: Translation in x direction.
            y: Translation in y direction.

        Returns:
            np.ndarray: 3x3 translation matrix.
        """
        return np.array(
            [
                [1, 0.0, x],
                [0.0, 1, y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )


class CachedMosaic(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmdet.datasets.transforms.CachedMosaic with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L3342-L3573

    TODO : optimize logic to torcivision pipeline

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (height, width).
            Defaults to (640, 640).
        center_ratio_range (tuple[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (float): Pad value. Defaults to 114.0.
        probability (float): Probability of applying this transformation.
            Defaults to 1.0.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
    """

    def __init__(
        self,
        img_scale: tuple[int, int] | list[int] = (640, 640),  # (H, W)
        center_ratio_range: tuple[float, float] = (0.5, 1.5),
        bbox_clip_border: bool = True,
        pad_val: float = 114.0,
        probability: float = 1.0,
        max_cached_images: int = 40,
        random_pop: bool = True,
        is_numpy_to_tvtensor: bool = True,
    ) -> None:
        super().__init__()

        assert isinstance(img_scale, (tuple, list))  # noqa: S101
        assert 0 <= probability <= 1.0, f"The probability should be in range [0,1]. got {probability}."  # noqa: S101

        self.img_scale = img_scale  # (H, W)
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = probability

        self.results_cache: list[OTXDataItem] = []  # type: ignore[valid-type]
        self.random_pop = random_pop
        assert max_cached_images >= 4, f"The length of cache must >= 4, but got {max_cached_images}."  # noqa: S101
        self.max_cached_images = max_cached_images

        self.cnt_cached_images = 0
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor

    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        """Call function to collect indexes.

        Args:
            cache (list): The results cache.

        Returns:
            list: indexes.
        """
        return [random.randint(0, len(cache) - 1) for _ in range(3)]

    @typing.no_type_check  # TODO(ashwinvaidya17): temporary
    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Forward for CachedMosaic."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            index = random.randint(0, len(self.results_cache) - 1) if self.random_pop else 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return self.convert(inputs)

        if random.uniform(0, 1) > self.prob:
            return self.convert(inputs)

        indices = self.get_indexes(self.results_cache)
        mix_results = [copy.deepcopy(self.results_cache[i]) for i in indices]

        # TODO (mmdetection): refactor mosaic to reuse these code.
        # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L3465
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_masks = []
        mosaic_polygons = []
        with_mask = bool(hasattr(inputs, "masks") or hasattr(inputs, "polygons"))

        inp_img: np.ndarray = to_np_image(inputs.image)
        if len(inp_img.shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=inp_img.dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=inp_img.dtype,
            )

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            results_patch = copy.deepcopy(inputs) if loc == "top_left" else copy.deepcopy(mix_results[i - 1])

            img_i: np.ndarray = to_np_image(results_patch.image)
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            img_i = cv2.resize(
                img_i,
                (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)),
                interpolation=cv2.INTER_LINEAR,
            )

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch.bboxes
            gt_bboxes_labels_i = results_patch.label

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i = rescale_bboxes(gt_bboxes_i, (scale_ratio_i, scale_ratio_i))
            gt_bboxes_i = translate_bboxes(gt_bboxes_i, (padw, padh))
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            if with_mask:
                if (gt_masks_i := getattr(results_patch, "masks", None)) is not None and len(gt_masks_i) > 0:
                    gt_masks_i = gt_masks_i.numpy() if not isinstance(gt_masks_i, np.ndarray) else gt_masks_i
                    gt_masks_i = rescale_masks(gt_masks_i, float(scale_ratio_i))
                    gt_masks_i = translate_masks(
                        gt_masks_i,
                        out_shape=(int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                        offset=padw,
                        direction="horizontal",
                    )
                    gt_masks_i = translate_masks(
                        gt_masks_i,
                        out_shape=(int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                        offset=padh,
                        direction="vertical",
                    )
                    mosaic_masks.append(gt_masks_i)

                if (gt_polygons_i := getattr(results_patch, "polygons", None)) is not None and len(gt_polygons_i) > 0:
                    gt_polygons_i = rescale_polygons(gt_polygons_i, float(scale_ratio_i))
                    gt_polygons_i = translate_polygons(
                        gt_polygons_i,
                        out_shape=(int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                        offset=padw,
                        direction="horizontal",
                    )
                    gt_polygons_i = translate_polygons(
                        gt_polygons_i,
                        out_shape=(int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                        offset=padh,
                        direction="vertical",
                    )
                    mosaic_polygons.append(gt_polygons_i)

        mosaic_bboxes = torch.cat(mosaic_bboxes, dim=0)
        mosaic_bboxes_labels = torch.cat(mosaic_bboxes_labels, dim=0)

        if self.bbox_clip_border:
            mosaic_bboxes = clip_bboxes(mosaic_bboxes, (2 * self.img_scale[0], 2 * self.img_scale[1]))

        # remove outside bboxes
        inside_inds = is_inside_bboxes(mosaic_bboxes, (2 * self.img_scale[0], 2 * self.img_scale[1])).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]

        inputs.image = mosaic_img
        inputs.img_info = _resized_crop_image_info(
            inputs.img_info,
            mosaic_img.shape[:2],
        )  # TODO (sungchul): need to add proper function

        inputs.bboxes = tv_tensors.BoundingBoxes(mosaic_bboxes, format="XYXY", canvas_size=mosaic_img.shape[:2])
        inputs.label = mosaic_bboxes_labels
        if with_mask:
            if len(mosaic_masks) > 0:
                inputs.masks = np.concatenate(mosaic_masks, axis=0)[inside_inds]
            if len(mosaic_polygons) > 0:
                inputs.polygons = [
                    polygon for ind, polygon in zip(inside_inds, itertools.chain(*mosaic_polygons)) if ind
                ]  # type: ignore[union-attr]
        return self.convert(inputs)

    def _mosaic_combine(
        self,
        loc: str,
        center_position_xy: Sequence[float],
        img_shape_wh: Sequence[int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Calculate global coordinate of mosaic image and local coordinate of cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ("top_left",
              "top_right", "bottom_left", "bottom_right").
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[int]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ("top_left", "top_right", "bottom_left", "bottom_right")  # noqa: S101
        if loc == "top_left":
            # index0 to top left part of image
            x1, y1, x2, y2 = map(
                int,
                (
                    max(center_position_xy[0] - img_shape_wh[0], 0),
                    max(center_position_xy[1] - img_shape_wh[1], 0),
                    center_position_xy[0],
                    center_position_xy[1],
                ),
            )
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == "top_right":
            # index1 to top right part of image
            x1, y1, x2, y2 = map(
                int,
                (
                    center_position_xy[0],
                    max(center_position_xy[1] - img_shape_wh[1], 0),
                    min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2),
                    center_position_xy[1],
                ),
            )
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == "bottom_left":
            # index2 to bottom left part of image
            x1, y1, x2, y2 = map(
                int,
                (
                    max(center_position_xy[0] - img_shape_wh[0], 0),
                    center_position_xy[1],
                    center_position_xy[0],
                    min(self.img_scale[0] * 2, center_position_xy[1] + img_shape_wh[1]),
                ),
            )
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = map(
                int,
                (
                    center_position_xy[0],
                    center_position_xy[1],
                    min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2),
                    min(self.img_scale[0] * 2, center_position_xy[1] + img_shape_wh[1]),
                ),
            )
            crop_coord = 0, 0, min(img_shape_wh[0], x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"center_ratio_range={self.center_ratio_range}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"max_cached_images={self.max_cached_images}, "
        repr_str += f"random_pop={self.random_pop}, "
        repr_str += f"is_numpy_to_tvtensor={self.is_numpy_to_tvtensor})"
        return repr_str


class CachedMixUp(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmdet.datasets.transforms.CachedMixup with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L3577-L3854

    TODO : optimize logic to torcivision pipeline

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (height, width). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (float): Pad value. Defaults to 114.0.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        probability (float): Probability of applying this transformation.
            Defaults to 1.0.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
    """

    def __init__(
        self,
        img_scale: tuple[int, int] | list[int] = (640, 640),  # (H, W)
        ratio_range: tuple[float, float] = (0.5, 1.5),
        flip_ratio: float = 0.5,
        pad_val: float = 114.0,
        max_iters: int = 15,
        bbox_clip_border: bool = True,
        max_cached_images: int = 20,
        random_pop: bool = True,
        probability: float = 1.0,
        is_numpy_to_tvtensor: bool = True,
    ) -> None:
        super().__init__()

        assert isinstance(img_scale, (tuple, list))  # noqa: S101
        assert max_cached_images >= 2, f"The length of cache must >= 2, but got {max_cached_images}."  # noqa: S101
        assert 0 <= probability <= 1.0, f"The probability should be in range [0,1]. got {probability}."  # noqa: S101
        self.dynamic_scale = img_scale  # (H, W)
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border
        self.results_cache: list[OTXDataItem] = []

        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.prob = probability
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor

    @cache_randomness
    def get_indexes(self, cache: list) -> int:
        """Call function to collect indexes.

        Args:
            cache (list): The result cache.

        Returns:
            int: index.
        """
        for _ in range(self.max_iters):
            index = random.randint(0, len(cache) - 1)
            gt_bboxes_i = cache[index].bboxes
            if len(gt_bboxes_i) != 0:
                break
        return index

    @typing.no_type_check  # TODO(ashwinvaidya17): temporary
    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """MixUp transform function."""
        # cache and pop images
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            index = random.randint(0, len(self.results_cache) - 1) if self.random_pop else 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 1:
            return self.convert(inputs)

        if random.uniform(0, 1) > self.prob:
            return self.convert(inputs)

        index = self.get_indexes(self.results_cache)
        retrieve_results = copy.deepcopy(self.results_cache[index])

        # TODO (mmdetection): refactor mixup to reuse these code.
        # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L3721
        if retrieve_results.bboxes.shape[0] == 0:
            # empty bbox
            return self.convert(inputs)

        retrieve_img: np.ndarray = to_np_image(retrieve_results.image)
        with_mask = bool(hasattr(inputs, "masks") or hasattr(inputs, "polygons"))

        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = (
                np.ones((self.dynamic_scale[0], self.dynamic_scale[1], 3), dtype=retrieve_img.dtype) * self.pad_val
            )
        else:
            out_img = np.ones(self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0], self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = cv2.resize(
            retrieve_img,
            (int(retrieve_img.shape[1] * scale_ratio), int(retrieve_img.shape[0] * scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2. paste
        out_img[: retrieve_img.shape[0], : retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = cv2.resize(
            out_img,
            (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)),
            interpolation=cv2.INTER_LINEAR,
        )

        # 4. flip
        if is_flip:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img: np.ndarray = to_np_image(inputs.image)
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.ones((max(origin_h, target_h), max(origin_w, target_w), 3)) * self.pad_val
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset : y_offset + target_h, x_offset : x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results.bboxes
        retrieve_gt_bboxes = rescale_bboxes(retrieve_gt_bboxes, (scale_ratio, scale_ratio))

        if self.bbox_clip_border:
            retrieve_gt_bboxes = clip_bboxes(retrieve_gt_bboxes, (origin_h, origin_w))

        if is_flip:
            retrieve_gt_bboxes = flip_bboxes(retrieve_gt_bboxes, (origin_h, origin_w), direction="horizontal")

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes = translate_bboxes(cp_retrieve_gt_bboxes, (-x_offset, -y_offset))

        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes = clip_bboxes(cp_retrieve_gt_bboxes, (target_h, target_w))

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        # TODO(ashwinvaidya17): remove this once we have a unified TorchDataItem
        if isinstance(retrieve_results, OTXDataItem):
            retrieve_gt_bboxes_labels = retrieve_results.label
        else:
            retrieve_gt_bboxes_labels = retrieve_results.labels

        mixup_gt_bboxes = torch.cat((inputs.bboxes, cp_retrieve_gt_bboxes), dim=0)
        # TODO(ashwinvaidya17): remove this once we have a unified TorchDataItem
        mixup_gt_bboxes_labels = torch.cat((inputs.label, retrieve_gt_bboxes_labels), dim=0)

        # remove outside bbox
        inside_inds = is_inside_bboxes(mixup_gt_bboxes, (target_h, target_w))
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]

        inputs.image = mixup_img.astype(np.uint8)
        inputs.img_info = _resized_crop_image_info(
            inputs.img_info,
            mixup_img.shape[:2],
        )  # TODO (sungchul): need to add proper function
        inputs.bboxes = tv_tensors.BoundingBoxes(mixup_gt_bboxes, format="XYXY", canvas_size=mixup_img.shape[:2])
        inputs.label = mixup_gt_bboxes_labels
        if with_mask:
            inside_inds = inside_inds.numpy()
            if (masks := getattr(retrieve_results, "masks", None)) is not None and len(masks) > 0:
                masks = masks.numpy() if not isinstance(masks, np.ndarray) else masks

                # 6. adjust bbox
                retrieve_gt_masks = rescale_masks(masks, scale_ratio)
                if is_flip:
                    retrieve_gt_masks = flip_masks(retrieve_gt_masks)

                # 7. filter
                retrieve_gt_masks = translate_masks(
                    retrieve_gt_masks,
                    out_shape=(target_h, target_w),
                    offset=-x_offset,
                    direction="horizontal",
                )
                retrieve_gt_masks = translate_masks(
                    retrieve_gt_masks,
                    out_shape=(target_h, target_w),
                    offset=-y_offset,
                    direction="vertical",
                )

                # 8. mix up
                inputs_masks = inputs.masks.numpy() if not isinstance(inputs.masks, np.ndarray) else inputs.masks
                mixup_gt_masks = np.concatenate([inputs_masks, retrieve_gt_masks])

                inputs.masks = mixup_gt_masks[inside_inds]

            if (polygons := getattr(retrieve_results, "polygons", None)) is not None and len(polygons) > 0:
                # 6. adjust bbox
                retrieve_gt_polygons = rescale_polygons(polygons, scale_ratio)
                if is_flip:
                    height, width = retrieve_results.img_info.img_shape
                    retrieve_gt_polygons = flip_polygons(retrieve_gt_polygons, height, width)

                # 7. filter
                retrieve_gt_polygons = translate_polygons(
                    retrieve_gt_polygons,
                    out_shape=(target_h, target_w),
                    offset=-x_offset,
                    direction="horizontal",
                )
                retrieve_gt_polygons = translate_polygons(
                    retrieve_gt_polygons,
                    out_shape=(target_h, target_w),
                    offset=-y_offset,
                    direction="vertical",
                )

                # 8. mix up
                mixup_gt_polygons = list(itertools.chain(*[inputs.polygons, retrieve_gt_polygons]))

                inputs.polygons = [mixup_gt_polygons[i] for i in np.where(inside_inds)[0]]

        return self.convert(inputs)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(dynamic_scale={self.dynamic_scale}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"flip_ratio={self.flip_ratio}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"max_iters={self.max_iters}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border}, "
        repr_str += f"max_cached_images={self.max_cached_images}, "
        repr_str += f"random_pop={self.random_pop}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"is_numpy_to_tvtensor={self.is_numpy_to_tvtensor})"
        return repr_str


class YOLOXHSVRandomAug(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmdet.datasets.transforms.YOLOXHSVRandomAug with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L2905-L2961

    TODO : optimize logic to torcivision pipeline

    Args:
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delat of value. Defaults to 30.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
    """

    def __init__(
        self,
        hue_delta: int = 5,
        saturation_delta: int = 30,
        value_delta: int = 30,
        is_numpy_to_tvtensor: bool = True,
    ) -> None:
        super().__init__()

        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor

    @cache_randomness
    def _get_hsv_gains(self) -> np.ndarray:
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta,
            self.saturation_delta,
            self.value_delta,
        ]
        # random selection of h, s, v
        hsv_gains *= random.randint(0, 2, 3)
        # prevent overflow
        return hsv_gains.astype(np.int16)

    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Forward for random hsv transform."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        img: np.ndarray = to_np_image(inputs.image)
        hsv_gains = self._get_hsv_gains()
        # TODO (sungchul): OTX det models except for YOLOX-S, L, X consume RGB images but mmdet assumes they are BGR.
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        img = cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR)
        inputs.image = img
        return self.convert(inputs)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(hue_delta={self.hue_delta}, "
        repr_str += f"saturation_delta={self.saturation_delta}, "
        repr_str += f"value_delta={self.value_delta}, "
        repr_str += f"is_numpy_to_tvtensor={self.is_numpy_to_tvtensor})"
        return repr_str


class Pad(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmdet.datasets.transforms.Pad with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L705-L784

    TODO : optimize logic to torcivision pipeline

    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (height, width). Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Defaults to False.
        pad_val (int | float | dict[str, int | float], optional) - Padding value for if
            the pad_mode is "constant".  If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 255. If it is a dict, it should have the
            following keys:

            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.
            Defaults to dict(img=0, seg=255).
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Defaults to 'constant'.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
        transform_mask (bool): Whether to transform masks. Defaults to False.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
    """

    border_type: ClassVar = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
    }

    def __init__(
        self,
        size: tuple[int, int] | None = None,  # (H, W)
        size_divisor: int | None = None,
        pad_to_square: bool = False,
        pad_val: int | float | dict | None = None,
        padding_mode: str = "constant",
        transform_point: bool = False,
        transform_mask: bool = False,
        is_numpy_to_tvtensor: bool = True,
    ) -> None:
        super().__init__()

        self.size = size
        self.size_divisor = size_divisor
        pad_val = pad_val or {"img": 0, "mask": 0}
        if isinstance(pad_val, int):
            pad_val = {"img": pad_val, "mask": 0}
        assert isinstance(pad_val, dict), "pad_val "  # noqa: S101
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None, "The size and size_divisor must be None when pad2square is True"  # noqa: S101
        else:
            assert size is not None or size_divisor is not None, "only one of size and size_divisor should be valid"  # noqa: S101
            assert size is None or size_divisor is None  # noqa: S101
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]  # noqa: S101
        self.padding_mode = padding_mode
        self.transform_point = transform_point
        self.transform_mask = transform_mask
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor

    def _pad_img(self, inputs: OTXDataItem) -> OTXDataItem:
        """Pad images according to ``self.size``."""
        img: np.ndarray = to_np_image(inputs.image)
        pad_val = self.pad_val.get("img", 0)

        size: tuple[int, int]
        if self.pad_to_square:
            max_size = max(img.shape[:2])
            size = (max_size, max_size)

        if self.size_divisor is not None:
            if not self.pad_to_square:
                size = (img.shape[0], img.shape[1])
            pad_h = int(np.ceil(size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size  # (H, W)

        if isinstance(pad_val, int) and img.ndim == 3:
            pad_val = tuple(pad_val for _ in range(img.shape[2]))

        width = max(size[1] - img.shape[1], 0)
        height = max(size[0] - img.shape[0], 0)
        padding = [0, 0, width, height]

        padded_img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            self.border_type[self.padding_mode],
            value=pad_val,
        )
        inputs.image = padded_img
        inputs.img_info = _pad_image_info(inputs.img_info, padding)
        return inputs

    @typing.no_type_check  # TODO(ashwinvaidya17): temporary
    def _pad_masks(self, inputs: OTXDataItem) -> OTXDataItem:
        """Pad masks according to inputs.image_info.padding."""
        if (masks := getattr(inputs, "masks", None)) is not None and len(masks) > 0:
            masks = masks.numpy() if not isinstance(masks, np.ndarray) else masks

            pad_val = self.pad_val.get("mask", 0)
            padding = inputs.img_info.padding

            padded_masks = []
            for mask in masks:
                orig_dtype = mask.dtype
                # cv2.copyMakeBorder does not support bool, so cast to uint8 if needed
                if mask.dtype == np.bool_:
                    mask_to_pad = mask.astype(np.uint8)
                    pad_val_cast = int(bool(pad_val))
                else:
                    mask_to_pad = mask
                    pad_val_cast = pad_val

                padded = cv2.copyMakeBorder(
                    mask_to_pad,
                    padding[1],
                    padding[3],
                    padding[0],
                    padding[2],
                    self.border_type[self.padding_mode],
                    value=pad_val_cast,
                )
                # Cast back to original dtype if needed
                if orig_dtype == np.bool_:
                    padded = padded.astype(np.bool_)
                padded_masks.append(padded)

            inputs.masks = np.stack(padded_masks)

        return inputs

    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Forward function to pad images."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        outputs = self._pad_img(inputs)

        if self.transform_mask:
            outputs = self._pad_masks(outputs)

        return self.convert(outputs)


class RandomResize(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmcv.transforms.RandomResize with torchvision format.

    Reference : https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/transforms/processing.py#L1381-L1562

    Args:
        scale (Sequence): Images scales for resizing with (height, width). Defaults to None.
        ratio_range (tuple[float], optional): (min_ratio, max_ratio). Defaults to None.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
        **resize_kwargs: Other keyword arguments for the ``resize_type``.
    """

    def __init__(
        self,
        scale: Sequence[int | tuple[int, int]],  # (H, W)
        ratio_range: tuple[float, float] | None = None,
        is_numpy_to_tvtensor: bool = True,
        **resize_kwargs,
    ) -> None:
        super().__init__()
        if isinstance(scale, list):
            scale = tuple(scale)
        self.scale = scale
        self.ratio_range = ratio_range
        self.resize_kwargs = resize_kwargs
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor
        self.resize = Resize(scale=0, **resize_kwargs, is_numpy_to_tvtensor=is_numpy_to_tvtensor)

    @staticmethod
    def _random_sample(scales: Sequence[tuple[int, int]]) -> tuple:
        """Private function to randomly sample a scale from a Sequence of tuples.

        Args:
            scales (Sequence[tuple]): Images scale range for sampling.
                There must be two tuples in scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple): The targeted scale of the image to be resized.
        """
        assert isinstance(scales, Sequence)  # noqa: S101
        assert all(isinstance(scale, tuple) for scale in scales)  # noqa: S101
        assert len(scales) == 2  # noqa: S101
        scale_0 = [scales[0][0], scales[1][0]]
        scale_1 = [scales[0][1], scales[1][1]]
        edge_0 = np.random.randint(min(scale_0), max(scale_0) + 1)
        edge_1 = np.random.randint(min(scale_1), max(scale_1) + 1)
        return (edge_0, edge_1)

    @staticmethod
    def _random_sample_ratio(scale: tuple, ratio_range: tuple[float, float]) -> tuple:
        """Private function to randomly sample a scale from a tuple.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``scale`` to
        generate sampled scale.

        Args:
            scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``scale``.

        Returns:
            (tuple): The targeted scale of the image to be resized.
        """
        assert isinstance(scale, tuple)  # noqa: S101
        assert len(scale) == 2  # noqa: S101
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio  # noqa: S101
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        return int(scale[0] * ratio), int(scale[1] * ratio)

    @cache_randomness
    def _random_scale(self) -> tuple:
        """Private function to randomly sample an scale according to the type of ``scale``.

        Returns:
            (tuple): The targeted scale of the image to be resized.
        """
        if isinstance(self.scale, tuple) and all(isinstance(s, int) for s in self.scale):
            assert self.ratio_range is not None  # noqa: S101
            assert len(self.ratio_range) == 2  # noqa: S101
            scale = self._random_sample_ratio(self.scale, self.ratio_range)
        elif all(isinstance(s, tuple) for s in self.scale):
            scale = self._random_sample(self.scale)  # type: ignore[arg-type]
        else:
            msg = f'Do not support sampling function for "{self.scale}"'
            raise NotImplementedError(msg)

        return scale

    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Transform function to resize images, bounding boxes, semantic segmentation map."""
        self.resize.scale = self._random_scale()
        outputs = self.resize(*_inputs)
        return self.convert(outputs)

    def __repr__(self) -> str:
        # TODO (sungchul): update other's repr
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"is_numpy_to_tvtensor={self.is_numpy_to_tvtensor}, "
        repr_str += f"resize_kwargs={self.resize_kwargs})"
        return repr_str


class RandomCrop(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Implementation of mmdet.datasets.transforms.RandomCrop with torchvision format.

    Reference : https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/transforms/processing.py#L1381-L1562

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`, then the cropped results are generated.

    Args:
        crop_size (tuple[int, int]): The relative ratio or absolute pixels of
            (height, width).
        crop_type (str, optional): One of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])].
            Defaults to "absolute".
        cat_max_ratio (float): The maximum ratio that single category could occupy.
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Defaults to False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Defaults to False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        ignore_index (int): The label index to be ignored. Defaults to 255.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
    """

    def __init__(
        self,
        crop_size: tuple[int, int],  # (H, W)
        crop_type: str = "absolute",
        cat_max_ratio: int | float = 1,
        allow_negative_crop: bool = False,
        recompute_bbox: bool = False,
        bbox_clip_border: bool = True,
        ignore_index: int = 255,
        is_numpy_to_tvtensor: bool = True,
    ) -> None:
        super().__init__()
        if crop_type not in ["relative_range", "relative", "absolute", "absolute_range"]:
            msg = f"Invalid crop_type {crop_type}."
            raise ValueError(msg)
        if crop_type in ["absolute", "absolute_range"]:
            assert crop_size[0] > 0  # noqa: S101
            assert crop_size[1] > 0  # noqa: S101
            assert isinstance(crop_size[0], int)  # noqa: S101
            assert isinstance(crop_size[1], int)  # noqa: S101
            if crop_type == "absolute_range":
                assert crop_size[0] <= crop_size[1]  # noqa: S101
        else:
            assert 0 < crop_size[0] <= 1  # noqa: S101
            assert 0 < crop_size[1] <= 1  # noqa: S101
        self.crop_size = crop_size  # (H, W)
        self.crop_type = crop_type
        self.cat_max_ratio = cat_max_ratio
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox
        self.ignore_index = ignore_index
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor

    def _generate_crop_bbox(
        self,
        orig_shape: tuple[int, int],
        crop_size: tuple[int, int],
    ) -> tuple:
        """Randomly get a crop bounding box.

        Args:
            orig_shape (tuple): The original shape of the image.
            crop_size (tuple): The size of the crop.

        Returns:
            tuple: Coordinates of the cropped image.
        """
        margin_h = max(orig_shape[0] - crop_size[0], 0)
        margin_w = max(orig_shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        return (crop_x1, crop_y1, crop_x2, crop_y2), offset_h, offset_w

    @typing.no_type_check  # TODO(ashwinvaidya17): temporary
    def _crop_data(
        self,
        inputs: OTXDataItem,
        crop_size: tuple[int, int],
        allow_negative_crop: bool,
    ) -> OTXDataItem | None:
        """Function to randomly crop images, bounding boxes, masks, semantic segmentation maps."""
        assert crop_size[0] > 0  # noqa: S101
        assert crop_size[1] > 0  # noqa: S101

        img: np.ndarray = to_np_image(inputs.image)
        orig_shape = inputs.img_info.img_shape
        crop_bbox, offset_h, offset_w = self._generate_crop_bbox(orig_shape, crop_size)

        # for semantic segmentation
        # reference : https://github.com/open-mmlab/mmsegmentation/blob/v1.2.1/mmseg/datasets/transforms/transforms.py#L281-L290
        if (self.cat_max_ratio < 1.0) and ((masks := getattr(inputs, "masks", None)) is not None and len(masks) > 0):
            # Repeat 10 times
            for _ in range(10):
                seg_temp = crop_masks(masks, np.array(crop_bbox))
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox, offset_h, offset_w = self._generate_crop_bbox(orig_shape, crop_size)

        # crop the image
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        cropped_img_shape = img.shape[:2]

        inputs.image = img
        inputs.img_info = _crop_image_info(inputs.img_info, *cropped_img_shape)

        valid_inds: np.ndarray = np.array([1])  # for semantic segmentation
        # crop bboxes accordingly and clip to the image boundary
        if (bboxes := getattr(inputs, "bboxes", None)) is not None:
            bboxes = translate_bboxes(bboxes, [-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes = clip_bboxes(bboxes, cropped_img_shape)

            valid_inds = is_inside_bboxes(bboxes, cropped_img_shape).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if not valid_inds.any() and not allow_negative_crop:
                return None

            inputs.bboxes = tv_tensors.BoundingBoxes(bboxes[valid_inds], format="XYXY", canvas_size=cropped_img_shape)

            if inputs.label is not None:
                inputs.label = inputs.label[valid_inds]

        if (masks := getattr(inputs, "masks", None)) is not None and len(masks) > 0:
            masks = masks.numpy() if not isinstance(masks, np.ndarray) else masks
            inputs.masks = crop_masks(
                masks[valid_inds.nonzero()[0]],
                np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]),
            )

            if self.recompute_bbox:
                inputs.bboxes = tv_tensors.wrap(
                    torch.as_tensor(get_bboxes_from_masks(inputs.masks)),
                    like=inputs.bboxes,
                )

        if (polygons := getattr(inputs, "polygons", None)) is not None and len(polygons) > 0:
            inputs.polygons = crop_polygons(
                [polygons[i] for i in valid_inds.nonzero()[0]],
                np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]),
                *orig_shape,
            )

            if self.recompute_bbox:
                inputs.bboxes = tv_tensors.wrap(
                    torch.as_tensor(get_bboxes_from_polygons(inputs.polygons, *cropped_img_shape)),  # type: ignore[union-attr]
                    like=inputs.bboxes,
                )

        return inputs

    @cache_randomness
    def _rand_offset(self, margin: tuple[int, int]) -> tuple[int, int]:
        """Randomly generate crop offset.

        Args:
            margin (tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return offset_h, offset_w

    @cache_randomness
    def _get_crop_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Randomly generates the absolute crop size based on `crop_type` and `image_size`.

        Args:
            image_size (tuple[int, int]): (h, w).

        Returns:
            crop_size (tuple[int, int]): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == "absolute":
            return min(self.crop_size[0], h), min(self.crop_size[1], w)

        if self.crop_type == "absolute_range":
            # `self.crop_size` is used as range, not absolute value
            crop_h = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w

        if self.crop_type == "relative":
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

        # 'relative_range'
        crop_size = np.asarray(self.crop_size, dtype=np.float32)
        crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
        return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    @typing.no_type_check  # TODO(ashwinvaidya17): temporary
    def forward(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Transform function to randomly crop images, bounding boxes, masks, and polygons."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        crop_size = self._get_crop_size(inputs.img_info.img_shape)

        outputs = self._crop_data(inputs, crop_size, self.allow_negative_crop)
        return self.convert(outputs)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(crop_size={self.crop_size}, "
        repr_str += f"crop_type={self.crop_type}, "
        repr_str += f"allow_negative_crop={self.allow_negative_crop}, "
        repr_str += f"recompute_bbox={self.recompute_bbox}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border}, "
        repr_str += f"is_numpy_to_tvtensor={self.is_numpy_to_tvtensor})"
        return repr_str


class Compose(tvt_v2.Compose):
    """Re-implementation of torchvision.transforms.v2.Compose.

    MMCV transforms can produce None, so it is required to skip the result.
    """

    def forward(self, *inputs: OTXDataItem) -> OTXDataItem | None:
        """Forward with skipping None."""
        needs_unpacking = len(inputs) > 1
        for transform in self.transforms:
            outputs = transform(*inputs)
            # MMCV transform can produce None. Please see
            # https://github.com/open-mmlab/mmengine/blob/26f22ed283ae4ac3a24b756809e5961efe6f9da8/mmengine/dataset/base_dataset.py#L59-L66
            if outputs is None:
                return outputs
            inputs = outputs if needs_unpacking else (outputs,)
        return outputs


class TopdownAffine(tvt_v2.Transform, NumpytoTVTensorMixin):
    """Get the bbox image as the model input by affine transform.

    Args:
        input_size (tuple[int, int]): The size of the model input.
        probability (float): The probability of applying affine
            transforms. Defaults to 0.5.
        is_numpy_to_tvtensor (bool): Whether convert outputs to tensor. Defaults to False.
        shift_factor (float): The factor of shift. Defaults to 0.16.
        shift_prob (float): The probability of shift. Defaults to 0.3.
        scale_factor (tuple[float, float]): The factor of scale. Defaults to (0.5, 1.5).
        scale_prob (float): The probability of scale. Defaults to 1.0.
        rotate_factor (float): The factor of rotate. Defaults to 80.0.
        rotate_prob (float): The probability of rotate. Defaults to 0.5.
        interpolation (str): The interpolation method. Defaults to "bilinear".
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        probability: float = 1.0,
        is_numpy_to_tvtensor: bool = True,
        shift_factor: float = 0.16,
        shift_prob: float = 0.3,
        scale_factor: tuple[float, float] = (0.5, 1.5),
        scale_prob: float = 1.0,
        rotate_factor: float = 80.0,
        rotate_prob: float = 0.5,
        interpolation: str = "bilinear",
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.is_numpy_to_tvtensor = is_numpy_to_tvtensor
        self.affine_transforms_prob = probability
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob
        self.interpolation = interpolation

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """
        w, h = np.hsplit(bbox_scale, [1])
        return np.where(w > h * aspect_ratio, np.hstack([w, w / aspect_ratio]), np.hstack([h * aspect_ratio, h]))

    @staticmethod
    def _truncnorm(low: float = -1.0, high: float = 1.0, size: int = 4) -> torch.Tensor:
        """Sample from a truncated normal distribution."""
        return truncnorm.rvs(low, high, size=(size)).astype(np.float32)

    @cache_randomness
    def _get_transform_params(self) -> tuple:
        """Get random transform parameters.

        Args:
            num_bboxes (int): The number of bboxes

        Returns:
            tuple:
            - offset (np.ndarray): Offset factor of each bbox in shape (n, 2)
            - scale (np.ndarray): Scaling factor of each bbox in shape (n, 1)
            - rotate (np.ndarray): Rotation degree of each bbox in shape (n,)
        """
        random_v = self._truncnorm()
        offset_v = random_v[:2]
        scale_v = random_v[2:3]
        rotate_v = random_v[3]

        # Get shift parameters
        offset = offset_v * self.shift_factor
        offset = np.where(np.random.rand(1) < self.shift_prob, offset, 0.0)

        # Get scaling parameters
        scale_min, scale_max = self.scale_factor
        mu = (scale_max + scale_min) * 0.5
        sigma = (scale_max - scale_min) * 0.5
        scale = scale_v * sigma + mu
        scale = np.where(np.random.rand(1) < self.scale_prob, scale, 1.0)

        # Get rotation parameters
        rotate = rotate_v * self.rotate_factor
        rotate = np.where(np.random.rand() < self.rotate_prob, rotate, 0.0)

        return offset, scale, rotate

    @staticmethod
    def _get_warp_matrix(
        center: np.ndarray,
        scale: np.ndarray,
        rot: float,
        output_size: tuple[int, int],
        shift: tuple[float, float] = (0.0, 0.0),
        inv: bool = False,
        fix_aspect_ratio: bool = True,
    ) -> np.ndarray:
        """Calculate the affine transformation matrix that can warp the bbox area.

        Args:
            center (np.ndarray[2, ]): Center of the bounding box (x, y).
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            rot (float): Rotation angle (degree).
            output_size (np.ndarray[2, ] | list(2,)): Size of the
                destination heatmaps.
            shift (float): Shift translation ratio wrt the width/height.
                Default (0., 0.).
            inv (bool): Option to inverse the affine transform direction.
                (inv=False: src->dst or inv=True: dst->src)
            fix_aspect_ratio (bool): Whether to fix aspect ratio during transform.
                Defaults to True.

        Returns:
            np.ndarray: A 2x3 transformation matrix
        """
        if len(center) != 2 or len(scale) != 2 or len(output_size) != 2 or len(shift) != 2:
            msg = "center, scale, output_size, and shift should have the length of 2."
            raise ValueError(msg)

        def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
            """Rotate a point by an angle."""
            sn, cs = np.sin(angle_rad), np.cos(angle_rad)
            rot_mat = np.array([[cs, -sn], [sn, cs]])
            return rot_mat @ pt

        def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """To calculate the affine matrix, three pairs of points are required.

            This function is used to get the 3rd point, given 2D points a & b.

            The 3rd point is defined by rotating vector `a - b` by 90 degrees
            anticlockwise, using b as the rotation center.
            """
            direction = a - b
            return b + np.r_[-direction[1], direction[0]]

        shift = np.array(shift)
        src_w, src_h = scale[:2]
        dst_w, dst_h = output_size[:2]

        rot_rad = np.deg2rad(rot)
        src_dir = _rotate_point(np.array([src_w * -0.5, 0.0]), rot_rad)
        dst_dir = np.array([dst_w * -0.5, 0.0])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        if fix_aspect_ratio:
            src[2, :] = _get_3rd_point(src[0, :], src[1, :])
            dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
        else:
            src_dir_2 = _rotate_point(np.array([0.0, src_h * -0.5]), rot_rad)
            dst_dir_2 = np.array([0.0, dst_h * -0.5])
            src[2, :] = center + src_dir_2 + scale * shift
            dst[2, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir_2

        if inv:
            warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return warp_mat

    @staticmethod
    def _get_warp_image(
        image: torch.Tensor | np.ndarray,
        warp_mat: np.ndarray,
        warp_size: tuple[int, int],
    ) -> torch.Tensor:
        numpy_image: np.ndarray = to_np_image(image)
        warped_image = cv2.warpAffine(numpy_image, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
        return torch.from_numpy(warped_image).to(dtype=torch.float32).permute(2, 0, 1)

    @typing.no_type_check  # TODO(ashwinvaidya17): temporary
    def __call__(self, *_inputs: OTXDataItem) -> OTXDataItem | None:
        """Transform function to affine image through warp matrix."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        h, w = self.input_size
        warp_size = (int(w), int(h))
        apply_transforms = np.random.rand()
        ori_img_shape = inputs.img_info.ori_shape

        if apply_transforms <= self.affine_transforms_prob:
            bbox_center = np.array(ori_img_shape) / 2.0
            bbox_scale = np.array(ori_img_shape)

            offset, scale, rotate = self._get_transform_params()
            center = bbox_center + offset * bbox_scale
            scale = self._fix_aspect_ratio(bbox_scale * scale, aspect_ratio=w / h)
            rot = rotate

            warp_mat = self._get_warp_matrix(center, scale, rot, output_size=(w, h))
            inputs.image = self._get_warp_image(inputs.image, warp_mat, warp_size)
            if inputs.keypoints is not None:
                keypoints = np.expand_dims(inputs.keypoints[:, :2], axis=0)
                inputs.keypoints[:, :2] = torch.as_tensor(cv2.transform(keypoints, warp_mat)[0])

        else:
            resized_numpy_image = cv2.resize(
                to_np_image(inputs.image),
                warp_size,
                interpolation=CV2_INTERP_CODES[self.interpolation],
            )
            inputs.image = torch.from_numpy(resized_numpy_image).to(dtype=torch.float32).permute(2, 0, 1)
            if inputs.keypoints is not None:
                scale_factor = (warp_size[0] / ori_img_shape[0], warp_size[1] / ori_img_shape[1])
                inputs.keypoints[:, :2] = rescale_keypoints(inputs.keypoints[:, :2], scale_factor)

        if inputs.keypoints is None:
            inputs.keypoints = torch.zeros([])
        else:
            # update keypoints_visible after affine transforms
            inputs.keypoints[:, 2] = inputs.keypoints[:, 2] * (inputs.keypoints[:, :2] > 0).all(axis=1)

        return self.convert(inputs)

    def __repr__(self) -> str:
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(input_size={self.input_size},"
        repr_str += f"is_numpy_to_tvtensor={self.is_numpy_to_tvtensor})."
        return repr_str


class TorchVisionTransformLib:
    """Helper to support TorchVision transforms (only V2) in OTX."""

    @classmethod
    def list_available_transforms(cls) -> list[type[tvt_v2.Transform]]:
        """List available TorchVision transform (only V2) classes."""
        return [
            obj
            for name in dir(tvt_v2)
            if (obj := getattr(tvt_v2, name)) and isclass(obj) and issubclass(obj, tvt_v2.Transform)
        ]

    @classmethod
    def generate(cls, config: SubsetConfig) -> Compose:
        """Generate TorchVision transforms from the configuration."""
        if isinstance(config.transforms, Compose):
            return config.transforms

        input_size = getattr(config, "input_size", None)
        transforms = []
        for cfg_transform in config.transforms:
            if isinstance(cfg_transform, (dict, DictConfig)):
                if not cfg_transform.get("enable", True):  # Optional "enable: false" flag would remove the transform
                    continue
                cls._configure_input_size(cfg_transform, input_size)
            transform = cls._dispatch_transform(cfg_transform)
            transforms.append(transform)

        return Compose(transforms)

    @classmethod
    def _configure_input_size(cls, cfg_transform: dict[str, Any], input_size: int | tuple[int, int] | None) -> None:
        """Evaluate the input_size and replace the placeholder in the init_args.

        Input size should be specified as $(input_size). (e.g. $(input_size) * 0.5)
        Only simple multiplication or division evaluation is supported. For example,
        $(input_size) * -0.5    => supported
        $(input_size) * 2.1 / 3 => supported
        $(input_size) + 1       => not supported
        The function decides to pass tuple type or int type based on the type hint of the argument.
        float point values are rounded to int.
        """
        if input_size is not None:
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
        for key, val in cfg_transform.get("init_args", {}).items():
            if not (isinstance(val, str) and "$(input_size)" in val):
                continue
            if input_size is None:
                msg = (
                    f"{cfg_transform['class_path'].split('.')[-1]} initial argument has `$(input_size)`, "
                    "but input_size is set to None."
                )
                raise RuntimeError(msg)

            if model_cls is None:
                model_cls = import_object_from_module(cfg_transform["class_path"])

            available_types = typing.get_type_hints(model_cls.__init__).get(key)
            if available_types is None or check_type(_input_size, available_types):  # pass tuple[int, int]
                cfg_transform["init_args"][key] = cls._eval_input_size_str(
                    val.replace("$(input_size)", str(_input_size)),
                )
            elif check_type(_input_size[0], available_types):  # pass int
                cfg_transform["init_args"][key] = cls._eval_input_size_str(
                    val.replace("$(input_size)", str(_input_size[0])),
                )
            else:
                msg = f"{key} argument should be able to get int or tuple[int, int], but it can get {available_types}"
                raise RuntimeError(msg)

    @classmethod
    def _eval_input_size_str(cls, str_to_eval: str) -> tuple[int, ...] | int:
        """Safe eval function for _configure_input_size.

        The function is implemented for `_configure_input_size`, so implementation is aligned to it as below
        - Only multiplication or division evaluation are supported.
        - Only constant and tuple can be operand.
        - tuple is changed to numpy array before evaluation.
        - result value is rounded to int.
        """
        bin_ops = {
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        un_ops = {
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
                return un_ops[type(node.op)](operand)  # type: ignore[operator]
            msg = f"Bad syntax, {type(node)}. Available operations for calcualting input size are {available_ops}"
            raise SyntaxError(msg)

        ret = _eval(tree)
        if isinstance(ret, np.ndarray):
            return tuple(ret.round().astype(np.int32).tolist())
        return round(ret)

    @classmethod
    def _dispatch_transform(cls, cfg_transform: DictConfig | dict | tvt_v2.Transform) -> tvt_v2.Transform:
        if isinstance(cfg_transform, (DictConfig, dict)):
            transform = instantiate_class(args=(), init=cfg_transform)

        elif isinstance(cfg_transform, tvt_v2.Transform):
            transform = cfg_transform
        else:
            msg = (
                "TorchVisionTransformLib accepts only three types "
                "for config.transforms: DictConfig | dict | tvt_v2.Transform. "
                f"However, its type is {type(cfg_transform)}."
            )
            raise TypeError(msg)

        return transform


class RandomIoUCrop(tvt_v2.RandomIoUCrop):
    """Random IoU crop with the option to set probability.

    Args:
        min_scale (float, optional): the same as RandomIoUCrop. Defaults to 0.3.
        max_scale (float, optional): the same as RandomIoUCrop. Defaults to 1.
        min_aspect_ratio (float, optional): the same as RandomIoUCrop. Defaults to 0.5.
        max_aspect_ratio (float, optional): the same as RandomIoUCrop. Defaults to 2.
        sampler_options (list[float] | None, optional): the same as RandomIoUCrop. Defaults to None.
        trials (int, optional): the same as RandomIoUCrop. Defaults to 40.
        probability (float, optional): probability. Defaults to 1.0.
    """

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: list[float] | None = None,
        trials: int = 40,
        probability: float = 1.0,
    ):
        super().__init__(
            min_scale,
            max_scale,
            min_aspect_ratio,
            max_aspect_ratio,
            sampler_options,
            trials,
        )
        self.p = probability

    def __call__(self, *inputs: Any) -> Any:  # noqa: ANN401
        """Apply the transform to the given inputs."""
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)
