# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXInstanceSegDataset."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch
from datumaro import Bbox, Ellipse, Image, Polygon
from torchvision import tv_tensors

from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataItem
from otx.data.utils.structures.mask.mask_util import polygon_to_bitmap
from otx.types import OTXTaskType
from otx.types.image import ImageColorChannel

from .base import OTXDataset, Transforms

if TYPE_CHECKING:
    from datumaro import Dataset as DmDataset


class OTXInstanceSegDataset(OTXDataset):
    """Dataset class for instance segmentation tasks in OTX.

    This class handles loading images and their instance segmentation annotations,
    supporting polygons, bounding boxes, and ellipses. Annotations can be kept as polygons
    or converted to bitmaps for training, depending on the `include_polygons` flag.

    Args:
        dm_subset (DmDataset): The subset of the dataset to use.
        transforms (Transforms, optional): Data transformations to be applied.
        task_type (OTXTaskType, optional): The task type. Defaults to INSTANCE_SEGMENTATION.
        max_refetch (int, optional): Maximum number of times to refetch data. Defaults to 1000.
        image_color_channel (ImageColorChannel, optional): Image color channel format. Defaults to RGB.
        stack_images (bool, optional): Whether to stack images. Defaults to True.
        to_tv_image (bool, optional): Whether to convert images to torchvision format. Defaults to True.
        data_format (str, optional): Data format string. Defaults to "".
        include_polygons (bool, optional): If True, polygons are included in the dataset.
            If False, polygons are converted to bitmaps for training. Defaults to False.
    """

    def __init__(
        self,
        dm_subset: DmDataset,
        transforms: Transforms | None = None,
        task_type: OTXTaskType = OTXTaskType.INSTANCE_SEGMENTATION,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
        include_polygons: bool = False,
    ) -> None:
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
            image_color_channel=image_color_channel,
            stack_images=stack_images,
            to_tv_image=to_tv_image,
            data_format=data_format,
        )
        self.include_polygons = include_polygons
        self._task_type = task_type

    def _get_item_impl(self, index: int) -> OTXDataItem | None:
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []
        img_data, img_shape, _ = self._get_img_data_and_shape(img)

        anno_collection: dict[str, list] = defaultdict(list)
        for anno in item.annotations:
            anno_collection[anno.__class__.__name__].append(anno)

        gt_bboxes, gt_labels, gt_masks, gt_polygons = [], [], [], []

        # TODO(Eugene): https://jira.devtools.intel.com/browse/CVS-159363
        # Temporary solution to handle multiple annotation types.
        # Ideally, we should pre-filter annotations during initialization of the dataset.
        if Polygon.__name__ in anno_collection:  # Polygon for InstSeg has higher priority
            for poly in anno_collection[Polygon.__name__]:
                bbox = Bbox(*poly.get_bbox()).points
                gt_bboxes.append(bbox)
                gt_labels.append(poly.label)

                if self.include_polygons:
                    gt_polygons.append(poly)
                else:
                    gt_masks.append(polygon_to_bitmap([poly], *img_shape)[0])
        elif Bbox.__name__ in anno_collection:
            bboxes = anno_collection[Bbox.__name__]
            gt_bboxes = [ann.points for ann in bboxes]
            gt_labels = [ann.label for ann in bboxes]
            for box in bboxes:
                poly = Polygon(box.as_polygon())
                if self.include_polygons:
                    gt_polygons.append(poly)
                else:
                    gt_masks.append(polygon_to_bitmap([poly], *img_shape)[0])
        elif Ellipse.__name__ in anno_collection:
            for ellipse in anno_collection[Ellipse.__name__]:
                bbox = Bbox(*ellipse.get_bbox()).points
                gt_bboxes.append(bbox)
                gt_labels.append(ellipse.label)
                poly = Polygon(ellipse.as_polygon(num_points=10))
                if self.include_polygons:
                    gt_polygons.append(poly)
                else:
                    gt_masks.append(polygon_to_bitmap([poly], *img_shape)[0])
        else:
            warnings.warn(f"No valid annotations found for image {item.id}!", stacklevel=2)

        bboxes = np.stack(gt_bboxes, dtype=np.float32, axis=0) if gt_bboxes else np.empty((0, 4))
        masks = np.stack(gt_masks, axis=0) if gt_masks else np.zeros((0, *img_shape), dtype=bool)

        labels = np.array(gt_labels, dtype=np.int64)

        entity = OTXDataItem(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
                dtype=torch.float32,
            ),
            masks=tv_tensors.Mask(masks, dtype=torch.uint8),
            label=torch.as_tensor(labels, dtype=torch.long),
            polygons=gt_polygons if len(gt_polygons) > 0 else None,
        )

        return self._apply_transforms(entity)  # type: ignore[return-value]

    @property
    def task_type(self) -> OTXTaskType:
        """OTX Task Type for the dataset.

        Returns:
            OTXTaskType: The instance segmentation task type.
        """
        return self._task_type
