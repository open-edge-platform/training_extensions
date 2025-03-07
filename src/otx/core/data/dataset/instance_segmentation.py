# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXInstanceSegDataset."""

from __future__ import annotations

import warnings
from collections import defaultdict
from functools import partial
from typing import Callable

import numpy as np
import torch
from datumaro import Bbox, Ellipse, Image, Polygon
from datumaro import Dataset as DmDataset
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegDataEntity
from otx.core.utils.mask_util import polygon_to_bitmap

from .base import OTXDataset, Transforms


class OTXInstanceSegDataset(OTXDataset):
    """OTXDataset class for instance segmentation.

    Args:
        dm_subset (DmDataset): The subset of the dataset.
        transforms (Transforms): Data transformations to be applied.
        include_polygons (bool): Flag indicating whether to include polygons in the dataset.
            If set to False, polygons will be converted to bitmaps, and bitmaps will be used for training.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(self, dm_subset: DmDataset, transforms: Transforms, include_polygons: bool, **kwargs) -> None:
        super().__init__(dm_subset, transforms, **kwargs)
        self.include_polygons = include_polygons

    def _get_item_impl(self, index: int) -> InstanceSegDataEntity | None:
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

        entity = InstanceSegDataEntity(
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
            labels=torch.as_tensor(labels),
            polygons=gt_polygons,
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect InstanceSegDataEntity into InstanceSegDataEntity in dataloader."""
        return partial(InstanceSegBatchDataEntity.collate_fn, stack_images=self.stack_images)
