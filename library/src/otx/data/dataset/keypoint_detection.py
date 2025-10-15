# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXKeypointDetectionDataset."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, List, Union

import numpy as np
import torch
from datumaro import AnnotationType, Bbox, Dataset, DatasetSubset, Image, Points
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import to_dtype, to_image

from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataItem
from otx.data.transform_libs.torchvision import Compose
from otx.types.image import ImageColorChannel
from otx.types.label import LabelInfo

from .base import OTXDataset

Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]


class OTXKeypointDetectionDataset(OTXDataset):
    """OTXDataset class for keypoint detection task."""

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        super().__init__(
            dm_subset,
            transforms,
            max_refetch,
            image_color_channel,
            stack_images,
            to_tv_image,
            data_format,
        )

        self.dm_subset = self._get_single_bbox_dataset(dm_subset)

        # arrow doesn't follow common coco convention, no need to fetch kp-specific labels
        if self.dm_subset.categories() and data_format != "arrow":
            kp_labels = self.dm_subset.categories()[AnnotationType.points][0].labels
            self.label_info = LabelInfo(
                label_names=kp_labels,
                label_groups=[],
                label_ids=[str(i) for i in range(len(kp_labels))],
            )

    def _get_single_bbox_dataset(self, dm_subset: DatasetSubset) -> Dataset:
        """Method for splitting dataset items into multiple items for each bbox/keypoint."""
        dm_items = []
        for item in dm_subset:
            new_items = defaultdict(list)
            for ann in item.annotations:
                if isinstance(ann, (Bbox, Points)):
                    new_items[ann.id].append(ann)
            for ann_id, anns in new_items.items():
                available_types = []
                for ann in anns:
                    if isinstance(ann, Bbox) and (ann.w <= 0 or ann.h <= 0):
                        continue
                    if isinstance(ann, Points) and max(ann.points) <= 0:
                        continue
                    available_types.append(ann.type)
                if AnnotationType.points not in available_types:
                    continue
                dm_items.append(item.wrap(id=item.id + "_" + str(ann_id), annotations=anns))
        if len(dm_items) == 0:
            msg = "No keypoints found in the dataset. Please, check dataset annotations."
            raise ValueError(msg)
        return Dataset.from_iterable(dm_items, categories=self.dm_subset.categories())

    def _get_item_impl(self, index: int) -> OTXDataItem | None:
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []  # This should be assigned form item
        img_data, img_shape, _ = self._get_img_data_and_shape(img)

        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]
        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        keypoint_anns = [ann for ann in item.annotations if isinstance(ann, Points)]
        keypoints = (
            np.stack([ann.points for ann in keypoint_anns], axis=0).astype(np.float32)
            if len(keypoint_anns) > 0
            else np.zeros((0, len(self.label_info.label_names) * 2), dtype=np.float32)
        ).reshape(-1, 2)

        keypoints_visible = (
            (np.array([ann.visibility for ann in keypoint_anns]) > 1).reshape(-1).astype(np.int8)
            if len(keypoint_anns) > 0 and hasattr(keypoint_anns[0], "visibility")
            else np.minimum(1, keypoints)[..., 0]
        )
        keypoints = np.hstack((keypoints, keypoints_visible.reshape(-1, 1)))

        entity = OTXDataItem(
            image=to_dtype(to_image(img_data), torch.float32),
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
            ),
            label=torch.as_tensor([ann.label for ann in bbox_anns], dtype=torch.long),
            keypoints=torch.as_tensor(keypoints, dtype=torch.float32),
        )

        return self._apply_transforms(entity)  # type: ignore[return-value]
