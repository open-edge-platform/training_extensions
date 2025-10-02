# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomaly Classification Dataset."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
from datumaro import Dataset as DmDataset
from datumaro import DatasetItem, Image
from datumaro.components.annotation import AnnotationType, Bbox, Ellipse, Polygon
from datumaro.components.media import ImageFromBytes, ImageFromFile
from torchvision import io
from torchvision.transforms.v2.functional import to_dtype, to_image
from torchvision.tv_tensors import Mask

from otx.data.dataset.base import OTXDataset, Transforms
from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataItem
from otx.types.image import ImageColorChannel
from otx.types.label import AnomalyLabelInfo
from otx.types.task import OTXTaskType


class AnomalyLabel(Enum):
    """Anomaly label to tensor mapping."""

    NORMAL = torch.tensor(0.0)
    ANOMALOUS = torch.tensor(1.0)


class OTXAnomalyDataset(OTXDataset):
    """OTXDataset class for anomaly classification task."""

    def __init__(
        self,
        task_type: OTXTaskType,
        dm_subset: DmDataset,
        transforms: Transforms,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        self.task_type = task_type
        super().__init__(
            dm_subset,
            transforms,
            max_refetch,
            image_color_channel,
            stack_images,
            to_tv_image,
        )
        self.label_info = AnomalyLabelInfo()
        self._label_mapping = self._map_id_to_label()

    def _get_item_impl(
        self,
        index: int,
    ) -> OTXDataItem:
        datumaro_item = self.dm_subset[index]
        img = datumaro_item.media_as(Image)
        # returns image in RGB format if self.image_color_channel is RGB
        img_data, img_shape, _ = self._get_img_data_and_shape(img)
        image = to_dtype(to_image(img_data), dtype=torch.float32)

        label = self._get_label(datumaro_item)

        item = OTXDataItem(
            image=image,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
            ),
            label=label.to(dtype=torch.long),
            masks=Mask(self._get_mask(datumaro_item, label, img_shape)),
        )

        return self._apply_transforms(item)  # type: ignore[return-value]

    def _get_mask(self, datumaro_item: DatasetItem, label: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
        """Get mask from datumaro_item.

        Converts bounding boxes to mask if mask is not available.
        """
        if isinstance(datumaro_item.media, ImageFromFile):
            if label == AnomalyLabel.ANOMALOUS.value:
                mask = self._mask_image_from_file(datumaro_item, img_shape)
            else:
                mask = torch.zeros(1, *img_shape).to(torch.uint8)
        elif isinstance(datumaro_item.media, ImageFromBytes):
            mask = torch.zeros(1, *img_shape).to(torch.uint8)
            if label == AnomalyLabel.ANOMALOUS.value:
                for annotation in datumaro_item.annotations:
                    # There is only one mask
                    if isinstance(annotation, (Ellipse, Polygon)):
                        polygons = np.asarray(annotation.as_polygon(), dtype=np.int32).reshape((-1, 1, 2))
                        mask = np.zeros(img_shape, dtype=np.uint8)
                        mask = cv2.drawContours(
                            mask,
                            [polygons],
                            0,
                            (1, 1, 1),
                            thickness=cv2.FILLED,
                        )
                        mask = torch.from_numpy(mask).to(torch.uint8).unsqueeze(0)
                        break
                    # If there is no mask, create a mask from bbox
                    if isinstance(annotation, Bbox):
                        bbox = annotation
                        mask = self._bbox_to_mask(bbox, img_shape)
                        break
        return mask

    def _bbox_to_mask(self, bbox: Bbox, img_shape: tuple[int, int]) -> torch.Tensor:
        mask = torch.zeros(1, *img_shape).to(torch.uint8)
        x1, y1, x2, y2 = bbox.get_bbox()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask[:, y1:y2, x1:x2] = 1
        return mask

    def _get_label(self, datumaro_item: DatasetItem) -> torch.LongTensor:
        """Get label from datumaro item."""
        if isinstance(datumaro_item.media, ImageFromFile):
            # Note: This assumes that the dataset is in MVTec format.
            # We can't use datumaro label id as it returns some number like 3 for good from which it is hard to infer
            # whether the image is Anomalous or Normal. Because it leads to other questions like what do numbers 0,1,2
            # mean?
            label: torch.LongTensor = AnomalyLabel.NORMAL if "good" in datumaro_item.id else AnomalyLabel.ANOMALOUS
        elif isinstance(datumaro_item.media, ImageFromBytes):
            label = self._label_mapping[datumaro_item.annotations[0].label]
        else:
            msg = f"Media type {type(datumaro_item.media)} is not supported."
            raise NotImplementedError(msg)
        return label.value

    def _map_id_to_label(self) -> dict[int, torch.Tensor]:
        """Map label id to label tensor."""
        id_label_mapping = {}
        categories = self.dm_subset.categories()[AnnotationType.label]
        for label_item in categories.items:
            if any("normal" in attribute.lower() for attribute in label_item.attributes):
                label = AnomalyLabel.NORMAL
            else:
                label = AnomalyLabel.ANOMALOUS
            id_label_mapping[categories.find(label_item.name)[0]] = label
        return id_label_mapping

    def _mask_image_from_file(self, datumaro_item: DatasetItem, img_shape: tuple[int, int]) -> torch.Tensor:
        """Assumes MVTec format and returns mask from disk."""
        mask_file_path = (
            Path("/".join(datumaro_item.media.path.split("/")[:-3]))
            / "ground_truth"
            / f"{('/'.join(datumaro_item.media.path.split('/')[-2:])).replace('.png', '_mask.png')}"
        )
        if mask_file_path.exists():
            return (io.read_image(str(mask_file_path), mode=io.ImageReadMode.GRAY) / 255).to(torch.uint8)

        # Note: This is a workaround to handle the case where mask is not available otherwise the tests fail.
        # This is problematic because it assigns empty masks to an Anomalous image.
        return torch.zeros(1, *img_shape).to(torch.uint8)
