# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXDetectionDataset."""

from __future__ import annotations

import torch
from datumaro import Bbox, Image
from torchvision.transforms.v2.functional import to_dtype, to_image

from otx.core.data.entity.base import ImageInfo
from otx.data import TorchDataItem
from otx.data.resolvers.torch import DatumaroResolver

from .base import OTXDataset


class OTXDetectionDataset(OTXDataset):
    """OTXDataset class for detection task."""

    def _get_item_impl(self, index: int) -> TorchDataItem | None:
        item = self.dm_subset[index]
        img = item.media_as(Image)
        img_data, img_shape, _ = self._get_img_data_and_shape(img)

        entity = TorchDataItem(
            image=to_dtype(to_image(img_data), torch.float32),
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
            ),
            bboxes=DatumaroResolver.resolve_bbox(item, img_shape),
            label=DatumaroResolver.resolve_label(item, Bbox),
        )

        return self._apply_transforms(entity)
