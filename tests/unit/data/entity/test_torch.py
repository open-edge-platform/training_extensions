# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of detection data entity."""

import torch
from torch import LongTensor
from torchvision import tv_tensors

from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataBatch, OTXDataItem


class TestOTXDataItem:
    def test_collate_fn(self) -> None:
        data_entities = [
            OTXDataItem(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                bboxes=tv_tensors.BoundingBoxes(
                    data=torch.Tensor([0, 0, 50, 50]),
                    format="xywh",
                    canvas_size=(224, 224),
                ),
                label=LongTensor([1]),
            ),
            OTXDataItem(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                bboxes=tv_tensors.BoundingBoxes(
                    data=torch.Tensor([0, 0, 50, 50]),
                    format="xywh",
                    canvas_size=(224, 224),
                ),
                label=LongTensor([1]),
            ),
            OTXDataItem(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                bboxes=tv_tensors.BoundingBoxes(
                    data=torch.Tensor([0, 0, 50, 50]),
                    format="xywh",
                    canvas_size=(224, 224),
                ),
                label=LongTensor([1]),
            ),
        ]

        data_batch = OTXDataItem.collate_fn(data_entities)
        assert len(data_batch.imgs_info) == len(data_batch.images)
        assert data_batch.__class__ == OTXDataBatch
        for field in OTXDataBatch.__dataclass_fields__:
            assert hasattr(data_batch, field), f"Field {field} is missing in the collated batch"
