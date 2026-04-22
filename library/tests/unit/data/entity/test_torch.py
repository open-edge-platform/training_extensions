# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of sample batch data entity."""

from unittest.mock import Mock

import torch
from torch import LongTensor
from torchvision import tv_tensors

from getitune.data.dataset.base import _default_collate_fn
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import BaseSample, SampleBatch


class TestCollateFn:
    def test_collate_fn(self) -> None:
        """Test _default_collate_fn function."""
        # Create mock samples with required attributes
        samples = []
        for i in range(3):
            sample = Mock(spec=BaseSample)
            # Use float32 images since _default_collate_fn expects tensors
            sample.image = tv_tensors.Image(torch.randn(3, 224, 224))
            sample.img_info = ImageInfo(img_idx=i, img_shape=(224, 224), ori_shape=(224, 224))
            sample.bboxes = tv_tensors.BoundingBoxes(
                data=torch.Tensor([0, 0, 50, 50]),
                format="xywh",
                canvas_size=(224, 224),
            )
            sample.label = LongTensor([1])
            sample.masks = None
            sample.keypoints = None
            samples.append(sample)

        data_batch = _default_collate_fn(samples)
        assert len(data_batch.imgs_info) == len(data_batch.images)
        assert isinstance(data_batch, SampleBatch)
        for field in SampleBatch.__dataclass_fields__:
            assert hasattr(data_batch, field), f"Field {field} is missing in the collated batch"
