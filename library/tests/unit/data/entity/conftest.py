# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for unit tests of data entities."""

import numpy as np
import pytest
import torch
from torchvision import tv_tensors

from otx.data.entity import ImageInfo, OTXDataItem


@pytest.fixture()
def fxt_numpy_data_entity() -> OTXDataItem:
    return OTXDataItem(
        image=np.ndarray((10, 10, 3), dtype=np.float32),
        img_info=ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
    )


@pytest.fixture()
def fxt_torchvision_data_entity() -> OTXDataItem:
    return OTXDataItem(
        image=tv_tensors.Image(torch.randn(3, 10, 10), dtype=torch.float32),
        img_info=ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
    )
