# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import numpy as np
import pytest
from datumaro.components.media import Image

from otx.data.dataset.base import OTXDataset


class TestOTXDataset:
    @pytest.fixture()
    def mock_image(self) -> Image:
        img = mock.Mock(spec=Image)
        img.data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img.path = "test_path"
        return img

    @pytest.fixture()
    def otx_dataset(self):
        class MockOTXDataset(OTXDataset):
            def _get_item_impl(self, idx: int) -> None:
                return None

            @property
            def collate_fn(self) -> None:
                return None

        dm_subset = mock.Mock()
        dm_subset.categories = mock.MagicMock()
        dm_subset.categories.return_value = None

        return MockOTXDataset(
            dm_subset=dm_subset,
            transforms=None,
        )

    def test_get_img_data_and_shape(self, otx_dataset, mock_image):
        img_data, img_shape, roi_meta = otx_dataset._get_img_data_and_shape(mock_image)
        assert img_data.shape == (10, 10, 3)
        assert img_shape == (10, 10)
        assert roi_meta is None

    def test_get_img_data_and_shape_with_roi(self, otx_dataset, mock_image):
        roi = {"shape": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}}
        img_data, img_shape, roi_meta = otx_dataset._get_img_data_and_shape(mock_image, roi)
        assert img_data.shape == (8, 8, 3)
        assert img_shape == (8, 8)
        assert roi_meta == {"x1": 1, "y1": 1, "x2": 9, "y2": 9, "orig_image_shape": (10, 10)}
