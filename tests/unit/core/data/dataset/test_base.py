from unittest import mock

import numpy as np
import pytest
from datumaro.components.media import Image
from otx.core.data.dataset.base import OTXDataset


class TestOTXDataset:
    @pytest.fixture()
    def mock_image(self) -> Image:
        img = mock.Mock(spec=Image)
        img.data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img.path = "test_path"
        return img

    @pytest.fixture()
    def mock_mem_cache_handler(self):
        mem_cache_handler = mock.MagicMock()
        mem_cache_handler.frozen = False
        return mem_cache_handler

    @pytest.fixture()
    def otx_dataset(self, mock_mem_cache_handler):
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
            mem_cache_handler=mock_mem_cache_handler,
            mem_cache_img_max_size=None,
        )

    def test_get_img_data_and_shape_no_cache(self, otx_dataset, mock_image, mock_mem_cache_handler):
        mock_mem_cache_handler.get.return_value = (None, None)
        img_data, img_shape, roi_meta = otx_dataset._get_img_data_and_shape(mock_image)
        assert img_data.shape == (10, 10, 3)
        assert img_shape == (10, 10)
        assert roi_meta is None

    def test_get_img_data_and_shape_with_cache(self, otx_dataset, mock_image, mock_mem_cache_handler):
        mock_mem_cache_handler.get.return_value = (np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8), None)
        img_data, img_shape, roi_meta = otx_dataset._get_img_data_and_shape(mock_image)
        assert img_data.shape == (10, 10, 3)
        assert img_shape == (10, 10)
        assert roi_meta is None

    def test_get_img_data_and_shape_with_roi(self, otx_dataset, mock_image, mock_mem_cache_handler):
        roi = {"shape": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}}
        mock_mem_cache_handler.get.return_value = (None, None)
        img_data, img_shape, roi_meta = otx_dataset._get_img_data_and_shape(mock_image, roi)
        assert img_data.shape == (8, 8, 3)
        assert img_shape == (8, 8)
        assert roi_meta == {"x1": 1, "y1": 1, "x2": 9, "y2": 9, "orig_image_shape": (10, 10)}

    def test_cache_img_no_resize(self, otx_dataset):
        img_data = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        key = "test_key"

        cached_img = otx_dataset._cache_img(key, img_data)

        assert np.array_equal(cached_img, img_data)
        otx_dataset.mem_cache_handler.put.assert_called_once_with(key=key, data=img_data, meta=None)

    def test_cache_img_with_resize(self, otx_dataset, mock_mem_cache_handler):
        otx_dataset.mem_cache_img_max_size = (100, 100)
        img_data = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        key = "test_key"

        cached_img = otx_dataset._cache_img(key, img_data)

        assert cached_img.shape == (100, 100, 3)
        mock_mem_cache_handler.put.assert_called_once()
        assert mock_mem_cache_handler.put.call_args[1]["data"].shape == (100, 100, 3)

    def test_cache_img_no_max_size(self, otx_dataset, mock_mem_cache_handler):
        otx_dataset.mem_cache_img_max_size = None
        img_data = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        key = "test_key"

        cached_img = otx_dataset._cache_img(key, img_data)

        assert np.array_equal(cached_img, img_data)
        mock_mem_cache_handler.put.assert_called_once_with(key=key, data=img_data, meta=None)

    def test_cache_img_frozen_handler(self, otx_dataset, mock_mem_cache_handler):
        mock_mem_cache_handler.frozen = True
        img_data = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        key = "test_key"

        cached_img = otx_dataset._cache_img(key, img_data)

        assert np.array_equal(cached_img, img_data)
        mock_mem_cache_handler.put.assert_not_called()
