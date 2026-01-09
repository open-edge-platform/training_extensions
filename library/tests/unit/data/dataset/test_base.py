# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for base_new OTXDataset."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch
from datumaro.experimental import Dataset

from otx.data.dataset.base import OTXDataset, _default_collate_fn
from otx.data.entity.sample import OTXSample
from otx.data.entity.torch.torch import OTXDataBatch


class TestDefaultCollateFn:
    """Test _default_collate_fn function."""

    def test_collate_with_torch_tensors(self):
        """Test collating items with torch tensor images."""
        # Create mock samples with torch tensor images
        sample1 = Mock(spec=OTXSample)
        sample1.image = torch.randn(3, 224, 224)
        sample1.label = torch.tensor(0)
        sample1.masks = None
        sample1.bboxes = None
        sample1.keypoints = None
        sample1.polygons = None
        sample1.img_info = None

        sample2 = Mock(spec=OTXSample)
        sample2.image = torch.randn(3, 224, 224)
        sample2.label = torch.tensor(1)
        sample2.masks = None
        sample2.bboxes = None
        sample2.keypoints = None
        sample2.polygons = None
        sample2.img_info = None

        items = [sample1, sample2]
        result = _default_collate_fn(items)

        assert isinstance(result, OTXDataBatch)
        assert result.batch_size == 2
        assert isinstance(result.images, torch.Tensor)
        assert result.images.shape == (2, 3, 224, 224)
        assert result.images.dtype == torch.float32
        assert result.labels == [torch.tensor(0), torch.tensor(1)]

    def test_collate_with_different_image_shapes(self):
        """Test collating items with different image shapes."""
        sample1 = Mock(spec=OTXSample)
        sample1.image = torch.randn(3, 224, 224)
        sample1.label = None
        sample1.masks = None
        sample1.bboxes = None
        sample1.keypoints = None
        sample1.polygons = None
        sample1.img_info = None

        sample2 = Mock(spec=OTXSample)
        sample2.image = torch.randn(3, 256, 256)
        sample2.label = None
        sample2.masks = None
        sample2.bboxes = None
        sample2.keypoints = None
        sample2.polygons = None
        sample2.img_info = None

        items = [sample1, sample2]
        result = _default_collate_fn(items)

        # When shapes are different, should return list instead of stacked tensor
        assert isinstance(result.images, list)
        assert len(result.images) == 2
        assert result.labels is None


class TestOTXDataset:
    """Test OTXDataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dm_subset = Mock(spec=Dataset)
        self.mock_dm_subset.__len__ = Mock(return_value=100)

        # Mock schema attributes for label_info
        mock_schema = Mock()
        mock_attributes = {"label": Mock()}
        mock_attributes["label"].categories = Mock()
        # Configure labels to be a list with proper length support
        mock_attributes["label"].categories.labels = ["class_0", "class_1", "class_2"]
        mock_schema.attributes = mock_attributes
        self.mock_dm_subset.schema = mock_schema

        self.mock_transforms = Mock()

    def test_apply_transforms_with_compose(self):
        """Test _apply_transforms with Compose transforms."""
        from otx.data.transform_libs.torchvision import Compose

        mock_compose = Mock(spec=Compose)
        mock_entity = Mock(spec=OTXSample)
        mock_result = Mock()
        mock_compose.return_value = mock_result

        dataset = OTXDataset(
            dm_subset=self.mock_dm_subset,
            transforms=mock_compose,
            data_format="arrow",
            to_tv_image=True,
        )

        result = dataset._apply_transforms(mock_entity)

        mock_compose.assert_called_once_with(mock_entity)
        assert result == mock_result

    def test_apply_transforms_with_callable(self):
        """Test _apply_transforms with callable transform."""
        mock_transform = Mock()
        mock_entity = Mock(spec=OTXSample)
        mock_result = Mock()
        mock_transform.return_value = mock_result

        dataset = OTXDataset(
            dm_subset=self.mock_dm_subset,
            transforms=mock_transform,
            data_format="arrow",
        )

        result = dataset._apply_transforms(mock_entity)

        mock_transform.assert_called_once_with(mock_entity)
        assert result == mock_result

    def test_apply_transforms_with_list(self):
        """Test _apply_transforms with list of transforms."""
        transform1 = Mock()
        transform2 = Mock()

        mock_entity = Mock(spec=OTXSample)
        intermediate_result = Mock()
        final_result = Mock()

        transform1.return_value = intermediate_result
        transform2.return_value = final_result

        dataset = OTXDataset(
            dm_subset=self.mock_dm_subset,
            transforms=[transform1, transform2],
            data_format="arrow",
        )

        result = dataset._apply_transforms(mock_entity)

        transform1.assert_called_once_with(mock_entity)
        transform2.assert_called_once_with(intermediate_result)
        assert result == final_result

    def test_apply_transforms_with_list_returns_none(self):
        """Test _apply_transforms with list that returns None."""
        transform1 = Mock()
        transform2 = Mock()

        mock_entity = Mock(spec=OTXSample)
        transform1.return_value = None  # First transform returns None

        dataset = OTXDataset(
            dm_subset=self.mock_dm_subset,
            transforms=[transform1, transform2],
            data_format="arrow",
        )

        result = dataset._apply_transforms(mock_entity)

        transform1.assert_called_once_with(mock_entity)
        transform2.assert_not_called()  # Should not be called since first returned None
        assert result is None

    def test_iterable_transforms_with_non_list(self):
        """Test _iterable_transforms with non-list iterable raises TypeError."""
        dataset = OTXDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            data_format="arrow",
        )

        mock_entity = Mock(spec=OTXSample)
        dataset.transforms = "not_a_list"  # String is iterable but not a list

        with pytest.raises(TypeError):
            dataset._iterable_transforms(mock_entity)

    def test_getitem_success(self):
        """Test __getitem__ with successful retrieval."""
        mock_item = Mock()
        self.mock_dm_subset.__getitem__ = Mock(return_value=mock_item)

        mock_transformed_item = Mock(spec=OTXSample)

        dataset = OTXDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            data_format="arrow",
        )

        with patch.object(dataset, "_apply_transforms", return_value=mock_transformed_item):
            result = dataset[5]

            self.mock_dm_subset.__getitem__.assert_called_once_with(5)
            assert result == mock_transformed_item

    def test_getitem_with_refetch(self):
        """Test __getitem__ with failed first attempt requiring refetch."""
        mock_item = Mock()
        self.mock_dm_subset.__getitem__ = Mock(return_value=mock_item)

        dataset = OTXDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            data_format="arrow",
            max_refetch=2,
        )

        mock_transformed_item = Mock(spec=OTXSample)

        # First call returns None, second returns valid item
        with patch.object(dataset, "_apply_transforms", side_effect=[None, mock_transformed_item]):
            result = dataset[5]

            assert result == mock_transformed_item
            assert dataset._apply_transforms.call_count == 2

    def test_collate_fn_property(self):
        """Test collate_fn property returns _default_collate_fn."""
        dataset = OTXDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            data_format="arrow",
        )

        assert dataset.collate_fn == _default_collate_fn
