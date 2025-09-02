# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for classification_new dataset."""

from __future__ import annotations

from unittest.mock import Mock

from datumaro.experimental import Dataset

from otx.data.dataset.classification_new import OTXMulticlassClsDataset
from otx.data.entity.sample import ClassificationSample


class TestOTXMulticlassClsDataset:
    """Test OTXMulticlassClsDataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dm_subset = Mock(spec=Dataset)
        self.mock_dm_subset.__len__ = Mock(return_value=10)

        # Mock schema attributes for label_info
        mock_schema = Mock()
        mock_attributes = {"label": Mock()}
        mock_attributes["label"].categories = Mock()
        mock_schema.attributes = mock_attributes
        self.mock_dm_subset.schema = mock_schema

        self.mock_transforms = Mock()

    def test_init_sets_sample_type(self):
        """Test that initialization sets sample_type to ClassificationSample."""
        dataset = OTXMulticlassClsDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            data_format="arrow",
        )

        assert dataset.sample_type == ClassificationSample

    def test_get_idx_list_per_classes_single_class(self):
        """Test get_idx_list_per_classes with single class."""
        # Mock dataset items with labels
        mock_items = []
        for i in range(5):
            mock_item = Mock()
            mock_item.label.item.return_value = 0  # All items have label 0
            mock_items.append(mock_item)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = OTXMulticlassClsDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            data_format="arrow",
        )

        # Override length for this test
        dataset.dm_subset.__len__ = Mock(return_value=5)

        result = dataset.get_idx_list_per_classes()

        expected = {0: [0, 1, 2, 3, 4]}
        assert result == expected
