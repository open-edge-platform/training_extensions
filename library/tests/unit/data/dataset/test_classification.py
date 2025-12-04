# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for classification_new dataset."""

from __future__ import annotations

from unittest.mock import Mock

from datumaro.experimental import Dataset

from otx.data.dataset.classification import OTXMulticlassClsDataset


class TestOTXMulticlassClsDataset:
    """Test OTXMulticlassClsDataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dm_subset = Mock(spec=Dataset)
        self.mock_dm_subset.__len__ = Mock(return_value=10)
        self.mock_dm_subset.convert_to_schema = Mock(return_value=self.mock_dm_subset)

        # Mock schema attributes for label_info
        mock_schema = Mock()
        mock_attributes = {"label": Mock()}
        mock_attributes["label"].categories = Mock()
        # Configure labels to be a list with proper length support
        mock_attributes["label"].categories.labels = ["class_0", "class_1", "class_2"]
        mock_schema.attributes = mock_attributes
        self.mock_dm_subset.schema = mock_schema

        self.mock_transforms = Mock()

    def test_get_idx_list_per_classes_single_class(self):
        """Test get_idx_list_per_classes with single class."""
        # Mock dataset items with labels
        mock_items = []
        for _ in range(5):
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

    def test_get_idx_list_per_classes_string_labels(self):
        """Test get_idx_list_per_classes with use_string_label=True."""
        # Create three items with labels 0, 1, 2
        mock_items = []
        for lbl in [0, 1, 2, 1, 0]:
            mock_item = Mock()
            mock_item.label.item.return_value = lbl
            mock_items.append(mock_item)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = OTXMulticlassClsDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            data_format="arrow",
        )

        # Override length for this test
        dataset.dm_subset.__len__ = Mock(return_value=5)

        result = dataset.get_idx_list_per_classes(use_string_label=True)

        expected = {
            "class_0": [0, 4],
            "class_1": [1, 3],
            "class_2": [2],
        }
        assert result == expected
