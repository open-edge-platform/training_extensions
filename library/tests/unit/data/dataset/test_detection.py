# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for detection dataset."""

from __future__ import annotations

from unittest.mock import Mock

from datumaro.experimental import Dataset

from otx.data.dataset.detection import OTXDetectionDataset
from otx.data.entity.sample import DetectionSample


class TestOTXDetectionDataset:
    """Test OTXDetectionDataset class."""

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

    def test_init_sets_sample_type(self):
        """Test that initialization sets sample_type to DetectionSample."""
        dataset = OTXDetectionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            data_format="arrow",
        )

        assert dataset.sample_type == DetectionSample

    def test_get_idx_list_per_classes_multiple_classes_per_item(self):
        """Test get_idx_list_per_classes with multiple classes per item."""
        # Mock dataset items with multiple labels per item
        mock_items = []
        # Item 0: classes [0, 1]
        mock_item0 = Mock()
        mock_item0.label.tolist.return_value = [0, 1]
        mock_items.append(mock_item0)

        # Item 1: class [1]
        mock_item1 = Mock()
        mock_item1.label.tolist.return_value = [1]
        mock_items.append(mock_item1)

        # Item 2: classes [0, 2]
        mock_item2 = Mock()
        mock_item2.label.tolist.return_value = [0, 2]
        mock_items.append(mock_item2)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = OTXDetectionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            data_format="arrow",
        )

        # Override length for this test
        dataset.dm_subset.__len__ = Mock(return_value=3)

        result = dataset.get_idx_list_per_classes()

        expected = {
            0: [0, 2],  # Items 0 and 2 contain class 0
            1: [0, 1],  # Items 0 and 1 contain class 1
            2: [2],  # Item 2 contains class 2
        }
        assert result == expected

    def test_get_idx_list_per_classes_string_labels(self):
        """Test get_idx_list_per_classes with use_string_label=True."""
        # Mock dataset items with multiple labels per item
        mock_items = []
        # Item 0: classes [0, 1]
        mock_item0 = Mock()
        mock_item0.label.tolist.return_value = [0, 1]
        mock_items.append(mock_item0)

        # Item 1: class [1]
        mock_item1 = Mock()
        mock_item1.label.tolist.return_value = [1]
        mock_items.append(mock_item1)

        # Item 2: classes [0, 2]
        mock_item2 = Mock()
        mock_item2.label.tolist.return_value = [0, 2]
        mock_items.append(mock_item2)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = OTXDetectionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            data_format="arrow",
        )

        # Override length for this test
        dataset.dm_subset.__len__ = Mock(return_value=3)

        result = dataset.get_idx_list_per_classes(use_string_label=True)

        expected = {
            "class_0": [0, 2],
            "class_1": [0, 1],
            "class_2": [2],
        }
        assert result == expected
