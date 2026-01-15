"""Unit tests for instance segmentation dataset."""

from __future__ import annotations

from unittest.mock import Mock

from datumaro.experimental import Dataset

from otx.data.dataset.instance_segmentation import OTXInstanceSegDataset


class TestOTXInstanceSegDataset:
    def setup_method(self):
        # Mock Datumaro experimental Dataset subset
        self.mock_dm_subset = Mock(spec=Dataset)
        self.mock_dm_subset.__len__ = Mock(return_value=5)
        self.mock_dm_subset.convert_to_schema = Mock(return_value=self.mock_dm_subset)

        # Mock schema attributes for label_info
        mock_schema = Mock()
        mock_attributes = {"label": Mock()}
        mock_attributes["label"].categories = Mock()
        mock_attributes["label"].categories.labels = ["person", "car", "dog"]
        mock_schema.attributes = mock_attributes
        self.mock_dm_subset.schema = mock_schema

        self.mock_transforms = Mock()

    def test_get_idx_list_per_classes_int_and_string(self):
        # Prepare items with multi-instance labels per sample
        mock_items = []
        for lbls in ([0, 1], [2], [1, 2], [0]):
            mock_item = Mock()
            mock_item.label.tolist.return_value = list(lbls)
            mock_items.append(mock_item)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = OTXInstanceSegDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        dataset.dm_subset.__len__ = Mock(return_value=4)

        # Integer mapping
        result_int = dataset.get_idx_list_per_classes()
        assert result_int == {0: [0, 3], 1: [0, 2], 2: [1, 2]}

        # String mapping
        # Reset side_effect for second pass
        self.mock_dm_subset.__getitem__.side_effect = mock_items
        result_str = dataset.get_idx_list_per_classes(use_string_label=True)
        assert result_str == {
            "person": [0, 3],
            "car": [0, 2],
            "dog": [1, 2],
        }
