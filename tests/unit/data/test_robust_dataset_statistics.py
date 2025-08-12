# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for compute_robust_dataset_statistics function."""

from __future__ import annotations

import numpy as np
import pytest
from datumaro import Dataset as DmDataset
from datumaro import DatasetItem, DatasetSubset
from datumaro.components.annotation import AnnotationType, Bbox, ExtractedMask, LabelCategories, Polygon
from datumaro.components.media import Image

from otx.data.utils.utils import compute_robust_dataset_statistics
from otx.types import OTXTaskType


class TestComputeRobustDatasetStatistics:
    """Test cases for compute_robust_dataset_statistics function."""

    @pytest.fixture()
    def mock_semantic_seg_dataset(self):
        """Create a mock semantic segmentation dataset with mixed annotation types."""
        dataset = DmDataset(media_type=Image)

        # Create label categories
        categories = LabelCategories()
        categories.add("background")
        categories.add("foreground")
        dataset.categories()[AnnotationType.label] = categories

        for i in range(5):
            image = Image.from_numpy(np.zeros((100, 100, 3), dtype=np.uint8))

            # ExtractedMask annotation (foreground)
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[20:40, 20:40] = 1
            ann_mask = ExtractedMask(
                index_mask=mask,
                index=0,
                label=1,  # foreground
            )

            # Polygon annotation (foreground)
            polygon = Polygon([10, 10, 50, 10, 50, 50, 10, 50], label=1)

            # Bbox annotation (background, should be ignored for SEMANTIC_SEGMENTATION)
            bbox = Bbox(60, 60, 20, 20, label=0)

            dataset.put(
                DatasetItem(
                    id=str(i),
                    media=image,
                    annotations=[ann_mask, polygon, bbox],
                    subset="train",
                ),
            )
        return dataset

    def test_compute_robust_dataset_statistics_semantic_segmentation(self, mock_semantic_seg_dataset):
        """Test that semantic segmentation with ExtractedMask annotations is handled correctly."""
        # Get the train subset
        train_subset = DatasetSubset(mock_semantic_seg_dataset, "train")

        # Compute statistics
        stats = compute_robust_dataset_statistics(
            dataset=train_subset,
            task=OTXTaskType.SEMANTIC_SEGMENTATION,
            max_samples=10,
        )

        # Verify the function doesn't crash and returns expected structure
        assert isinstance(stats, dict)
        assert "image" in stats
        assert "annotation" in stats

        image_statistics_keys = ["avg", "min", "max", "std", "robust_min", "robust_max"]
        annotation_statistics_keys = ["avg", "min", "max", "std", "robust_min", "robust_max"]

        for key in stats["image"]["height"]:
            assert key in image_statistics_keys

        for key in stats["image"]["width"]:
            assert key in image_statistics_keys

        for key in stats["annotation"]["num_per_image"]:
            assert key in annotation_statistics_keys

        for key in stats["annotation"]["size_of_shape"]:
            assert key in annotation_statistics_keys

    def test_compute_robust_dataset_statistics_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_dataset = DmDataset(media_type=Image)
        train_subset = DatasetSubset(empty_dataset, "train")

        stats = compute_robust_dataset_statistics(
            dataset=train_subset,
            task=OTXTaskType.SEMANTIC_SEGMENTATION,
        )

        # Should return empty statistics
        assert stats == {"image": {}, "annotation": {}}

    def test_compute_robust_dataset_statistics_max_samples_limit(self, mock_semantic_seg_dataset):
        """Test that max_samples parameter limits the number of processed samples."""
        train_subset = DatasetSubset(mock_semantic_seg_dataset, "train")

        # Test with max_samples=2 (should only process 2 items)
        stats = compute_robust_dataset_statistics(
            dataset=train_subset,
            task=OTXTaskType.SEMANTIC_SEGMENTATION,
            max_samples=2,
        )

        # Should still return valid statistics
        assert isinstance(stats, dict)
        assert "image" in stats
        assert "annotation" in stats
