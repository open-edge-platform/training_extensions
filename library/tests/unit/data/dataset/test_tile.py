"""Unit tests for tile datasets and factory behavior."""

from __future__ import annotations

from unittest.mock import Mock

from datumaro.experimental.fields import Subset

from otx.data.dataset.tile import (
    OTXTileDatasetFactory,
    OTXTileDetTestDataset,
    OTXTileInstSegTestDataset,
    OTXTileSemanticSegTestDataset,
)
from otx.types import OTXTaskType


class DummyTileConfig:
    def __init__(self):
        self.tile_size = (128, 128)
        self.overlap = 0


class TestOTXTileDatasetFactory:
    def _make_mock_dm_subset(self, subset: Subset) -> Mock:
        mock_item = Mock()
        mock_item.subset = subset

        # dm_dataset.transform(...) is used and returns a new dm_dataset
        mock_dm = Mock()
        mock_dm.__getitem__ = Mock(return_value=mock_item)
        mock_dm.transform = Mock(return_value=mock_dm)
        mock_dm.dtype = "inmemory"
        mock_dm.__len__ = Mock(return_value=1)
        return mock_dm

    def _make_mock_otx_dataset(self, task_type: OTXTaskType, subset: Subset) -> Mock:
        mock_ds = Mock()
        mock_ds.task_type = task_type
        mock_ds.dm_subset = self._make_mock_dm_subset(subset)
        mock_ds.transforms = None
        mock_ds.max_refetch = 10
        mock_ds.stack_images = True
        mock_ds.to_tv_image = True
        # collate_fn used by OTXTileDataset for base case
        mock_ds.collate_fn = lambda x: x
        return mock_ds

    def test_create_returns_training_wrapped_dataset(self):
        dataset = self._make_mock_otx_dataset(OTXTaskType.DETECTION, Subset.TRAINING)
        cfg = DummyTileConfig()

        out = OTXTileDatasetFactory.create(dataset, cfg)

        # For training subset, factory returns the original dataset (after transforms) and assigns back dm_subset
        assert out is dataset
        # ensure tiling transform and filtering were attempted by confirming transform() called at least once
        assert dataset.dm_subset.transform.call_count >= 1

    def test_create_returns_det_tile_dataset_for_validation(self):
        dataset = self._make_mock_otx_dataset(OTXTaskType.DETECTION, Subset.VALIDATION)
        cfg = DummyTileConfig()

        out = OTXTileDatasetFactory.create(dataset, cfg)
        assert isinstance(out, OTXTileDetTestDataset)

    def test_create_returns_inst_seg_tile_dataset_for_test(self):
        dataset = self._make_mock_otx_dataset(OTXTaskType.INSTANCE_SEGMENTATION, Subset.TESTING)
        cfg = DummyTileConfig()

        out = OTXTileDatasetFactory.create(dataset, cfg)
        assert isinstance(out, OTXTileInstSegTestDataset)

    def test_create_returns_sem_seg_tile_dataset_for_validation(self):
        dataset = self._make_mock_otx_dataset(OTXTaskType.SEMANTIC_SEGMENTATION, Subset.VALIDATION)
        cfg = DummyTileConfig()

        out = OTXTileDatasetFactory.create(dataset, cfg)
        assert isinstance(out, OTXTileSemanticSegTestDataset)
