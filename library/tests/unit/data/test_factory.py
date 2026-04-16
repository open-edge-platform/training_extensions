# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test Factory classes for dataset and transforms."""

from unittest.mock import MagicMock, PropertyMock

import polars as pl
import pytest
from datumaro.experimental import Dataset

from getitune.config.data import SubsetConfig
from getitune.data.dataset.classification import (
    HlabelClsDataset,
    HLabelInfo,
    MulticlassClsDataset,
    MultilabelClsDataset,
)
from getitune.data.dataset.detection import DetectionDataset
from getitune.data.dataset.instance_segmentation import InstanceSegDataset
from getitune.data.dataset.segmentation import SegmentationDataset
from getitune.data.factory import DatasetFactory, TransformLibFactory
from getitune.types.task import TaskType


class TestDatasetFactory:
    @pytest.mark.parametrize(
        ("task_type", "dataset_cls", "dm_subset_fxt_name"),
        [
            (TaskType.MULTI_CLASS_CLS, MulticlassClsDataset, "fxt_mock_classification_dm_subset"),
            (TaskType.MULTI_LABEL_CLS, MultilabelClsDataset, "fxt_mock_classification_dm_subset"),
            (TaskType.H_LABEL_CLS, HlabelClsDataset, "fxt_mock_classification_dm_subset"),
            (TaskType.DETECTION, DetectionDataset, "fxt_mock_detection_dm_subset"),
            (TaskType.ROTATED_DETECTION, InstanceSegDataset, "fxt_mock_segmentation_dm_subset"),
            (TaskType.INSTANCE_SEGMENTATION, InstanceSegDataset, "fxt_mock_segmentation_dm_subset"),
            (TaskType.SEMANTIC_SEGMENTATION, SegmentationDataset, "fxt_mock_segmentation_dm_subset"),
        ],
    )
    def test_create(
        self,
        request,
        fxt_mock_hlabelinfo,
        task_type,
        dataset_cls,
        dm_subset_fxt_name,
        mocker,
    ) -> None:
        mocker.patch.object(TransformLibFactory, "generate", return_value=None)
        dm_subset = request.getfixturevalue(dm_subset_fxt_name)
        mock_schema = mocker.MagicMock()
        mock_label = mocker.MagicMock()
        mock_label.categories.labels = []
        mock_schema.attributes = {"label": mock_label, "masks": mock_label}
        dm_subset.schema = mock_schema
        cfg_subset = mocker.MagicMock(spec=SubsetConfig)
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        mocker.patch.object(Dataset, "convert_to_schema", return_value=dm_subset)

        assert isinstance(
            DatasetFactory.create(
                task=task_type,
                dm_subset=dm_subset,
                cfg_subset=cfg_subset,
            ),
            dataset_cls,
        )


class TestDetectStorageDtype:
    """Tests for DatasetFactory._detect_storage_dtype."""

    def test_schema_uint16(self):
        """Schema-declared UInt16 dtype → 'uint16'."""
        mock_subset = MagicMock(spec=Dataset)
        # Make iteration raise StopIteration (empty dataset)
        mock_subset.__iter__ = MagicMock(return_value=iter([]))
        mock_field = MagicMock()
        mock_field.dtype = pl.UInt16
        mock_img_attr = MagicMock()
        mock_img_attr.field = mock_field
        mock_schema = MagicMock()
        mock_schema.attributes = {"image": mock_img_attr}
        type(mock_subset).schema = PropertyMock(return_value=mock_schema)

        assert DatasetFactory._detect_storage_dtype(mock_subset) == "uint16"

    def test_schema_float32(self):
        """Schema-declared Float32 dtype → 'float32'."""
        mock_subset = MagicMock(spec=Dataset)
        mock_subset.__iter__ = MagicMock(return_value=iter([]))
        mock_field = MagicMock()
        mock_field.dtype = pl.Float32
        mock_img_attr = MagicMock()
        mock_img_attr.field = mock_field
        mock_schema = MagicMock()
        mock_schema.attributes = {"image": mock_img_attr}
        type(mock_subset).schema = PropertyMock(return_value=mock_schema)

        assert DatasetFactory._detect_storage_dtype(mock_subset) == "float32"

    def test_schema_unknown_defaults_uint8(self):
        """Unknown/missing schema dtype → default 'uint8'."""
        mock_subset = MagicMock(spec=Dataset)
        mock_subset.__iter__ = MagicMock(return_value=iter([]))
        mock_schema = MagicMock()
        mock_schema.attributes = {}
        type(mock_subset).schema = PropertyMock(return_value=mock_schema)

        assert DatasetFactory._detect_storage_dtype(mock_subset) == "uint8"

    def test_file_based_detection(self, tmp_path):
        """File-header probing detects uint8 from a real PNG."""
        import numpy as np
        from PIL import Image as PILImage

        # Create a small uint8 PNG
        img_path = tmp_path / "test.png"
        PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(img_path)

        mock_media = MagicMock()
        mock_media.path = str(img_path)
        mock_item = MagicMock()
        mock_item.media = mock_media
        mock_subset = MagicMock(spec=Dataset)
        mock_subset.__iter__ = MagicMock(return_value=iter([mock_item]))

        assert DatasetFactory._detect_storage_dtype(mock_subset) == "uint8"
