# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test Factory classes for dataset and transforms."""

import pytest
from datumaro.experimental import Dataset

from otx.config.data import SubsetConfig
from otx.data.dataset.classification import (
    HLabelInfo,
    OTXHlabelClsDataset,
    OTXMulticlassClsDataset,
    OTXMultilabelClsDataset,
)
from otx.data.dataset.detection import OTXDetectionDataset
from otx.data.dataset.instance_segmentation import OTXInstanceSegDataset
from otx.data.dataset.segmentation import OTXSegmentationDataset
from otx.data.factory import OTXDatasetFactory, TransformLibFactory
from otx.data.transform_libs.torchvision import TorchVisionTransformLib
from otx.types.task import OTXTaskType
from otx.types.transformer_libs import TransformLibType

lib_type_parameters = [(TransformLibType.TORCHVISION, TorchVisionTransformLib)]


class TestTransformLibFactory:
    @pytest.mark.parametrize(
        ("lib_type", "lib"),
        lib_type_parameters,
    )
    def test_generate(self, lib_type, lib, mocker) -> None:
        mock_generate = mocker.patch.object(lib, "generate")
        config = mocker.MagicMock(spec=SubsetConfig)
        config.transform_lib_type = lib_type
        _ = TransformLibFactory.generate(config)
        mock_generate.assert_called_once_with(config)


class TestOTXDatasetFactory:
    @pytest.mark.parametrize(
        ("task_type", "dataset_cls", "dm_subset_fxt_name"),
        [
            (OTXTaskType.MULTI_CLASS_CLS, OTXMulticlassClsDataset, "fxt_mock_classification_dm_subset"),
            (OTXTaskType.MULTI_LABEL_CLS, OTXMultilabelClsDataset, "fxt_mock_classification_dm_subset"),
            (OTXTaskType.H_LABEL_CLS, OTXHlabelClsDataset, "fxt_mock_classification_dm_subset"),
            (OTXTaskType.DETECTION, OTXDetectionDataset, "fxt_mock_detection_dm_subset"),
            (OTXTaskType.ROTATED_DETECTION, OTXInstanceSegDataset, "fxt_mock_segmentation_dm_subset"),
            (OTXTaskType.INSTANCE_SEGMENTATION, OTXInstanceSegDataset, "fxt_mock_segmentation_dm_subset"),
            (OTXTaskType.SEMANTIC_SEGMENTATION, OTXSegmentationDataset, "fxt_mock_segmentation_dm_subset"),
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
            OTXDatasetFactory.create(
                task=task_type,
                dm_subset=dm_subset,
                cfg_subset=cfg_subset,
                data_format="",
            ),
            dataset_cls,
        )
