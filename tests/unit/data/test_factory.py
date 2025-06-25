# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test Factory classes for dataset and transforms."""

import pytest

from otx.config.data import SubsetConfig
from otx.data.dataset.anomaly import OTXAnomalyDataset
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
from otx.types.image import ImageColorChannel
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
        ("task_type", "dataset_cls"),
        [
            (OTXTaskType.MULTI_CLASS_CLS, OTXMulticlassClsDataset),
            (OTXTaskType.MULTI_LABEL_CLS, OTXMultilabelClsDataset),
            (OTXTaskType.H_LABEL_CLS, OTXHlabelClsDataset),
            (OTXTaskType.DETECTION, OTXDetectionDataset),
            (OTXTaskType.ROTATED_DETECTION, OTXInstanceSegDataset),
            (OTXTaskType.INSTANCE_SEGMENTATION, OTXInstanceSegDataset),
            (OTXTaskType.SEMANTIC_SEGMENTATION, OTXSegmentationDataset),
            (OTXTaskType.ANOMALY, OTXAnomalyDataset),
            (OTXTaskType.ANOMALY_CLASSIFICATION, OTXAnomalyDataset),
            (OTXTaskType.ANOMALY_DETECTION, OTXAnomalyDataset),
            (OTXTaskType.ANOMALY_SEGMENTATION, OTXAnomalyDataset),
        ],
    )
    def test_create(
        self,
        fxt_mock_hlabelinfo,
        fxt_mock_dm_subset,
        task_type,
        dataset_cls,
        mocker,
    ) -> None:
        mocker.patch.object(TransformLibFactory, "generate", return_value=None)
        cfg_subset = mocker.MagicMock(spec=SubsetConfig)
        image_color_channel = ImageColorChannel.BGR
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        assert isinstance(
            OTXDatasetFactory.create(
                task=task_type,
                dm_subset=fxt_mock_dm_subset,
                cfg_subset=cfg_subset,
                image_color_channel=image_color_channel,
                data_format="",
            ),
            dataset_cls,
        )
