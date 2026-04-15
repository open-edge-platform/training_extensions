# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from getitune.data import OTXDetectionDataset, OTXInstanceSegDataset, OTXMulticlassClsDataset, OTXMultilabelClsDataset
from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from getitune.metrics.mean_ap import MaskRLEMeanAPCallable, MeanAPCallable
from getitune.metrics.types import MetricCallable
from getitune.types.task import OTXTaskType

from app.execution.common.otx_converters import (
    get_metric_by_task,
    get_otx_dataset_class_by_task_type,
    get_otx_task_type_by_task,
)
from app.models import Task, TaskType


class TestGetOtxTaskTypeByTask:
    @pytest.mark.parametrize(
        "task_type,exclusive_labels,expected",
        [
            (TaskType.CLASSIFICATION, True, OTXTaskType.MULTI_CLASS_CLS),
            (TaskType.CLASSIFICATION, False, OTXTaskType.MULTI_LABEL_CLS),
            (TaskType.DETECTION, False, OTXTaskType.DETECTION),
            (TaskType.INSTANCE_SEGMENTATION, False, OTXTaskType.INSTANCE_SEGMENTATION),
        ],
        ids=["multiclass_cls", "multilabel_cls", "detection", "instance_seg"],
    )
    def test_supported_task_types(self, task_type: TaskType, exclusive_labels: bool, expected: OTXTaskType):
        task = Task(task_type=task_type, exclusive_labels=exclusive_labels)
        assert get_otx_task_type_by_task(task) == expected

    def test_unsupported_task_type_raises(self):
        """An unknown task type must raise ValueError."""
        task = Mock(spec=Task)
        task.task_type = "semantic_segmentation"
        with pytest.raises(ValueError, match="Unsupported task type"):
            get_otx_task_type_by_task(task)


class TestGetMetricByTask:
    @pytest.mark.parametrize(
        "task_type,exclusive_labels,expected",
        [
            (TaskType.CLASSIFICATION, True, MultiClassClsMetricCallable),
            (TaskType.CLASSIFICATION, False, MultiLabelClsMetricCallable),
            (TaskType.DETECTION, False, MeanAPCallable),
            (TaskType.INSTANCE_SEGMENTATION, False, MaskRLEMeanAPCallable),
        ],
        ids=["multiclass_cls", "multilabel_cls", "detection", "instance_seg"],
    )
    def test_supported_task_types(self, task_type: TaskType, exclusive_labels: bool, expected: MetricCallable):
        task = Task(task_type=task_type, exclusive_labels=exclusive_labels)
        assert get_metric_by_task(task) is expected

    def test_unsupported_task_type_raises(self):
        """An unknown task type must raise ValueError."""
        task = Mock(spec=Task)
        task.task_type = "semantic_segmentation"
        with pytest.raises(ValueError, match="Unsupported task type"):
            get_metric_by_task(task)


class TestGetOtxDatasetClassByTaskType:
    @pytest.mark.parametrize(
        "otx_task_type,expected_class",
        [
            (OTXTaskType.MULTI_CLASS_CLS, OTXMulticlassClsDataset),
            (OTXTaskType.MULTI_LABEL_CLS, OTXMultilabelClsDataset),
            (OTXTaskType.DETECTION, OTXDetectionDataset),
            (OTXTaskType.INSTANCE_SEGMENTATION, OTXInstanceSegDataset),
        ],
        ids=["multiclass_cls", "multilabel_cls", "detection", "instance_seg"],
    )
    def test_supported_otx_task_types(self, otx_task_type: OTXTaskType, expected_class: type):
        assert get_otx_dataset_class_by_task_type(otx_task_type) is expected_class

    @pytest.mark.parametrize(
        "otx_task_type",
        [
            OTXTaskType.SEMANTIC_SEGMENTATION,
            OTXTaskType.H_LABEL_CLS,
            OTXTaskType.ROTATED_DETECTION,
            OTXTaskType.KEYPOINT_DETECTION,
        ],
        ids=["semantic_seg", "h_label_cls", "rotated_det", "keypoint_det"],
    )
    def test_unsupported_otx_task_type_raises(self, otx_task_type: OTXTaskType):
        """An OTX task type without a mapped dataset class must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported OTX task type"):
            get_otx_dataset_class_by_task_type(otx_task_type)
